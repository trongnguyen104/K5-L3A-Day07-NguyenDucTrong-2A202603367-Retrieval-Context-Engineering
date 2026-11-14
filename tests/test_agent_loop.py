"""Test offline cho vòng lặp agent: không gọi mạng, không cần DB.

Model được giả lập bằng script hành động cố định; tool là fake trả chunk cho trước.
Mục đích là kiểm tra luật của vòng lặp (giới hạn tìm kiếm, citation, abstain, lỗi),
không phải chất lượng model.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import AgentConfig, DenseSearchTool, ModelController, run_question
from src.run_agent import build_cases, summarize


def chunk(chunk_id, text='Hoàn tiền mất bảy ngày.', page=1):
    return {'id': chunk_id, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf',
            'page': page, 'metadata': {'family': 'STM32F1'}, 'score': 0.9, 'rank': 1}


class StubModel:
    """Trả lần lượt các quyết định trong script; ghi lại prompt đã nhận."""

    name = 'stub-model'

    def __init__(self, script):
        self.script = list(script)
        self.prompts = []
        self.last_usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}

    def complete_json(self, system, user):
        self.prompts.append((system, user))
        if not self.script:
            raise AssertionError('model bị gọi quá số lần script cho phép')
        return self.script.pop(0)


class FakeTool:
    name = 'fake_search'

    def __init__(self, results=None):
        self.top_k = 5
        self.results = results if results is not None else [chunk('c1'), chunk('c2', page=2)]
        self.queries = []

    def search(self, query, *, filters=None):
        self.queries.append(query)
        return self.results[:self.top_k]


class BrokenTool:
    name = 'broken_search'
    top_k = 5

    def search(self, query, *, filters=None):
        raise RuntimeError('pgvector không phản hồi')


QUESTION = {'id': 'q1', 'split': 'dev', 'question': 'Hoàn tiền mất bao lâu?',
            'expected_answer': 'bảy ngày', 'answerable': True,
            'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'bảy ngày'}]}


def run(script, *, question=None, tool=None, **kwargs):
    model = StubModel(script)
    controller = ModelController(model)
    trace = run_question(question or QUESTION, controller=controller,
                         tool=tool or FakeTool(), config=AgentConfig(**kwargs))
    return trace, controller, model


class AgentLoopTests(unittest.TestCase):
    def test_search_then_answer_with_valid_citation(self):
        trace, controller, model = run([
            {'action': 'search', 'query': 'thời gian hoàn tiền'},
            {'action': 'answer', 'answer': 'Hoàn tiền mất 7 ngày [c1].', 'citations': ['c1']},
        ])
        self.assertEqual(trace['status'], 'answered')
        self.assertEqual(trace['citations'], ['c1'])
        self.assertEqual(trace['n_searches'], 1)
        self.assertEqual([e['event'] for e in trace['events']], ['search', 'model_answer'])
        search_event = trace['events'][0]
        self.assertEqual(search_event['tool'], 'fake_search')
        self.assertEqual(search_event['query'], 'thời gian hoàn tiền')
        self.assertIn('[c1', search_event['context'])
        self.assertEqual(search_event['context_ids'], ['c1', 'c2'])
        self.assertEqual(controller.usage_total['total_tokens'], 30)

    def test_gold_never_reaches_prompt_or_tool(self):
        marked = {**QUESTION,
                  'expected_answer': 'MARKER_ANSWER_123',
                  'gold': [{'source_id': 'doc', 'page': 1,
                            'evidence': 'MARKER_EVIDENCE_456'}]}
        trace, _, model = run([
            {'action': 'search', 'query': 'hoàn tiền'},
            {'action': 'answer', 'answer': 'bảy ngày [c1]', 'citations': ['c1']},
        ], question=marked)
        prompts = ' '.join(user for _, user in model.prompts)
        for marker in ('MARKER_ANSWER_123', 'MARKER_EVIDENCE_456', 'expected_answer', 'gold'):
            self.assertNotIn(marker, prompts)
        self.assertNotIn('gold', [key for key in trace])
        self.assertIn('MARKER_ANSWER_123', json.dumps(marked))  # chỉ tồn tại ở phía nhãn
        self.assertEqual(trace['events'][0]['results'][0]['id'], 'c1')

    def test_citation_must_be_in_context(self):
        trace, _, _ = run([
            {'action': 'search', 'query': 'hoàn tiền'},
            {'action': 'answer', 'answer': 'bảy ngày [c9]', 'citations': ['c9']},
        ])
        self.assertEqual(trace['status'], 'abstain')
        self.assertEqual(trace['citations'], [])
        events = [e['event'] for e in trace['events']]
        self.assertIn('invalid_or_missing_citations', events)
        self.assertEqual(trace['events'][-1]['not_in_context'], ['c9'])

    def test_citation_id_must_appear_in_answer_text(self):
        trace, _, _ = run([
            {'action': 'search', 'query': 'hoàn tiền'},
            {'action': 'answer', 'answer': 'bảy ngày', 'citations': ['c1']},
        ])
        self.assertEqual(trace['status'], 'abstain')
        self.assertEqual(trace['events'][-1]['not_in_answer_text'], ['c1'])

    def test_model_abstain_is_recorded(self):
        trace, _, _ = run([
            {'action': 'search', 'query': 'ethernet'},
            {'action': 'abstain', 'reason': 'context không nói về Ethernet'},
        ])
        self.assertEqual(trace['status'], 'abstain')
        self.assertEqual(trace['citations'], [])
        self.assertEqual(trace['events'][-1]['event'], 'model_abstain')
        self.assertEqual(trace['n_searches'], 1)

    def test_search_limit_stops_loop(self):
        trace, _, model = run(
            [{'action': 'search', 'query': f'q{i}'} for i in range(5)],
            max_searches=2,
        )
        self.assertEqual(trace['status'], 'abstain')
        self.assertEqual(trace['n_searches'], 2)
        self.assertEqual(trace['events'][-1]['event'], 'search_limit')
        self.assertEqual(len(model.prompts), 3)  # 2 lượt tìm + 1 lượt bị chặn

    def test_context_budget_drops_chunks(self):
        big = FakeTool([chunk('c1', text='x' * 1200), chunk('c2', text='y' * 1200, page=2)])
        trace, _, _ = run([
            {'action': 'search', 'query': 'hoàn tiền'},
            {'action': 'answer', 'answer': 'bảy ngày [c1]', 'citations': ['c1']},
        ], tool=big, context_budget=1500)
        self.assertEqual(trace['events'][0]['context_ids'], ['c1'])
        self.assertEqual(trace['status'], 'answered')

    def test_tool_error_is_recorded_not_raised(self):
        trace, _, _ = run([{'action': 'search', 'query': 'x'}], tool=BrokenTool())
        self.assertEqual(trace['status'], 'error')
        self.assertIn('pgvector không phản hồi', trace['error'])
        self.assertEqual(trace['answer'], None)

    def test_invalid_action_is_error(self):
        trace, _, _ = run([{'action': 'delete'}])
        self.assertEqual(trace['status'], 'error')
        self.assertIn('action', trace['error'])

    def test_empty_query_is_error(self):
        trace, _, _ = run([{'action': 'search', 'query': '  '}])
        self.assertEqual(trace['status'], 'error')

    def test_filters_are_forwarded_to_tool(self):
        question = {**QUESTION, 'filter': {'family': 'STM32F1'}}
        tool = FakeTool()
        run([{'action': 'search', 'query': 'hoàn tiền'},
             {'action': 'answer', 'answer': 'ok [c1]', 'citations': ['c1']}],
            question=question, tool=tool)
        self.assertEqual(tool.queries, ['hoàn tiền'])


class ModelControllerTests(unittest.TestCase):
    def test_usage_delta(self):
        model = StubModel([{'action': 'abstain', 'reason': 'x'}])
        controller = ModelController(model)
        before = dict(controller.usage_total)
        controller.decide('q', 'u')
        self.assertEqual(controller.delta(before),
                         {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})
        self.assertEqual(controller.delta(controller.usage_total)['total_tokens'], 0)

    def test_missing_usage_is_ignored(self):
        model = StubModel([{'action': 'abstain', 'reason': 'x'}])
        model.last_usage = None
        controller = ModelController(model)
        controller.decide('q', 'u')
        self.assertEqual(controller.usage_total['total_tokens'], 0)


class ToolTests(unittest.TestCase):
    def test_tool_passes_top_k_and_name(self):
        class Recording:
            name = 'dense'

            def __init__(self):
                self.calls = []

            def search(self, query, *, top_k=3, filters=None):
                self.calls.append((query, top_k, filters))
                return []

        retriever = Recording()
        tool = DenseSearchTool(retriever, top_k=5, name='dense_search')
        tool.search('q', filters={'page': 1})
        self.assertEqual(retriever.calls, [('q', 5, {'page': 1})])
        self.assertEqual(tool.name, 'dense_search')


class CliHelperTests(unittest.TestCase):
    def test_build_cases_and_summarize(self):
        questions = [QUESTION, {**QUESTION, 'id': 'q2', 'answerable': False}]
        cases = build_cases(questions)
        self.assertEqual([c['answerable'] for c in cases], [True, False])

        traces = [
            {'query_id': 'q1', 'status': 'answered', 'n_searches': 2,
             'citations': ['c1'], 'elapsed_ms': 10.0, 'events': [],
             'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
            {'query_id': 'q2', 'status': 'abstain', 'n_searches': 1,
             'citations': [], 'elapsed_ms': 20.0,
             'events': [{'event': 'model_abstain'}],
             'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
            {'query_id': 'q3', 'status': 'error', 'n_searches': 0, 'citations': [],
             'elapsed_ms': 0.0, 'events': [], 'usage': {}},
        ]
        summary = summarize(traces, split='dev')
        self.assertEqual(summary['cases'], 3)
        self.assertEqual(summary['answered'], 1)
        self.assertEqual(summary['abstain'], 1)
        self.assertEqual(summary['provider_errors'], 1)
        self.assertEqual(summary['tool_use'], {'n': 2, 'N': 2})
        self.assertEqual(summary['searches_per_case'], {'1': 1, '2': 1})
        self.assertEqual(summary['citations'], {'total': 1, 'answers_with_citation': 1})
        self.assertEqual(summary['usage']['total_tokens'], 30)
        self.assertEqual(summary['median_elapsed_ms'], 15.0)

    def test_cli_requires_model_key(self):
        env = {k: v for k, v in os.environ.items()
               if k not in {'LLM_API_KEY', 'OPENAI_API_KEY'}}
        result = subprocess.run(
            [sys.executable, '-m', 'src.run_agent',
             '--questions', 'data/benchmark_questions.json',
             '--split', 'dev', '--output', 'src/results/_no_key'],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8',
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn('LLM_API_KEY', result.stderr)


if __name__ == '__main__':
    unittest.main()

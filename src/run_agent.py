"""CLI chạy agent Day 7 trên dev hoặc test và lưu trace đầy đủ.

Ví dụ:
  python -m src.run_agent --questions data/benchmark_questions.json \
      --split dev --output src/results/agent-dev

Cần model thật: `LLM_MODEL`, `LLM_API_BASE`, và `LLM_API_KEY` (hoặc `OPENAI_API_KEY`).
Sau khi chạy, chấm nhãn thủ công vào `reviews.json` rồi tổng hợp bằng
`starter/agent_metrics.py` (xem src/README.md).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .agent import AgentConfig, DenseSearchTool, ModelController, run_question
from .chunks import load_questions
from .config import ChatConfig, RetrievalConfig
from .day6_store import Day6StoreError
from .embeddings import EmbeddingError
from .models import ChatModel, ModelError
from .run_retrieval import build_retriever


def build_cases(questions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """cases.json cho starter/agent_metrics.py: id/split/answerable."""
    return [
        {'id': question['id'], 'split': question['split'],
         'answerable': question.get('answerable', True)}
        for question in questions
    ]


def summarize(traces: Sequence[dict[str, Any]], *, split: str) -> dict[str, Any]:
    ok = [trace for trace in traces if trace['status'] != 'error']
    answered = [trace for trace in ok if trace['status'] == 'answered']
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    for trace in traces:
        for key, value in (trace.get('usage') or {}).items():
            if key in usage and isinstance(value, int):
                usage[key] += value
    return {
        'split': split,
        'cases': len(traces),
        'answered': len(answered),
        'abstain': sum(1 for trace in ok if trace['status'] == 'abstain'),
        'provider_errors': len(traces) - len(ok),
        'tool_use': {
            'n': sum(1 for trace in ok if trace.get('n_searches', 0) > 0),
            'N': len(ok),
        },
        'searches_per_case': {
            str(searches): sum(1 for trace in ok if trace.get('n_searches', 0) == searches)
            for searches in sorted({trace.get('n_searches', 0) for trace in ok})
        },
        'max_searches_reached': sum(
            1 for trace in ok
            if any(event.get('event') == 'search_limit' for event in trace.get('events', []))
        ),
        'invalid_citation_cases': sum(
            1 for trace in ok
            if any(event.get('event') == 'invalid_or_missing_citations'
                   for event in trace.get('events', []))
        ),
        'citations': {
            'total': sum(len(trace.get('citations') or []) for trace in answered),
            'answers_with_citation': sum(1 for trace in answered if trace.get('citations')),
        },
        'median_elapsed_ms': statistics.median(
            [trace.get('elapsed_ms', 0.0) for trace in ok]
        ) if ok else 0.0,
        'usage': usage,
        'note': 'Grounded answer 0/1/2, citation support và abstention cần nhãn thủ công '
                '(reviews.json); script này chỉ tổng hợp trace.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Chạy agent Day 7 (model thật) trên một split')
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', choices=('dense', 'bm25', 'hybrid'), default='dense',
                        help='Phương pháp retrieval đã chốt trên dev là dense')
    parser.add_argument('--store', choices=('day6', 'memory'), default='day6')
    parser.add_argument('--chunks', type=Path, help='JSON chunk export cho --store memory')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--budget', type=int, default=1800)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None, help='Chỉ chạy N câu đầu (smoke test)')
    args = parser.parse_args()

    questions = load_questions(args.questions, split=args.split)
    if args.limit:
        questions = questions[:args.limit]

    # Model được dựng trước để báo lỗi thiếu key rõ ràng, không cần chạm tới DB.
    try:
        model = ChatModel(ChatConfig.from_env())
    except ModelError as error:
        print(f'Không dựng được model: {error}', file=sys.stderr)
        raise SystemExit(2) from error
    controller = ModelController(model)
    try:
        retriever = build_retriever(args.method, args, RetrievalConfig.from_env())
    except (Day6StoreError, EmbeddingError) as error:
        print(f'Không dựng được retriever: {error}', file=sys.stderr)
        raise SystemExit(2) from error
    tool = DenseSearchTool(retriever, top_k=args.top_k, name=f'{args.method}_search')
    config = AgentConfig(top_k=args.top_k, context_budget=args.budget,
                         max_searches=args.max_searches)

    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, question in enumerate(questions, start=1):
        before = dict(controller.usage_total)
        trace = run_question(question, controller=controller, tool=tool, config=config)
        trace['usage'] = controller.delta(before)
        trace['model'] = model.name
        traces.append(trace)
        print(f"[{index}/{len(questions)}] {question['id']} {trace['status']} "
              f"search={trace.get('n_searches')} cite={len(trace.get('citations') or [])}",
              file=sys.stderr)

    summary = summarize(traces, split=args.split)
    summary['method'] = args.method
    summary['model'] = model.name
    summary['config'] = {'top_k': args.top_k, 'context_budget': args.budget,
                         'max_searches': args.max_searches}
    summary['elapsed_seconds'] = time.perf_counter() - started

    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ('traces.json', traces),
        ('cases.json', build_cases(questions)),
        ('summary.json', summary),
    ):
        (args.output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

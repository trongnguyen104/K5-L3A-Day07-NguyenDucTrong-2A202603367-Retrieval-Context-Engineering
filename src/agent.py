"""Agent Day 7: model thật điều phối tool dense retrieval, trả lời có citation.

Cấu hình đã chốt trên dev: dense trên pgvector Day 6, top_k=5, context budget 1800.
Agent chỉ nhận context từ tool, không bao giờ thấy gold. Mọi tool call, tham số,
kết quả, context gửi model, câu trả lời, citation và lỗi đều được ghi vào trace.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .context import pack_context
from .retrieval import Retriever

SYSTEM_PROMPT = """Bạn là agent trả lời câu hỏi về datasheet vi điều khiển STM32F103x8/B.

Bạn có một tool: dense_search(query) — tìm các đoạn tài liệu liên quan và trả về
đoạn văn kèm ID dạng [chunk-id | source | page N].

Quy tắc bắt buộc:
1. Chỉ dùng thông tin trong CONTEXT được tool trả về. Không dùng kiến thức bên ngoài.
2. Nếu CONTEXT chưa đủ, gọi dense_search với truy vấn khác (tối đa 4 lần).
3. Trả lời ngắn gọn bằng tiếng Việt, và với mỗi khẳng định phải kèm citation dạng
   [chunk-id] lấy đúng từ CONTEXT. Không bịa ID.
4. Nếu sau khi tìm vẫn không có evidence cho câu trả lời, KHÔNG được đoán: trả
   action "abstain" kèm lý do.

Chỉ trả về một JSON object, không kèm giải thích ngoài JSON:
- Tìm tài liệu: {"action": "search", "query": "<truy vấn tiếng Việt>"}
- Trả lời: {"action": "answer", "answer": "<câu trả lời có [chunk-id]>", "citations": ["<chunk-id>"]}
- Không đủ bằng chứng: {"action": "abstain", "reason": "<thiếu gì>"}"""


class SearchTool(Protocol):
    name: str

    def search(self, query: str, *, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


class DenseSearchTool:
    """Tool retrieval duy nhất, mặc định dùng cấu hình đã chốt (dense, top_k=5)."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        top_k: int = 5,
        name: str = 'dense_search',
    ):
        if top_k < 1:
            raise ValueError('top_k phải >= 1')
        self.retriever = retriever
        self.top_k = top_k
        self.name = name

    def search(self, query: str, *, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.retriever.search(query, top_k=self.top_k, filters=filters)


class ModelController:
    """Bọc chat model thật: gọi model, cộng dồn token, trả JSON hành động.

    Không có fallback: lỗi provider lan ra ngoài để `run_question` ghi
    `status: "error"` thay vì bịa câu trả lời.
    """

    def __init__(self, model: Any, *, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.system_prompt = system_prompt
        self.name = getattr(model, 'name', 'chat-model')
        self.usage_total = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    def decide(self, question: str, user_message: str) -> dict[str, Any]:
        payload = self.model.complete_json(self.system_prompt, user_message)
        if not isinstance(payload, dict):
            raise ValueError('controller phải trả JSON object')
        self._add_usage()
        return payload

    def delta(self, before: dict[str, int]) -> dict[str, int]:
        return {key: value - before.get(key, 0) for key, value in self.usage_total.items()}

    def _add_usage(self) -> None:
        usage = getattr(self.model, 'last_usage', None) or {}
        for key in self.usage_total:
            value = usage.get(key)
            if isinstance(value, int):
                self.usage_total[key] += value


@dataclass
class AgentConfig:
    top_k: int = 5
    context_budget: int = 1800
    max_searches: int = 4


def render_user_message(
    question: str,
    context: str,
    search_log: Sequence[dict[str, Any]],
    *,
    max_searches: int,
) -> str:
    history = '\n'.join(
        f'- lần {index + 1}: {item["query"]} → {len(item["ids"])} đoạn'
        for index, item in enumerate(search_log)
    ) or '(chưa tìm lần nào)'
    return (
        f'CÂU HỎI: {question}\n\n'
        f'LỊCH SỬ TÌM KIẾM ({len(search_log)}/{max_searches}):\n{history}\n\n'
        f'CONTEXT:\n{context or "(rỗng)"}'
    )


def run_question(
    question: dict[str, Any],
    *,
    controller: Any,
    tool: SearchTool,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Chạy một câu; trả trace. Không đưa gold vào prompt hay tool."""
    config = config or AgentConfig()
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    search_log: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    context, context_ids = '', []
    trace: dict[str, Any] = {
        'query_id': question['id'],
        'split': question['split'],
        'question': question['question'],
        'answerable': question.get('answerable', True),
        'model': getattr(controller, 'name', 'unknown'),
        'tool': tool.name,
        'config': {'top_k': tool.top_k, 'context_budget': config.context_budget,
                   'max_searches': config.max_searches},
        'events': events,
        'citations': [],
    }
    if not question['question'].strip():
        raise ValueError('câu hỏi rỗng')
    filters = question.get('filter')

    try:
        while True:
            decision = controller.decide(
                question['question'],
                render_user_message(question['question'], context, search_log,
                                    max_searches=config.max_searches),
            )
            action = decision.get('action')
            if action == 'search':
                if len(search_log) >= config.max_searches:
                    events.append({'event': 'search_limit', 'searches': len(search_log)})
                    return _finish(trace, 'abstain', 'Đã đạt giới hạn tìm kiếm mà chưa đủ bằng chứng.',
                                   [], events, started, config)
                query = decision.get('query')
                if not isinstance(query, str) or not query.strip():
                    raise ValueError('model trả query rỗng')
                t0 = time.perf_counter()
                results = tool.search(query.strip(), filters=filters)
                latency_ms = (time.perf_counter() - t0) * 1000
                candidates.extend(results)
                context, context_ids = pack_context(candidates, config.context_budget)
                chunks = {c['id']: c for c in candidates if c['id'] in context_ids}
                search_log.append({'query': query.strip(), 'ids': list(context_ids)})
                events.append({
                    'event': 'search',
                    'tool': tool.name,
                    'query': query.strip(),
                    'filter': filters,
                    'results': [
                        {'id': r['id'], 'source': r['source'], 'page': r['page'],
                         'score': round(r.get('score', 0.0), 4), 'rank': r.get('rank')}
                        for r in results
                    ],
                    'context': context,
                    'context_ids': list(context_ids),
                    'latency_ms': latency_ms,
                })
                continue

            if action == 'abstain':
                events.append({'event': 'model_abstain', 'reason': decision.get('reason', ''),
                               'raw': decision})
                return _finish(trace, 'abstain', 'Không đủ bằng chứng để trả lời.', [],
                               events, started, config)

            if action == 'answer':
                answer = decision.get('answer')
                citations = decision.get('citations')
                events.append({'event': 'model_answer', 'raw': decision})
                if not isinstance(answer, str) or not answer.strip() or not isinstance(citations, list):
                    raise ValueError('answer/citations không hợp lệ')
                citations = [str(item) for item in citations]
                invalid = [item for item in citations if item not in chunks]
                unsupported_format = [item for item in citations if f'[{item}]' not in answer]
                if not citations or invalid or unsupported_format:
                    events.append({
                        'event': 'invalid_or_missing_citations',
                        'not_in_context': invalid,
                        'not_in_answer_text': unsupported_format,
                    })
                    return _finish(trace, 'abstain',
                                   'Không đủ bằng chứng để trả lời.', [], events, started, config)
                return _finish(trace, 'answered', answer.strip(), citations, events, started, config)

            raise ValueError(f'model trả action không hợp lệ: {action!r}')
    except Exception as error:  # noqa: BLE001 - lỗi phải vào trace, không thay bằng kết quả giả
        trace.update({
            'status': 'error',
            'error': f'{type(error).__name__}: {error}',
            'answer': None,
            'citations': [],
            'n_searches': sum(1 for e in events if e['event'] == 'search'),
            'elapsed_ms': (time.perf_counter() - started) * 1000,
            'context': context,
            'context_ids': list(context_ids),
        })
        return trace


def _finish(
    trace: dict[str, Any],
    status: str,
    answer: str,
    citations: list[str],
    events: list[dict[str, Any]],
    started: float,
    config: AgentConfig,
) -> dict[str, Any]:
    trace.update({
        'status': status,
        'answer': answer,
        'citations': citations,
        'n_searches': sum(1 for e in events if e['event'] == 'search'),
        'elapsed_ms': (time.perf_counter() - started) * 1000,
    })
    return trace

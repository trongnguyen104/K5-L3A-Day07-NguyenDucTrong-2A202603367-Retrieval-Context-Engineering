"""Đánh giá retrieval trên một split: Hit@1, Evidence Hit@3, Context Evidence Rate.

Câu `answerable: false` vẫn được chạy qua retriever để lấy trace nhưng không vào
mẫu số của retrieval metrics (theo BENCHMARK.md).
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Sequence

from .chunks import evidence_in_chunk
from .context import pack_context
from .retrieval import Retriever


def gold_hit(chunk: dict[str, Any], gold: Sequence[dict[str, Any]]) -> bool:
    return any(evidence_in_chunk(chunk, item) for item in gold)


def run_split(
    retriever: Retriever,
    questions: Sequence[dict[str, Any]],
    *,
    split: str,
    top_k: int = 3,
    context_budget: int = 800,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chạy retriever cho từng câu; trả (traces, summary).

    Metrics luôn tính trên top-3 bất kể top_k của pipeline, để so sánh công bằng.
    Lỗi của retriever được ghi vào trace với `status: "error"` và báo riêng,
    không thay bằng kết quả giả.
    """
    if top_k < 1:
        raise ValueError('top_k phải >= 1')
    selected = [question for question in questions if question.get('split') == split]
    if not selected:
        raise ValueError(f'Không có câu hỏi nào cho split {split}')
    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    for question in selected:
        answerable = question.get('answerable', True)
        use_filters = question.get('filter')
        metrics_k = max(top_k, 3)
        query_started = time.perf_counter()
        try:
            results = retriever.search(
                question['question'], top_k=metrics_k, filters=use_filters
            )
        except Exception as error:  # noqa: BLE001 - lỗi phải vào trace, không nuốt mất
            traces.append({
                'query_id': question['id'],
                'split': split,
                'question': question['question'],
                'answerable': answerable,
                'filter': use_filters,
                'method': getattr(retriever, 'name', 'unknown'),
                'status': 'error',
                'error': f'{type(error).__name__}: {error}',
                'results': [],
                'context': '',
                'context_ids': [],
            })
            continue
        elapsed_ms = (time.perf_counter() - query_started) * 1000
        context, context_ids = pack_context(results, context_budget)
        gold = question.get('gold') or []
        marked = [
            {
                **result,
                'hit': gold_hit(result, gold) if answerable else None,
            }
            for result in results
        ]
        context_hit = any(
            chunk['hit'] and chunk['id'] in context_ids for chunk in marked
        )
        traces.append({
            'query_id': question['id'],
            'split': split,
            'question': question['question'],
            'answerable': answerable,
            'filter': use_filters,
            'method': getattr(retriever, 'name', 'unknown'),
            'status': 'ok',
            'top_k': metrics_k,
            'results': marked,
            'context': context,
            'context_ids': context_ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']) if answerable else None,
            'hit3': any(chunk['hit'] for chunk in marked[:3]) if answerable else None,
            'context_hit': context_hit if answerable else None,
        })

    answerable_traces = [t for t in traces if t['answerable'] and t['status'] == 'ok']
    errors = [t for t in traces if t['status'] == 'error']
    total = len(answerable_traces)
    summary = {
        'method': getattr(retriever, 'name', 'unknown'),
        'split': split,
        'n_answerable': sum(1 for t in traces if t['answerable']),
        'n_unanswerable': sum(1 for t in traces if not t['answerable']),
        'n_scored': total,
        'errors': len(errors),
        'hit1': sum(1 for t in answerable_traces if t['hit1']),
        'hit3': sum(1 for t in answerable_traces if t['hit3']),
        'context_evidence': sum(1 for t in answerable_traces if t['context_hit']),
        'hit1_rate': _rate(sum(1 for t in answerable_traces if t['hit1']), total),
        'hit3_rate': _rate(sum(1 for t in answerable_traces if t['hit3']), total),
        'context_evidence_rate': _rate(
            sum(1 for t in answerable_traces if t['context_hit']), total
        ),
        'avg_context_chars': statistics.mean(
            [t['context_chars'] for t in answerable_traces]
        ) if answerable_traces else 0,
        'median_ms': _median([t['elapsed_ms'] for t in traces if t['status'] == 'ok']),
        'elapsed_seconds': time.perf_counter() - started,
        'top_k': top_k,
        'context_budget': context_budget,
    }
    return traces, summary


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0

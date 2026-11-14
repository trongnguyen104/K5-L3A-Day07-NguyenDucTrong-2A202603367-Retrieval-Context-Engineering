"""Hybrid retrieval: hợp nhất dense + lexical bằng Reciprocal Rank Fusion."""
from __future__ import annotations

from typing import Any, Sequence

from . import Retriever, normalize, validate_filters


class HybridRRF:
    """RRF: score(d) = Σ weight_i / (k + rank_i(d)).

    RRF chỉ dùng thứ hạng nên không phải chuẩn hoá score giữa hai retriever;
    mỗi retriever chạy với candidate_k riêng trước khi hợp nhất.
    """

    name = 'hybrid-rrf'

    def __init__(
        self,
        retrievers: Sequence[tuple[Retriever, float]],
        *,
        k: int = 60,
        candidate_k: int = 10,
    ):
        if not retrievers:
            raise ValueError('HybridRRF cần ít nhất một retriever')
        if k < 1 or candidate_k < 1:
            raise ValueError('k và candidate_k phải >= 1')
        if all(weight <= 0 for _, weight in retrievers):
            raise ValueError('cần ít nhất một weight > 0')
        self.retrievers = list(retrievers)
        self.k = k
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filters = validate_filters(filters)
        if not query.strip():
            return []
        fused: dict[str, dict[str, Any]] = {}
        for retriever, weight in self.retrievers:
            if weight <= 0:
                continue
            results = retriever.search(
                query, top_k=max(self.candidate_k, top_k), filters=filters
            )
            for rank, result in enumerate(results, start=1):
                entry = fused.get(result['id'])
                if entry is None:
                    entry = {**result, 'score': 0.0, 'ranks': {}}
                    fused[result['id']] = entry
                entry['score'] += weight / (self.k + rank)
                entry['ranks'][retriever.name] = rank
                entry['rank_weights'] = entry.get('rank_weights', {})
                entry['rank_weights'][retriever.name] = weight
        merged = sorted(fused.values(), key=lambda item: (-item['score'], item['id']))
        return normalize(merged, method=self.name, top_k=top_k)

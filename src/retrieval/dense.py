"""Dense retrieval trên vector store Day 6 (pgvector) hoặc store in-memory.

Store in-memory chỉ dùng khi có sẵn chunk export; metric vẫn là dense cosine
với embedding thật, không phải vector ngẫu nhiên.
"""
from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

from ..embeddings import Embedder
from . import matches_filters, normalize, validate_filters


class VectorStore(Protocol):
    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f'độ dài vector khác nhau: {len(left)} vs {len(right)}')
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class MemoryVectorStore:
    """Store trong RAM cho một tập chunk đã export; embed một lần khi khởi tạo."""

    def __init__(self, chunks: Sequence[dict[str, Any]], embedder: Embedder):
        if not chunks:
            raise ValueError('MemoryVectorStore cần ít nhất một chunk')
        self.chunks = list(chunks)
        self.embedder = embedder
        vectors = embedder.embed_documents([chunk['text'] for chunk in self.chunks])
        if len(vectors) != len(self.chunks):
            raise ValueError('số vector khác số chunk')
        self.vectors = [list(vector) for vector in vectors]

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filters = validate_filters(filters)
        scored: list[dict[str, Any]] = []
        for chunk, doc_vector in zip(self.chunks, self.vectors):
            if filters and not matches_filters(chunk, filters):
                continue
            scored.append({
                'id': chunk['id'],
                'text': chunk['text'],
                'source_id': chunk['source_id'],
                'source': chunk['source'],
                'page': chunk['page'],
                'metadata': chunk.get('metadata') or {},
                'score': cosine_similarity(vector, doc_vector),
            })
        scored.sort(key=lambda item: (-item['score'], item['id']))
        return scored[:top_k]


class DenseRetriever:
    """Embed query bằng model cấu hình qua env rồi truy vấn store tương thích."""

    name = 'dense'

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

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
        vector = self.embedder.embed_query(query)
        results = self.store.search(vector, top_k=top_k, filters=filters)
        return normalize(results, method=self.name, top_k=top_k, filters=filters)

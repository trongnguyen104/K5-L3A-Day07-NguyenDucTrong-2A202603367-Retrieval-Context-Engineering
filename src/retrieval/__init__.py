"""Giao diện chung cho mọi retriever và chuẩn hoá kết quả."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

# Filter chỉ dựa trên thông tin người dùng cung cấp hoặc suy ra từ query,
# không bao giờ lấy từ gold.
CHUNK_FILTER_FIELDS = ('id', 'source_id', 'source', 'page')
METADATA_FILTER_KEY = 'metadata'


class Retriever(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


def validate_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chuẩn hoá filter: cột chunk (id/source_id/source/page) hoặc khoá metadata.

    `{"metadata": {"version": "2025"}}` và `{"version": "2025"}` tương đương.
    Khoá rỗng hoặc filter không phải dict bị từ chối thay vì bị bỏ qua im lặng.
    """
    if not filters:
        return None
    if not isinstance(filters, dict):
        raise ValueError('filter phải là object')
    normalized: dict[str, Any] = {}
    for key, value in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError('khoá filter phải là chuỗi không rỗng')
        if key == METADATA_FILTER_KEY:
            if not isinstance(value, dict) or not value:
                raise ValueError('filter metadata phải là object không rỗng')
            normalized[key] = dict(value)
        else:
            normalized[key] = value
    return normalized


def matches_filters(result: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    metadata = result.get('metadata') or {}
    for key, value in filters.items():
        if key == METADATA_FILTER_KEY:
            for meta_key, meta_value in value.items():
                if meta_key not in metadata or str(metadata[meta_key]) != str(meta_value):
                    return False
        elif key in CHUNK_FILTER_FIELDS:
            if result.get(key) != value:
                return False
        elif key not in metadata or str(metadata[key]) != str(value):
            return False
    return True


def normalize(
    results: Sequence[dict[str, Any]],
    *,
    method: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Sắp theo score giảm dần, gán rank 1..n, lọc trùng ID và filter."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: (-item.get('score', 0.0), item['id'])):
        if result['id'] in seen or not matches_filters(result, filters):
            continue
        seen.add(result['id'])
        unique.append(result)
    trimmed = unique[:top_k]
    normalized = []
    for rank, result in enumerate(trimmed, start=1):
        item = {**result}
        item.update({
            'metadata': result.get('metadata') or {},
            'score': float(result.get('score', 0.0)),
            'rank': rank,
            'method': method,
        })
        normalized.append(item)
    return normalized

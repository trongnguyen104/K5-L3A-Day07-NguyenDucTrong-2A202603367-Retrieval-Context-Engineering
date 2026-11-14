"""Chọn context từ kết quả retrieval dưới một budget ký tự.

Dedup theo ID chunk và giữ nguyên nguồn/trang trong header để kiểm chứng được.
Chunk không vừa budget bị bỏ (không cắt giữa chừng) để evidence không bị mất.
"""
from __future__ import annotations

from typing import Any, Sequence


def render_block(chunk: dict[str, Any]) -> str:
    return f"[{chunk['id']} | {chunk['source']} | page {chunk['page']}]\n{chunk['text']}"


def pack_context(
    results: Sequence[dict[str, Any]], budget: int = 800
) -> tuple[str, list[str]]:
    if budget < 1:
        raise ValueError('budget phải >= 1')
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in results:
        if chunk['id'] in seen:
            continue
        seen.add(chunk['id'])
        block = render_block(chunk)
        if len('\n\n'.join([*blocks, block])) <= budget:
            blocks.append(block)
            ids.append(chunk['id'])
    context = '\n\n'.join(blocks)
    return context, ids

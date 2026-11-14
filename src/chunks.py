"""Đọc và kiểm tra chunk/câu hỏi theo schema trong data/README.md."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CHUNK_KEYS = ('id', 'text', 'source_id', 'source', 'page', 'metadata')
STRING_KEYS = ('id', 'text', 'source_id', 'source')


def load_json(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError(f'{path} phải là JSON array')
    return data


def validate_chunk(chunk: dict[str, Any]) -> None:
    if not isinstance(chunk, dict):
        raise ValueError('chunk phải là object')
    for key in CHUNK_KEYS:
        if key not in chunk or chunk[key] is None:
            raise ValueError(f'chunk thiếu {key}')
    if not all(isinstance(chunk[key], str) and chunk[key].strip() for key in STRING_KEYS):
        raise ValueError('chunk id/text/source_id/source phải là chuỗi không rỗng')
    if not isinstance(chunk['page'], (int, str)):
        raise ValueError('chunk page phải là số hoặc chuỗi')
    if not isinstance(chunk['metadata'], dict):
        raise ValueError('chunk metadata phải là object')


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunks = load_json(path)
    if not chunks:
        raise ValueError('chunks phải là JSON array không rỗng')
    seen: set[str] = set()
    for chunk in chunks:
        validate_chunk(chunk)
        if chunk['id'] in seen:
            raise ValueError(f"chunk ID trùng: {chunk['id']}")
        seen.add(chunk['id'])
    return chunks


def load_questions(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    questions = load_json(path)
    if not questions:
        raise ValueError('questions phải là JSON array không rỗng')
    seen: set[str] = set()
    selected = []
    for question in questions:
        qid = question.get('id')
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError('question id phải là chuỗi không rỗng và duy nhất')
        seen.add(qid)
        if question.get('split') not in ('dev', 'test'):
            raise ValueError(f'{qid}: split phải là dev hoặc test')
        if not isinstance(question.get('question'), str) or not question['question'].strip():
            raise ValueError(f'{qid}: question không hợp lệ')
        answerable = question.get('answerable', True)
        if answerable:
            if not isinstance(question.get('expected_answer'), str) or not question['expected_answer'].strip():
                raise ValueError(f'{qid}: expected_answer phải là chuỗi không rỗng')
            gold = question.get('gold')
            if not isinstance(gold, list) or not gold:
                raise ValueError(f'{qid}: gold phải là danh sách không rỗng')
            for item in gold:
                if not all(key in item and item[key] is not None for key in ('source_id', 'page', 'evidence')):
                    raise ValueError(f'{qid}: gold thiếu source_id/page/evidence')
                if not isinstance(item['evidence'], str) or not item['evidence'].strip():
                    raise ValueError(f'{qid}: evidence phải là chuỗi không rỗng')
        elif question.get('expected_behavior') != 'search_and_abstain':
            raise ValueError(f'{qid}: câu thiếu evidence cần expected_behavior="search_and_abstain"')
        if split is None or question['split'] == split:
            selected.append(question)
    return selected


def normalize_text(text: str) -> str:
    """Gộp khoảng trắng về một dấu cách để so khớp evidence trong PDF ngắt dòng."""
    return re.sub(r'\s+', ' ', text).strip()


def evidence_in_chunk(chunk: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Evidence phải nằm trọn trong chunk đúng nguồn/trang (bỏ qua ngắt dòng PDF)."""
    return (
        chunk['source_id'] == gold['source_id']
        and chunk['page'] == gold['page']
        and normalize_text(gold['evidence']) in normalize_text(chunk['text'])
    )


def check_gold_against_chunks(chunks: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    """Bảo đảm mỗi evidence nằm trọn trong một chunk đúng nguồn và trang."""
    for question in questions:
        for gold in question.get('gold') or []:
            if not any(evidence_in_chunk(chunk, gold) for chunk in chunks):
                raise ValueError(
                    f"{question['id']}: evidence không nằm trọn trong chunk đúng nguồn/trang "
                    f"({gold['source_id']}, page {gold['page']})"
                )

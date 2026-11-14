"""Tái tạo chunk Day 6 từ PDF khi không truy cập được PostgreSQL/pgvector.

Chỉ dùng để kiểm tra gold và chạy BM25/hybrid offline trên đúng văn bản Day 6.
Tham số khớp Day 6: fixed size 500/overlap 50 và by_paragraph max 500, ID giữ
nguyên định dạng `{document_id}_p{page}_...`. Kết quả KHÔNG thay vector store
thật: khi PostgreSQL chạy, dùng `--store day6`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def extract_pages(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - phụ thuộc môi trường
        raise SystemExit('Cần pypdf để đọc PDF: pip install pypdf') from error
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or '') for page in reader.pages]


def fixed_size(
    text: str,
    *,
    document_id: str,
    source: str,
    page: int,
    metadata: dict[str, Any],
    size: int = 500,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError('size phải > 0 và 0 <= overlap < size')
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if not piece.strip():
            continue
        chunks.append({
            'id': f'{document_id}_p{page}_fixed_{start}',
            'text': piece,
            'source_id': document_id,
            'source': source,
            'page': page,
            'metadata': {**metadata, 'chunking_strategy': 'fixed',
                         'start_char': start, 'end_char': start + len(piece)},
        })
        if start + size >= len(text):
            break
    return chunks


def by_paragraph(
    text: str,
    *,
    document_id: str,
    source: str,
    page: int,
    metadata: dict[str, Any],
    max_size: int = 500,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    paragraphs = [part.strip() for part in text.split('\n\n') if part.strip()]
    current = ''
    index = 0

    def save(value: str) -> None:
        nonlocal index
        if not value.strip():
            return
        chunks.append({
            'id': f'{document_id}_p{page}_paragraph_{index}',
            'text': value,
            'source_id': document_id,
            'source': source,
            'page': page,
            'metadata': {**metadata, 'chunking_strategy': 'paragraph'},
        })
        index += 1

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                save(current)
                current = ''
            for start in range(0, len(paragraph), max_size):
                save(paragraph[start:start + max_size])
            continue
        candidate = f'{current}\n\n{paragraph}' if current else paragraph
        if len(candidate) <= max_size:
            current = candidate
        else:
            save(current)
            current = paragraph
    if current:
        save(current)
    return chunks


def build_chunks(
    pages: list[str],
    *,
    document_id: str,
    source: str,
    metadata: dict[str, Any],
    strategy: str = 'fixed',
    first_page: int = 1,
    last_page: int | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    builder = fixed_size if strategy == 'fixed' else by_paragraph
    for offset, text in enumerate(pages):
        page = first_page + offset
        if last_page is not None and page > last_page:
            break
        if not text.strip():
            continue
        chunks.extend(
            builder(text, document_id=document_id, source=source,
                    page=page, metadata=metadata)
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description='Tạo chunks JSON từ PDF Day 6 (offline)')
    parser.add_argument('--pdf', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--document-id', default=os.getenv('DAY6_DOCUMENT_ID', 'stm32f103c8'))
    parser.add_argument('--strategy', choices=('fixed', 'paragraph'), default=os.getenv('DAY6_STRATEGY', 'fixed'))
    parser.add_argument('--first-page', type=int, default=int(os.getenv('DAY6_FIRST_PAGE', '1')))
    parser.add_argument('--last-page', type=int, default=int(os.getenv('DAY6_LAST_PAGE', '0')) or None,
                        help='Trang cuối của content_pages trong manifest Day 6')
    args = parser.parse_args()
    metadata = {
        'device': 'STM32F103C8',
        'family': 'STM32F1',
        'manufacturer': 'STMicroelectronics',
    }
    chunks = build_chunks(
        extract_pages(args.pdf),
        document_id=args.document_id,
        source=args.pdf.name,
        metadata=metadata,
        strategy=args.strategy,
        first_page=args.first_page,
        last_page=args.last_page,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'chunks': len(chunks),
                      'strategy': args.strategy}, ensure_ascii=False))


if __name__ == '__main__':
    main()

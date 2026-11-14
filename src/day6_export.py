"""Export chunk văn bản từ PostgreSQL Day 6 ra JSON theo schema data/README.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DatabaseConfig
from .day6_store import Day6StoreError, PgVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description='Export chunk Day 6 (pgvector) ra JSON')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    try:
        with PgVectorStore(DatabaseConfig.from_env()) as store:
            chunks = store.fetch_chunks(limit=args.limit)
    except Day6StoreError as error:
        raise SystemExit(f'Lỗi kết nối Day 6: {error}') from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'chunks': len(chunks)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

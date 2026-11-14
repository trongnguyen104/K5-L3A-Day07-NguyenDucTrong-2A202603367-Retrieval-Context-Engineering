"""CLI chạy một phương pháp retrieval trên một split và ghi traces + summary.

Ví dụ:
  python -m src.run_retrieval --chunks data/chunks.json --questions data/benchmark_questions.json \
      --method bm25 --split dev --output src/results/dev-bm25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .chunks import check_gold_against_chunks, load_chunks, load_questions
from .config import DatabaseConfig, EmbeddingConfig, RetrievalConfig
from .day6_store import Day6StoreError, PgVectorStore
from .embeddings import EmbeddingError, build_embedder
from .evaluate import run_split
from .retrieval import Retriever
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever, MemoryVectorStore
from .retrieval.hybrid import HybridRRF


def build_retriever(method: str, args: argparse.Namespace, retrieval: RetrievalConfig) -> Retriever:
    if args.store == 'memory':
        if not args.chunks:
            raise SystemExit('--store memory cần --chunks (JSON export)')
        chunks = load_chunks(args.chunks)
    else:
        try:
            with PgVectorStore(DatabaseConfig.from_env()) as store:
                chunks = store.fetch_chunks()
        except Day6StoreError as error:
            raise SystemExit(f'Lỗi kết nối Day 6: {error}') from error
    bm25 = BM25Retriever(
        chunks,
        k1=retrieval.bm25_k1,
        b=retrieval.bm25_b,
        fold_accents=retrieval.bm25_fold_accents,
    )
    if method == 'bm25':
        return bm25
    embedder = build_embedder(EmbeddingConfig.from_env())
    if args.store == 'memory':
        store = MemoryVectorStore(chunks, embedder)
    else:
        store = PgVectorStore(DatabaseConfig.from_env())
    dense = DenseRetriever(store, embedder)
    if method == 'dense':
        return dense
    return HybridRRF(
        [(dense, retrieval.dense_weight), (bm25, retrieval.lexical_weight)],
        k=retrieval.rrf_k,
        candidate_k=retrieval.hybrid_candidate_k,
    )


def main() -> None:
    retrieval = RetrievalConfig.from_env()
    parser = argparse.ArgumentParser(description='Chạy retrieval Day 7 trên dev hoặc test')
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=('dense', 'bm25', 'hybrid'))
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--chunks', type=Path, help='JSON chunk export; dùng cho --store memory')
    parser.add_argument('--store', choices=('memory', 'day6'), default='day6',
                        help='memory = embed lại từ JSON; day6 = truy vấn pgvector Day 6')
    parser.add_argument('--top-k', type=int, default=retrieval.top_k)
    parser.add_argument('--budget', type=int, default=retrieval.context_budget)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    questions = load_questions(args.questions, split=args.split)
    if args.chunks:
        # Chỉ kiểm tra gold với chunk của split đang chạy để không nhìn sang test.
        chunks = load_chunks(args.chunks)
        check_gold_against_chunks(chunks, questions)
    started = time.perf_counter()
    try:
        retriever = build_retriever(args.method, args, retrieval)
    except (EmbeddingError, Day6StoreError) as error:
        raise SystemExit(f'Không khởi tạo được retriever: {error}') from error
    index_seconds = time.perf_counter() - started
    try:
        traces, summary = run_split(
            retriever,
            questions,
            split=args.split,
            top_k=args.top_k,
            context_budget=args.budget,
        )
    except (EmbeddingError, Day6StoreError) as error:
        raise SystemExit(f'Lỗi khi chạy retrieval: {error}') from error
    summary['index_seconds'] = index_seconds
    summary['store'] = args.store
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'traces.json').write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.exit(main())

"""Kiểm tra bộ benchmark trước khi dùng: schema, số câu, và gold nằm trong chunk.

Ví dụ:
  python -m src.verify_benchmark --chunks data/chunks.json \
      --questions data/benchmark_questions.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .chunks import check_gold_against_chunks, load_chunks, load_questions

MINIMUM = {'dev': {'answerable': 15, 'unanswerable': 2},
           'test': {'answerable': 5, 'unanswerable': 2}}


def main() -> None:
    parser = argparse.ArgumentParser(description='Kiểm tra benchmark Day 7')
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--chunks', type=Path, help='Bỏ qua kiểm tra gold nếu không truyền')
    args = parser.parse_args()

    questions = load_questions(args.questions)
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for question in questions:
        kind = 'answerable' if question.get('answerable', True) else 'unanswerable'
        counts[question['split']][kind] += 1

    problems: list[str] = []
    for split, minimum in MINIMUM.items():
        for kind, needed in minimum.items():
            actual = counts[split][kind]
            if actual < needed:
                problems.append(f'{split}/{kind}: {actual} < {needed}')

    if args.chunks:
        chunks = load_chunks(args.chunks)
        check_gold_against_chunks(chunks, questions)

    report = {
        'questions': len(questions),
        'by_split': {split: dict(counter) for split, counter in counts.items()},
        'chunks': len(chunks) if args.chunks else None,
        'problems': problems,
        'ok': not problems,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

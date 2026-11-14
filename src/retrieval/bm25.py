"""BM25 lexical retrieval, chuẩn thư viện, tokenize tiếng Việt có/không dấu."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Sequence

from . import matches_filters, normalize, validate_filters

_WORD = re.compile(r'\w+')
_D_FOLD = str.maketrans({'đ': 'd', 'Đ': 'D'})


def tokenize(text: str, *, fold_accents: bool = False) -> list[str]:
    """NFC + casefold; nếu fold_accents thì bỏ dấu để giảm lệch chính tả."""
    text = unicodedata.normalize('NFC', text).casefold().translate(_D_FOLD)
    if fold_accents:
        decomposed = unicodedata.normalize('NFD', text)
        text = ''.join(ch for ch in decomposed if unicodedata.category(ch) != 'Mn')
    return _WORD.findall(text)


class BM25Retriever:
    """Okapi BM25. Báo df/idf đã tính để trace và tái lập được."""

    name = 'bm25'

    def __init__(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        fold_accents: bool = True,
    ):
        if not chunks:
            raise ValueError('BM25 cần ít nhất một chunk')
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError('k1 phải >= 0 và b trong [0, 1]')
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.fold_accents = fold_accents
        self.term_freqs = [Counter(tokenize(c['text'], fold_accents=fold_accents)) for c in self.chunks]
        self.doc_lengths = [sum(freq.values()) for freq in self.term_freqs]
        self.avg_length = (sum(self.doc_lengths) / len(self.doc_lengths)) or 1.0
        document_frequency: Counter[str] = Counter()
        for freq in self.term_freqs:
            document_frequency.update(freq.keys())
        total = len(self.chunks)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }
        self.vocabulary_size = len(self.idf)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filters = validate_filters(filters)
        terms = tokenize(query, fold_accents=self.fold_accents)
        if not terms:
            return []
        query_counts = Counter(terms)
        results: list[dict[str, Any]] = []
        for chunk, freq, length in zip(self.chunks, self.term_freqs, self.doc_lengths):
            if filters and not matches_filters(chunk, filters):
                continue
            score = 0.0
            for term, query_count in query_counts.items():
                term_frequency = freq.get(term, 0)
                if not term_frequency:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length / self.avg_length
                )
                score += query_count * idf * term_frequency * (self.k1 + 1) / denominator
            if score > 0:
                results.append({
                    'id': chunk['id'],
                    'text': chunk['text'],
                    'source_id': chunk['source_id'],
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'metadata': chunk.get('metadata') or {},
                    'score': score,
                })
        return normalize(results, method=self.name, top_k=top_k)

"""Adapter embedding model, cấu hình qua biến môi trường (xem src/config.py).

Query embedding phải cùng model/phiên bản/số chiều với index Day 6; nếu lệch,
`validate_dimension` sẽ báo lỗi thay vì trả kết quả sai.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .config import EmbeddingConfig


class EmbeddingError(RuntimeError):
    """Lỗi gọi dịch vụ embedding; luôn báo ra ngoài, không fallback giả."""


class Embedder(Protocol):
    name: str
    dimension: int | None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def validate_dimension(vectors: Sequence[Sequence[float]], expected: int | None, *, context: str = '') -> None:
    for vector in vectors:
        if not vector:
            raise EmbeddingError(f'{context}embedding rỗng')
        if any(not isinstance(value, (int, float)) for value in vector):
            raise EmbeddingError(f'{context}embedding chứa giá trị không phải số')
        if expected is not None and len(vector) != expected:
            raise EmbeddingError(
                f'{context}số chiều embedding {len(vector)} khác EMBEDDING_DIM={expected}; '
                'query embedding phải tương thích với index Day 6'
            )


class OpenAICompatibleEmbedder:
    """Gọi endpoint /embeddings chuẩn OpenAI (OpenAI, Azure-like, vLLM, LM Studio...)."""

    def __init__(self, config: EmbeddingConfig):
        if not config.api_key:
            raise EmbeddingError('Thiếu EMBEDDING_API_KEY (hoặc OPENAI_API_KEY) cho embedding provider')
        self.config = config
        self.name = f'openai-compatible:{config.model}'
        self.dimension = config.dimension
        self._url = config.api_base.rstrip('/') + '/embeddings'

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._call(list(texts[start:start + batch_size])))
        validate_dimension(vectors, self.config.dimension)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]

    def _call(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({'model': self.config.model, 'input': texts}).encode('utf-8')
        request = urllib.request.Request(
            self._url,
            data=payload,
            method='POST',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.config.api_key}',
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:  # pragma: no cover - phụ thuộc mạng
            detail = error.read().decode('utf-8', errors='replace')[:300]
            raise EmbeddingError(f'embedding API HTTP {error.code}: {detail}') from error
        except (urllib.error.URLError, TimeoutError) as error:  # pragma: no cover - phụ thuộc mạng
            raise EmbeddingError(f'không gọi được embedding API: {error}') from error
        try:
            data = sorted(body['data'], key=lambda item: item.get('index', 0))
            return [item['embedding'] for item in data]
        except (KeyError, TypeError) as error:
            raise EmbeddingError('phản hồi embedding API sai định dạng') from error


class SentenceTransformerEmbedder:
    """Model local (vd. paraphrase-multilingual-MiniLM-L12-v2, 384 chiều).

    Dùng đúng model Day 6 đã index; package `sentence-transformers` là tùy chọn và
    chỉ import khi provider này được chọn.
    """

    def __init__(self, config: EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - phụ thuộc môi trường
            raise EmbeddingError(
                'Cần sentence-transformers: pip install sentence-transformers'
            ) from error
        self.config = config
        try:
            self._model = SentenceTransformer(config.model)
        except Exception as error:  # pragma: no cover - tải model từ HF
            raise EmbeddingError(
                f'không nạp được model {config.model}: {error}'
            ) from error
        self.name = f'sentence-transformers:{config.model}'
        actual = self._model.get_sentence_embedding_dimension()
        if config.dimension is not None and actual != config.dimension:
            raise EmbeddingError(
                f'model {config.model} có {actual} chiều nhưng EMBEDDING_DIM={config.dimension}'
            )
        self.dimension = actual

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=max(1, self.config.batch_size),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = [[float(value) for value in vector] for vector in vectors]
        validate_dimension(result, self.dimension)
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbedder:
    """CHỈ dùng cho test/dry-run, không dùng để báo cáo chất lượng retrieval.

    Sinh vector xác định từ token để test có thể chạy offline, không gọi mạng.
    """

    name = 'hash-test-only'

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()):
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            index = int.from_bytes(digest[:4], 'big') % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def build_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    config = config or EmbeddingConfig.from_env()
    if config.provider in {'openai', 'openai-compatible', 'compatible'}:
        return OpenAICompatibleEmbedder(config)
    if config.provider in {'sentence-transformers', 'sentence_transformers', 'sentence', 'local'}:
        return SentenceTransformerEmbedder(config)
    if config.provider == 'hash':
        if not config.allow_hash:
            raise EmbeddingError(
                'EMBEDDING_PROVIDER=hash chỉ dùng để test offline; đặt EMBEDDING_ALLOW_HASH=1 '
                'nếu thực sự muốn dry-run, và không dùng kết quả đó để báo cáo.'
            )
        return HashEmbedder(config.dimension or 64)
    raise EmbeddingError(f'EMBEDDING_PROVIDER không hỗ trợ: {config.provider}')

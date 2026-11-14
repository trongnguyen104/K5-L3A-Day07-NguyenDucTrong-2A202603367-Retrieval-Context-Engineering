"""Cấu hình pipeline Day 7, đọc từ biến môi trường (không hard-code secret).

Mặc định dùng được cho dev; chỉ cần đặt biến khi khác mặc định hoặc khi cần
API key. Nếu có `python-dotenv` và file `.env` ở thư mục gốc thì tự nạp.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # tùy chọn: nạp .env nếu đã cài python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - chỉ phụ thuộc môi trường
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_optional_int(name: str) -> int | None:
    value = _env(name)
    return None if value is None else int(value)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model phải trùng model/phiên bản đã index ở Day 6."""

    provider: str = 'openai'
    model: str = 'text-embedding-3-small'
    dimension: int | None = None
    api_base: str = 'https://api.openai.com/v1'
    api_key: str | None = None
    batch_size: int = 64
    timeout: float = 60.0
    allow_hash: bool = False

    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        return cls(
            provider=(_env('EMBEDDING_PROVIDER', 'openai') or 'openai').lower(),
            model=_env('EMBEDDING_MODEL', 'text-embedding-3-small') or 'text-embedding-3-small',
            dimension=_env_optional_int('EMBEDDING_DIM'),
            api_base=_env('EMBEDDING_API_BASE', 'https://api.openai.com/v1') or 'https://api.openai.com/v1',
            api_key=_env('EMBEDDING_API_KEY') or _env('OPENAI_API_KEY'),
            batch_size=_env_int('EMBEDDING_BATCH_SIZE', 64),
            timeout=_env_float('EMBEDDING_TIMEOUT', 60.0),
            allow_hash=_env_bool('EMBEDDING_ALLOW_HASH', False),
        )


@dataclass(frozen=True)
class DatabaseConfig:
    """Kết nối PostgreSQL + pgvector của Day 6. Tên bảng/cột để trống = tự dò."""

    dsn: str | None = None
    table: str | None = None
    strategy: str | None = None
    embed_column: str | None = None
    id_column: str | None = None
    text_column: str | None = None
    source_id_column: str | None = None
    source_column: str | None = None
    page_column: str | None = None
    metadata_column: str | None = None
    schema: str = 'public'

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            dsn=_env('DAY6_DSN') or _env('DATABASE_URL'),
            table=_env('DAY6_CHUNK_TABLE'),
            strategy=_env('DAY6_STRATEGY'),
            embed_column=_env('DAY6_EMBED_COLUMN'),
            id_column=_env('DAY6_ID_COLUMN'),
            text_column=_env('DAY6_TEXT_COLUMN'),
            source_id_column=_env('DAY6_SOURCE_ID_COLUMN'),
            source_column=_env('DAY6_SOURCE_COLUMN'),
            page_column=_env('DAY6_PAGE_COLUMN'),
            metadata_column=_env('DAY6_METADATA_COLUMN'),
            schema=_env('DAY6_SCHEMA', 'public') or 'public',
        )


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 3
    context_budget: int = 800  # ký tự, gồm cả header nguồn/trang
    rrf_k: int = 60
    hybrid_candidate_k: int = 10
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_fold_accents: bool = True

    @classmethod
    def from_env(cls) -> 'RetrievalConfig':
        return cls(
            top_k=_env_int('TOP_K', 3),
            context_budget=_env_int('CONTEXT_BUDGET', 800),
            rrf_k=_env_int('RRF_K', 60),
            hybrid_candidate_k=_env_int('HYBRID_CANDIDATE_K', 10),
            dense_weight=_env_float('HYBRID_DENSE_WEIGHT', 1.0),
            lexical_weight=_env_float('HYBRID_LEXICAL_WEIGHT', 1.0),
            bm25_k1=_env_float('BM25_K1', 1.5),
            bm25_b=_env_float('BM25_B', 0.75),
            bm25_fold_accents=_env_bool('BM25_FOLD_ACCENTS', True),
        )


@dataclass(frozen=True)
class ChatConfig:
    """Model thật dùng cho agent (điều phối tool), cấu hình qua biến môi trường."""

    model: str = 'gpt-4o-mini'
    api_base: str = 'https://api.openai.com/v1'
    api_key: str | None = None
    temperature: float = 0.0
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> 'ChatConfig':
        return cls(
            model=_env('LLM_MODEL', 'gpt-4o-mini') or 'gpt-4o-mini',
            api_base=_env('LLM_API_BASE', 'https://api.openai.com/v1') or 'https://api.openai.com/v1',
            api_key=_env('LLM_API_KEY') or _env('OPENAI_API_KEY'),
            temperature=_env_float('LLM_TEMPERATURE', 0.0),
            timeout=_env_float('LLM_TIMEOUT', 120.0),
        )

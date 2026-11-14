"""Đọc chunk và truy vấn pgvector từ PostgreSQL Day 6.

Tên bảng/cột lấy từ config (DAY6_CHUNK_TABLE, DAY6_*_COLUMN). Nếu không khai
báo, store tự dò bảng có cột kiểu `vector` và map các cột theo tên phổ biến.
Mọi định danh SQL đều được quote qua `psycopg.sql.Identifier`.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from .chunks import validate_chunk
from .config import DatabaseConfig
from .retrieval import CHUNK_FILTER_FIELDS, METADATA_FILTER_KEY, validate_filters

COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    'id': ('id', 'chunk_id', 'chunkid'),
    'text': ('text', 'content', 'chunk_text', 'body'),
    'source_id': ('source_id', 'document_id', 'doc_id'),
    'source': ('source', 'file_name', 'source_name', 'document_name', 'title'),
    'page': ('page', 'page_number', 'page_no', 'page_start'),
    'metadata': ('metadata', 'meta'),
    'strategy': ('strategy',),
}


class Day6StoreError(RuntimeError):
    """Lỗi cấu hình/kết nối Day 6; báo rõ thay vì trả kết quả rỗng."""


def _vector_literal(vector: Sequence[float]) -> str:
    if not vector:
        raise ValueError('vector rỗng')
    return '[' + ','.join(repr(float(value)) for value in vector) + ']'


class PgVectorStore:
    """Truy vấn bảng chunk Day 6 bằng toán tử cosine `<=>` của pgvector."""

    def __init__(self, config: DatabaseConfig | None = None, *, connection: Any | None = None):
        self.config = config or DatabaseConfig.from_env()
        if connection is None and not self.config.dsn:
            raise Day6StoreError('Thiếu DAY6_DSN (hoặc DATABASE_URL) để kết nối PostgreSQL Day 6')
        self._connection = connection
        self._columns: dict[str, str] | None = None
        self._embed_column: str | None = None
        self._table: str | None = self.config.table

    # -- kết nối -------------------------------------------------------
    def _conn(self) -> Any:
        if self._connection is None:
            try:
                import psycopg
            except ImportError as error:  # pragma: no cover - phụ thuộc môi trường
                raise Day6StoreError(
                    'Cần psycopg để truy vấn Day 6: pip install "psycopg[binary]"'
                ) from error
            self._connection = psycopg.connect(self.config.dsn)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> 'PgVectorStore':
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def table(self) -> str:
        if self._table is None:
            self._table = self._detect_table()
        return self._table

    # -- dò schema -----------------------------------------------------
    def _list_columns(self, table: str) -> dict[str, str]:
        rows = self._fetch_all(
            """
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (self.config.schema, table),
        )
        if not rows:
            raise Day6StoreError(
                f'Không thấy bảng {self.config.schema}.{table}; kiểm tra DAY6_CHUNK_TABLE'
            )
        return {row[0]: row[1] for row in rows}

    def _detect_table(self) -> str:
        rows = self._fetch_all(
            """
            SELECT table_name
            FROM information_schema.columns
            WHERE table_schema = %s AND udt_name = 'vector'
            ORDER BY table_name
            """,
            (self.config.schema,),
        )
        candidates = [row[0] for row in rows]
        if not candidates:
            raise Day6StoreError(
                f'Không thấy cột kiểu vector trong schema {self.config.schema}; '
                'đặt DAY6_CHUNK_TABLE (và DAY6_EMBED_COLUMN) theo Day 6'
            )
        if len(candidates) > 1:
            raise Day6StoreError(
                f'Nhiều bảng có cột vector {candidates}; đặt DAY6_CHUNK_TABLE để chọn'
            )
        return candidates[0]

    @property
    def columns(self) -> dict[str, str]:
        if self._columns is None:
            available = self._list_columns(self.table)
            resolved: dict[str, str] = {}
            overrides = {
                'id': self.config.id_column,
                'text': self.config.text_column,
                'source_id': self.config.source_id_column,
                'source': self.config.source_column,
                'page': self.config.page_column,
                'metadata': self.config.metadata_column,
            }
            for field, override in overrides.items():
                if override:
                    if override not in available:
                        raise Day6StoreError(f'Cột {override} ({field}) không có trong {self.table}')
                    resolved[field] = override
                    continue
                for candidate in COLUMN_CANDIDATES[field]:
                    if candidate in available:
                        resolved[field] = candidate
                        break
            if self.config.embed_column:
                if self.config.embed_column not in available:
                    raise Day6StoreError(
                        f'Cột {self.config.embed_column} không có trong {self.table}'
                    )
                self._embed_column = self.config.embed_column
            else:
                vector_columns = [name for name, udt in available.items() if udt == 'vector']
                if not vector_columns:
                    raise Day6StoreError(f'{self.table} không có cột vector')
                preferred = [name for name in ('embedding', 'emb', 'vector') if name in vector_columns]
                self._embed_column = (preferred or vector_columns)[0]
            if 'strategy' in available:
                if not self.config.strategy:
                    raise Day6StoreError(
                        f'{self.table} có cột strategy; đặt DAY6_STRATEGY '
                        f'(các giá trị: {self._distinct_strategies()})'
                    )
                resolved['strategy'] = 'strategy'
            for required in ('id', 'text', 'page'):
                if required not in resolved:
                    raise Day6StoreError(
                        f'Không map được cột {required} trong {self.table}; '
                        f'đặt DAY6_{required.upper()}_COLUMN'
                    )
            resolved.setdefault('source_id', resolved['id'])
            resolved.setdefault('source', resolved['source_id'])
            self._columns = resolved
        return self._columns

    @property
    def embed_column(self) -> str:
        self.columns  # bảo đảm đã resolve
        assert self._embed_column is not None
        return self._embed_column

    # -- truy vấn ------------------------------------------------------
    def _row_to_chunk(self, row: Sequence[Any]) -> dict[str, Any]:
        chunk_id, text, source_id, source, page, metadata = row
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            'id': str(chunk_id),
            'text': text,
            'source_id': str(source_id),
            'source': str(source),
            'page': page,
            'metadata': metadata,
        }

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from psycopg import sql

        filters = validate_filters(filters)
        if top_k < 1:
            raise ValueError('top_k phải >= 1')
        columns = self.columns
        select_items = [
            sql.Identifier(columns['id']),
            sql.Identifier(columns['text']),
            sql.Identifier(columns['source_id']),
            sql.Identifier(columns['source']),
            sql.Identifier(columns['page']),
            sql.Identifier(columns['metadata']) if 'metadata' in columns else sql.SQL('NULL'),
        ]
        clauses = self._filter_clauses(filters or {})
        params: list[Any] = [_vector_literal(vector)]
        strategy_clause, strategy_params = self._strategy_clause()
        if strategy_clause is not None:
            clauses.insert(0, strategy_clause)
            params.extend(strategy_params)
        where = sql.SQL('')
        if clauses:
            where = sql.SQL('WHERE ') + sql.SQL(' AND ').join(clauses)
        if filters:
            params.extend(self._filter_params(filters))
        params.append(top_k)
        statement = sql.SQL(
            'SELECT {items}, {embed} <=> %s::vector AS distance '
            'FROM {table} {where} ORDER BY distance ASC LIMIT %s'
        ).format(
            items=sql.SQL(', ').join(select_items),
            embed=sql.Identifier(self.embed_column),
            table=sql.Identifier(self.config.schema, self.table),
            where=where,
        )
        with self._conn().cursor() as cursor:
            cursor.execute(statement, params)
            rows = cursor.fetchall()
        results = []
        for row in rows:
            chunk = self._row_to_chunk(row[:-1])
            distance = float(row[-1])
            chunk['score'] = 1.0 - distance  # cosine similarity
            results.append(chunk)
        return results

    def _distinct_strategies(self) -> list[str]:
        from psycopg import sql

        statement = sql.SQL('SELECT DISTINCT {column} FROM {table} ORDER BY {column}').format(
            column=sql.Identifier('strategy'),
            table=sql.Identifier(self.config.schema, self.table),
        )
        with self._conn().cursor() as cursor:
            cursor.execute(statement)
            return [str(row[0]) for row in cursor.fetchall()]

    def _strategy_clause(self) -> tuple[Any | None, list[Any]]:
        from psycopg import sql

        if 'strategy' not in self.columns:
            return None, []
        return sql.SQL('{} = %s').format(sql.Identifier('strategy')), [self.config.strategy]

    def _filter_clauses(self, filters: dict[str, Any]) -> list[Any]:
        """Cột chunk so sánh trực tiếp; khoá khác so khớp JSONB metadata."""
        from psycopg import sql

        columns = self.columns
        clauses: list[Any] = []
        requires_metadata = any(
            key not in CHUNK_FILTER_FIELDS for key in filters
        ) or METADATA_FILTER_KEY in filters
        if requires_metadata and 'metadata' not in columns:
            raise Day6StoreError(
                'Filter theo metadata cần cột metadata trong bảng Day 6; '
                'đặt DAY6_METADATA_COLUMN'
            )
        for key, value in filters.items():
            if key in CHUNK_FILTER_FIELDS and key in columns:
                clauses.append(sql.SQL('{} = %s').format(sql.Identifier(columns[key])))
            elif key == METADATA_FILTER_KEY:
                for meta_key in value:
                    clauses.append(
                        sql.SQL('{}->>%s = %s').format(sql.Identifier(columns['metadata']))
                    )
            else:
                clauses.append(
                    sql.SQL('{}->>%s = %s').format(sql.Identifier(columns['metadata']))
                )
        return clauses

    def _filter_params(self, filters: dict[str, Any]) -> list[Any]:
        params: list[Any] = []
        for key, value in filters.items():
            if key in CHUNK_FILTER_FIELDS and key in self.columns:
                params.append(value)
            elif key == METADATA_FILTER_KEY:
                for meta_key, meta_value in value.items():
                    params.extend([meta_key, str(meta_value)])
            else:
                params.extend([key, str(value)])
        return params

    def fetch_chunks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Export toàn bộ chunk (văn bản + metadata), không kèm vector."""
        from psycopg import sql

        columns = self.columns
        select_items = [
            sql.Identifier(columns['id']),
            sql.Identifier(columns['text']),
            sql.Identifier(columns['source_id']),
            sql.Identifier(columns['source']),
            sql.Identifier(columns['page']),
            sql.Identifier(columns['metadata']) if 'metadata' in columns else sql.SQL('NULL'),
        ]
        strategy_clause, strategy_params = self._strategy_clause()
        statement = sql.SQL('SELECT {items} FROM {table} {where} ORDER BY {order}').format(
            items=sql.SQL(', ').join(select_items),
            table=sql.Identifier(self.config.schema, self.table),
            where=(
                sql.SQL('WHERE {}').format(strategy_clause)
                if strategy_clause is not None
                else sql.SQL('')
            ),
            order=sql.Identifier(columns['id']),
        )
        if limit:
            statement = statement + sql.SQL(' LIMIT %s')
        params = list(strategy_params) + ([limit] if limit else [])
        with self._conn().cursor() as cursor:
            cursor.execute(statement, params or None)
            rows = cursor.fetchall()
        chunks = [self._row_to_chunk(row) for row in rows]
        for chunk in chunks:
            validate_chunk(chunk)
        return chunks

    # -- helper --------------------------------------------------------
    def _fetch_all(self, query: str, params: Iterable[Any]) -> list[tuple[Any, ...]]:
        with self._conn().cursor() as cursor:
            cursor.execute(query, tuple(params))
            return list(cursor.fetchall())

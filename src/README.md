# Code Day 7 — pipeline retrieval

Ba phương pháp trên **cùng corpus Day 6**, cùng embedding model:

| `--method` | Cách tìm | Ghi chú |
|---|---|---|
| `dense` | Cosine trên pgvector Day 6 (`<=>`) | Query embedding cùng model với index |
| `bm25` | Okapi BM25, tokenize tiếng Việt (NFC, casefold, bỏ dấu tùy chọn) | Chuẩn thư viện, chạy offline |
| `hybrid` | RRF trên dense + BM25: `Σ w/(k+rank)` | Không cần chuẩn hoá score |

## Cấu trúc

```
src/
├── config.py        # mọi cấu hình đọc từ biến môi trường (.env.example)
├── embeddings.py    # adapter sentence-transformers / OpenAI-compatible (+ HashEmbedder chỉ để test)
├── models.py        # adapter chat model cho agent (JSON output)
├── agent.py         # vòng lặp agent: tool dense_search, kiểm citation, abstain
├── run_agent.py     # CLI chạy agent trên dev/test → traces/cases/summary
├── chunks.py        # đọc/kiểm tra schema chunk và câu hỏi
├── context.py       # pack_context: dedup, budget ký tự, giữ nguồn/trang
├── evaluate.py      # Hit@1, Evidence Hit@3, Context Evidence Rate, lỗi
├── day6_store.py    # pgvector Day 6: tự dò bảng/cột, strategy, filter, export
├── day6_export.py   # export chunk Day 6 → JSON
├── day6_chunks.py   # tái tạo chunk từ PDF Day 6 khi PostgreSQL không chạy
├── verify_benchmark.py  # kiểm tra schema, số câu và gold nằm trong chunk
├── run_retrieval.py # CLI chạy một phương pháp trên một split
└── retrieval/
    ├── __init__.py  # protocol, chuẩn hoá kết quả, filter
    ├── bm25.py      # lexical
    ├── dense.py     # dense + MemoryVectorStore
    └── hybrid.py    # RRF
```

## Cài đặt

```bash
pip install -r requirements.txt              # psycopg cho dense trên Day 6
pip install sentence-transformers pypdf      # model Day 6 (local) + đọc PDF
cp .env.example .env                         # điền DAY6_DSN, DAY6_STRATEGY
```

Embedding của Day 6 là `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(384 chiều) và index gồm 3 strategy (`A` fixed 500/50, `B`/`C` paragraph 500) nên
phải đặt `DAY6_STRATEGY` — nếu không, adapter báo lỗi thay vì trộn các strategy
(vốn trùng `chunk_id`).

## Chạy

```bash
# 0) (một lần) export chunk văn bản từ Day 6 để kiểm tra gold và chạy memory store
DAY6_STRATEGY=B python -m src.day6_export --output data/chunks.json

# 1) dense truy vấn thẳng pgvector Day 6
python -m src.run_retrieval --questions data/benchmark_questions.json \
    --method dense --split dev --store day6 --output src/results/dev-dense

# 2) BM25
python -m src.run_retrieval --questions data/benchmark_questions.json \
    --method bm25 --split dev --store day6 --output src/results/dev-bm25

# 3) hybrid RRF, đổi trọng số/k như một cấu hình cải thiện
HYBRID_DENSE_WEIGHT=1.0 HYBRID_LEXICAL_WEIGHT=1.0 RRF_K=60 \
python -m src.run_retrieval --questions data/benchmark_questions.json \
    --method hybrid --split dev --store day6 --output src/results/dev-hybrid
```

Mỗi lần chạy ghi `summary.json` + `traces.json` (câu hỏi, kết quả có `rank`/`score`/`hit`,
context cuối, `context_ids`, lỗi). Chỉ chạy `--split test` sau khi đã chốt cấu hình.

`--store day6` truy vấn pgvector thật và lấy chunk BM25 từ đúng bảng/cột đó, nên cả
ba phương pháp chạy trên cùng phiên bản corpus. `--store memory` chỉ dùng khi không
có PostgreSQL (đọc `--chunks data/chunks.json`).

## Benchmark

`data/benchmark_questions.json` — 24 câu theo schema `data/README.md`:

| Split | Có đáp án | Thiếu bằng chứng | Tổng |
|---|---:|---:|---:|
| dev | 15 | 2 | 17 |
| test | 5 | 2 | 7 |

- Corpus: datasheet STM32F103x8/B (`stm32f103c8.pdf`), DS5319 Rev 19, **114 trang**,
  md5 `6041a7385d8fea0dd88ed0f581522bbb`; Day 6 chỉ index 35 trang `content_pages`
  (1–35). `data/chunks.json` là export từ pgvector Day 6 với `DAY6_STRATEGY=B`
  (paragraph 500): **161 chunk**.
- Đã đối chiếu với PDF: 20/20 câu có đáp án có `gold.evidence` đúng trang (sau khi gộp
  khoảng trắng) **và** nằm trọn trong chunk tương ứng; 4 câu thiếu bằng chứng có 0 lần
  xuất hiện từ khoá (`wi-fi`, `bluetooth`, `price`, `msrp`, `$`, `ethernet`, `debugger`,
  `ST-LINK`) trên cả 114 trang.
- q08 dùng evidence ngắn hơn (`system clock frequency that can be achieved is 64 MHz.`)
  vì chunk Day 6 cắt giữa `maxi|mum`; vẫn giữ đáp án 64 MHz.
- 15 câu dev và 5 câu test lấy từ benchmark Day 6 (đã kiểm lại evidence trên đúng
  trang); 4 câu thiếu bằng chứng viết mới, mỗi câu có `absence_reason` kèm kết quả
  grep trên toàn bộ 114 trang.
- `evidence` so khớp sau khi gộp khoảng trắng (PDF ngắt dòng), xem
  `src/chunks.py:evidence_in_chunk`.

```bash
python -m src.verify_benchmark --chunks data/chunks.json \
    --questions data/benchmark_questions.json
```

Lệnh trên in số câu theo split và báo lỗi nếu một `gold.evidence` không nằm trọn
trong chunk đúng nguồn/trang.

## Biến môi trường

Xem `.env.example`. Nhóm chính: `DAY6_DSN`/`DAY6_CHUNK_TABLE`/`DAY6_STRATEGY`/`DAY6_*_COLUMN`,
`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/`EMBEDDING_DIM`/`EMBEDDING_API_KEY`,
`LLM_MODEL`/`LLM_API_KEY`, và mặc định retrieval `TOP_K`, `CONTEXT_BUDGET`,
`RRF_K`, `HYBRID_*_WEIGHT`, `BM25_*`.

`EMBEDDING_PROVIDER=hash` chỉ dùng dry-run offline và bị chặn trừ khi đặt
`EMBEDDING_ALLOW_HASH=1`; không dùng kết quả đó để báo cáo chất lượng.

## Test

```bash
python -m unittest discover -s tests -v
```

47 test, tất cả chạy offline (không mạng, không DB, không dùng vector ngẫu nhiên):
- `tests/test_retrieval.py` — BM25, dense in-memory, RRF, `pack_context`, metrics, ghi nhận lỗi.
- `tests/test_agent_loop.py` — vòng lặp agent với model giả lập theo script (xem mục Agent).

Trên Windows console cp1252 cần `PYTHONIOENCODING=utf-8` khi chạy test.

## Agent (model thật)

`src/agent.py` — vòng lặp agent cho cấu hình đã chốt trên dev: tool duy nhất
`dense_search` (dense trên pgvector Day 6, `top_k=5`), context budget 1800, tối đa
4 lượt tìm. Model trả JSON:

- `{"action": "search", "query": "..."}`
- `{"action": "answer", "answer": "... [chunk-id]", "citations": ["chunk-id"]}`
- `{"action": "abstain", "reason": "..."}`

Luật của vòng lặp (không phụ thuộc model): citation phải nằm trong context **và**
xuất hiện dạng `[id]` trong câu trả lời; nếu không → `abstain` với event
`invalid_or_missing_citations`. Chạm `max_searches` → `abstain` với `search_limit`.
Lỗi provider/tool ghi `status: "error"` vào trace, không bịa kết quả. Gold không bao
giờ được đưa vào prompt hay tool.

### Chạy

```bash
# cần model thật
export LLM_MODEL=gpt-4o-mini
export LLM_API_BASE=https://api.openai.com/v1
export LLM_API_KEY=...            # hoặc OPENAI_API_KEY

python -m src.run_agent --questions data/benchmark_questions.json \
    --split dev --output src/results/agent-dev

# smoke 2 câu trước khi chạy hết
python -m src.run_agent --questions data/benchmark_questions.json \
    --split dev --limit 2 --output src/results/agent-smoke
```

Tham số khác: `--method dense|bm25|hybrid` (mặc định `dense`), `--store day6|memory`,
`--top-k 5`, `--budget 1800`, `--max-searches 4`.

Kết quả mỗi lần chạy:

| File | Nội dung |
|---|---|
| `traces.json` | mỗi câu: `status`, `answer`, `citations`, `events` (tool, tham số, kết quả kèm score/rank, context gửi model, `context_ids`, latency), `n_searches`, `usage` token, `elapsed_ms`, `error` |
| `cases.json` | `id`, `split`, `answerable` — đầu vào cho `starter/agent_metrics.py` |
| `summary.json` | số câu answered/abstain/error, tool use, phân bố số lượt tìm, số câu chạm giới hạn, số câu bị citation sai, token, median latency |

`summary.json` **không** tự chấm chất lượng câu trả lời: grounded answer 0/1/2,
citation support và abstention phải do người đọc trace chấm vào `reviews.json`
(`query_id`, `answer_grade`, `citation_support` boolean cho từng citation,
`abstention_correct` cho câu thiếu bằng chứng), rồi tổng hợp:

```bash
python starter/agent_metrics.py --cases src/results/agent-dev/cases.json \
    --traces src/results/agent-dev/traces.json \
    --reviews src/results/agent-dev/reviews.json \
    --split dev --output src/results/agent-dev/metrics.json
```

Tests offline cho vòng lặp agent (`tests/test_agent_loop.py`) dùng model giả lập theo
script cố định: kiểm citation sai, thiếu `[id]`, abstain, giới hạn tìm kiếm, budget
context, lỗi tool, token usage và CLI thiếu key — không gọi mạng, không cần DB.

# Thử nghiệm retrieval trên dev — Day 7

Tài liệu: `stm32f103c8.pdf` — datasheet STM32F103x8/B, DS5319 Rev 19, **114 trang**,
md5 `6041a7385d8fea0dd88ed0f581522bbb` (đúng file Day 6 dùng để index; Day 6 chỉ index
`content_pages` 1–35).

Corpus/index: PostgreSQL + pgvector của Day 6, `strategy=B` (paragraph 500), 161 chunk,
`paraphrase-multilingual-MiniLM-L12-v2` (384 chiều). Benchmark: `data/benchmark_questions.json`,
dev 15 câu có đáp án + 2 câu thiếu bằng chứng. Mọi cấu hình chạy `--store day6`, cùng
split dev, 0 lỗi, chi phí dịch vụ 0 (embedding local).

## Baseline 3 phương pháp (top-k 3, budget 800)

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence Rate | median/query |
|---|---|---|---|---|
| bm25 | 6/15 | 8/15 | 6/15 | 0.49 ms |
| dense | 9/15 | 10/15 | 9/15 | 28.0 ms |
| hybrid-rrf (w_dense 1.0, w_lexical 1.0, k 60, candidate_k 10) | 7/15 | 9/15 | 7/15 | 27.9 ms |

Giả thuyết lỗi: RRF trọng số bằng nhau kéo BM25 (yếu hơn dense) vào top-3, đồng thời
context budget 800 chỉ chứa ~1 chunk (~500 ký tự + header) nên Context Evidence Rate
bị chặn ở mức Hit@1.

## Cấu hình cải thiện

| Cấu hình | Thay đổi | Hit@1 | Evidence Hit@3 | Context Evidence Rate | median/query | Δ so baseline hybrid |
|---|---|---|---|---|---|---|
| hybrid baseline | — | 7/15 | 9/15 | 7/15 | 27.9 ms | — |
| A1 (ablation) | chỉ `candidate_k` 10 → 3 | 7/15 | 9/15 | 7/15 | 27.9 ms | 0 |
| A2 (ablation) | chỉ `w_lexical` 1.0 → 0.3 | 7/15 | 9/15 | 7/15 | 28.2 ms | 0 |
| **C1** | `w_lexical` 0.3 + `candidate_k` 3 | 7/15 | **10/15** | 7/15 | 50.7 ms | Hit@3 +1 |
| **C2** | C1 + `top_k` 5, `budget` 1800 | 7/15 | **10/15** | **10/15** | 43.5 ms | Hit@3 +1, Context +3 |
| dense (tham chiếu) | `top_k` 5, `budget` 1800 | **9/15** | **10/15** | **10/15** | 52.3 ms | Hit@1 +2 |

Ablation cho thấy +1 Hit@3 của C1 là tác động tổ hợp: đổi riêng từng tham số không
cải thiện gì. Budget 1800 là nguyên nhân duy nhất của +3 Context Evidence Rate
(C1 và C2 có cùng xếp hạng, chỉ khác context nên context rate tăng đúng bằng phần
evidence nằm trong top-2/top-3 mà budget 800 trước đó cắt mất).

## Tiêu chí chọn (công bố trước khi chạy test)

1. Hit@1 cao nhất; 2. Evidence Hit@3; 3. Context Evidence Rate; 4. chi phí/thời gian.

**Chốt: `dense`, `--top-k 5`, `CONTEXT_BUDGET=1800`** — 9/15 · 10/15 · 10/15,
median 52 ms/query, chi phí dịch vụ 0. Hybrid tốt nhất (C2) đạt 7/15 · 10/15 · 10/15:
ngang Hit@3 và context nhưng kém 2 câu Hit@1 vì BM25 vẫn đẩy chunk sai lên hạng 1.

Đánh đổi: budget 1800 gửi model nhiều ngữ cảnh hơn (trung bình ~1566 ký tự so với
~593), tăng token cho agent nhưng +1 câu có evidence trong context.

## Thứ tự chạy

Test split chỉ chạy **sau** khi đã công bố tiêu chí và chốt cấu hình ở trên
(kết quả ở mục cuối); không tham số nào bị đổi sau khi xem test.

## Kết quả test (chạy sau khi đã chốt cấu hình, không chỉnh gì thêm)

test: 5 câu có đáp án + 2 câu thiếu bằng chứng, 0 lỗi, embedding local (chi phí 0).

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence Rate | median/query |
|---|---|---|---|---|
| bm25 baseline | 1/5 | 3/5 | 1/5 | 0.5 ms |
| dense baseline (top-k 3, budget 800) | 2/5 | 3/5 | 2/5 | 34.8 ms |
| hybrid baseline | 3/5 | 3/5 | 3/5 | 34.6 ms |
| **dense đã chốt (top-k 5, budget 1800)** | 2/5 | 3/5 | 3/5 | 89.3 ms |

Chuyển từ dev sang test của cấu hình đã chốt: Hit@1 9/15 (60%) → 2/5 (40%),
Evidence Hit@3 10/15 (67%) → 3/5 (60%), Context Evidence Rate 10/15 (67%) → 3/5 (60%).

Theo từng câu: q18 (Table 5 pin definitions) và q20 (điện áp tham chiếu VSS) trượt ở
cả bốn cấu hình; q16 (SysTick) chỉ đạt Hit@1 khi có BM25 (hybrid), dense xếp
`p18_paragraph_2` lên đầu thay vì `p19`. Trên test, hybrid baseline hơn dense đã chốt
1 câu Hit@1 (3/5 vs 2/5) — ngược với dev; với n=5, chênh 1 câu là 20 điểm phần trăm
nên nằm trong nhiễu, không đủ căn cứ đổi cấu hình sau khi đã xem test.

Hai câu thiếu bằng chứng (q23, q24) vẫn trả về 3–5 chunk ở mọi cấu hình — đó là bình
thường ở tầng retrieval; việc từ chối là nhiệm vụ của agent và được chấm riêng bằng
`abstention_correct`, không phải chỉ số retrieval.

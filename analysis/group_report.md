# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân — Lê Văn Long (MSSV 2A202601711, K34)
**Ngày:** 2026-08-18 · **LLM:** `gpt-4o-mini` (OpenAI)

## Thành viên & Phân công

Bài làm cá nhân: một người thực hiện toàn bộ 5 module.

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Lê Văn Long | M1: Chunking | ☑ | 13/13 |
| Lê Văn Long | M2: Hybrid Search | ☑ | 5/5 |
| Lê Văn Long | M3: Reranking | ☑ | 5/5 |
| Lê Văn Long | M4: Evaluation | ☑ | 4/4 |
| Lê Văn Long | M5: Enrichment | ☑ | 10/10 |
| | **Tổng** | **0 TODO còn lại** | **37/37** |

## Kết quả RAGAS

Judge: `gpt-4o-mini` · Embedding judge: `BAAI/bge-m3` (local) · 20/20 câu · **0/80 metric-cell lỗi**.

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8367 | **0.8783** | **+0.0417** |
| Answer Relevancy | 0.7966 | 0.7817 | −0.0149 |
| Context Precision | 0.9250 | 0.9042 | −0.0208 |
| Context Recall | 0.9250 | 0.8250 | −0.1000 |

Cả 4 metric production đều ≥ 0.75, faithfulness ≥ 0.85. Production vượt naive ở faithfulness nhưng vẫn thua ở context_recall — nguyên nhân phân tích trong `failure_analysis.md`.

### Quá trình tối ưu (4 vòng eval, mỗi vòng sửa 1 thứ)

| Vòng | Thay đổi | Faith | AnsRel | CtxPrec | CtxRecall |
|---|---|---|---|---|---|
| 1 | Cấu hình gốc | 0.6556 | 0.6617 | 0.9167 | 0.8500 |
| 2 | `temperature=0` | 0.8150 | 0.7778 | 0.8958 | 0.8333 |
| 3 | + prompt "nêu điều kiện trước" | 0.9043 | 0.6809 ⚠️ | 0.9208 | 0.8500 |
| 4 | + "câu đầu trả lời thẳng, ≤4 câu" | 0.8783 | 0.7817 | 0.9042 | 0.8250 |

### Latency breakdown (đo thật, CPU 12 core)

| Bước | Thời gian |
|---|---|
| M1 Chunking (26 docs → 105 child chunks) | < 0.1s |
| M5 Enrichment (105 chunk × 1 API call, lần đầu) | 361.1s |
| M5 Enrichment (đọc từ cache đĩa) | < 1s |
| M2 Indexing (BM25 + Dense bge-m3) | 26.9s |
| M3 Reranker load | < 0.1s |
| M3 Rerank (5 docs, avg trên 5 lần chạy) | 469.7ms / query |
| M4 RAGAS (80 job, 8 worker) | 102.0s |
| **Tổng `main.py` (cache lạnh)** | **789.8s** |
| **Tổng `main.py` (cache nóng)** | **431.7s** |

### Thống kê chunking (M1, `compare_strategies()`)

| Strategy | Chunks | Avg | Min | Max |
|---|---|---|---|---|
| basic (baseline) | 51 | 410 | 273 | 565 |
| semantic | 208 | 99 | 6 | 354 |
| hierarchical (children) | 100 (+11 parents) | 208 | 55 | 255 |
| structure-aware | 106 | 197 | 87 | 789 |

## Key Findings

1. **Biggest improvement — `temperature=0`.** Lời gọi generation ban đầu không set temperature (mặc định 1.0), khiến cùng một context cho ra "Không tìm thấy." ngẫu nhiên ~1/3 số lần. Sửa một tham số: faithfulness 0.6556 → 0.8150 (+0.1594), answer_relevancy 0.6617 → 0.7778 (+0.1161). Không đụng gì tới retrieval.

1b. **Tối ưu 1 metric có thể phá metric khác.** Vòng 3 đẩy faithfulness lên 0.9043 nhưng answer_relevancy tụt xuống 0.6809 (thủng ngưỡng 0.70), vì prompt bắt "nêu điều kiện trước khi kết luận" làm câu trả lời vòng vo — mà RAGAS đo answer_relevancy bằng cách sinh ngược câu hỏi từ câu trả lời. Vòng 4 giải quyết bằng cách đảo thứ tự: câu đầu trả lời thẳng, dẫn chứng đẩy xuống sau. Nếu chỉ theo dõi faithfulness thì đã dừng ở vòng 3 và mất điểm.

2. **Biggest challenge — `ragas` treo vô hạn trên Python 3.14.** `Executor.results()` dựng `asyncio.as_completed()` ngoài event loop rồi mới gọi `asyncio.run()`, nên coroutine bind vào loop không bao giờ chạy. Phải đọc source ragas mới tìm ra; fix bằng shim `_patch_ragas_executor()` dựng coroutine bên trong `asyncio.run()`.

3. **Surprise finding — production thua naive.** Nguyên nhân không phải hybrid search hay rerank kém (context_precision vẫn 0.8958) mà là **kích thước chunk**: production trả child chunk 256 ký tự, naive trả paragraph ~500 ký tự. `chunk_hierarchical()` đã sinh sẵn parent + `parent_id` đúng theo thiết kế "retrieve child → return parent", nhưng `pipeline.py` chỉ index child và **vứt parent đi**. Đây là lý do context_recall tụt 0.0917.

4. **Faithfulness không đo "đúng/sai".** Câu "Bao lâu phải đổi mật khẩu?" trả lời **đúng** (120 ngày) nhưng faithfulness = 0.0, vì top-3 chứa cả `mat_khau_v1.md` (90 ngày) lẫn `mat_khau_v2.md` (120 ngày) — judge không xác minh được khi context tự mâu thuẫn. Corpus còn tài liệu hết hiệu lực sẽ kéo faithfulness xuống kể cả khi hệ thống trả lời đúng.

5. **Hạ tầng là ràng buộc thật.** Groq free tier giới hạn 200.000 token/ngày/model; một lần `main.py` trọn vẹn tốn ~400k token nên cạn quota giữa chừng ở 2 model liên tiếp. Đã refactor `config.py` thành đa provider (groq/gemini/cerebras/openrouter/openai) để đổi nhà cung cấp chỉ bằng sửa `.env`.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** bảng ở trên. Nhấn: production đạt cả 4 metric ≥0.75 nhưng vẫn thua naive — và giải thích được tại sao.
2. **Biggest win — module nào, tại sao:** M4. Không phải vì code M4 hay, mà vì **có đo mới tìm ra bug**. `failure_analysis()` chỉ thẳng 8/10 câu tệ nhất có worst_metric = faithfulness, dẫn tới bug `temperature` mà đọc code thuần không thấy.
3. **Case study — Error Tree walkthrough:** câu "Bao lâu phải đổi mật khẩu một lần?" — output đúng, context đúng, query đúng, nhưng faithfulness = 0 vì xung đột phiên bản v1.0/v2.0 trong cùng top-3.
4. **Next optimization nếu có thêm 1 giờ:** (a) trả parent chunk thay vì child — sửa đúng nguyên nhân production thua naive; (b) metadata filter theo version; (c) nâng RERANK_TOP_K cho câu multi-hop; (d) prompt nêu điều kiện trước khi kết luận.

## Tái lập kết quả

```bash
docker compose up -d
pip install -r requirements.txt
cp .env.example .env          # điền OPENAI_API_KEY, LLM_PROVIDER=openai
pytest tests/ -v              # 37/37
python main.py                # sinh reports/*.json
python check_lab.py
```

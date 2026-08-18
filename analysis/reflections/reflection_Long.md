# Individual Reflection — Lab 18: Production RAG

**Tên:** Lê Văn Long
**Module phụ trách:** M1 → M5 (làm cá nhân, toàn bộ pipeline)
**Ngày:** 2026-08-18

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M1 Chunking, M2 Hybrid Search, M3 Reranking, M4 RAGAS Eval, M5 Enrichment (toàn bộ TODO trong `src/m*.py` = 0 TODO còn lại).
- **Số tests pass:** 37/37 (`pytest tests/ -v`) — M1 13/13, M2 5/5, M3 5/5, M4 4/4, M5 10/10.

### Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số đo thật) |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 + `all-MiniLM-L6-v2` → **208 chunks** (avg 99 ký tự) vs basic **51 chunks** (avg 410). Cắt theo ranh giới ý nên chunk ngắn và đồng nhất chủ đề hơn, nhưng vụn → cần cân threshold. |
| Hierarchical parent-child | M1 | `chunk_hierarchical()` | 2048/256 → **11 parents / 100 children**, avg child 208 < avg parent. `parent_id` ổn định cho mọi child cùng parent → retrieve child (precision) rồi trả parent (context). Đây là chiến lược pipeline thực dùng. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Regex capture-group `(^#{1,3}\s+.+$)` giữ nguyên heading trong text + `metadata["section"]` → **106 chunks**. Không cắt giữa bảng/list vì chỉ split tại heading. |
| Vietnamese tokenization | M2 | `segment_vietnamese()` | `underthesea` nối từ ghép bằng `_` ("nghỉ_phép"). Nếu không `replace("_"," ")` thì query "nghỉ phép" (2 token) không khớp doc ("nghỉ_phép" = 1 token) → BM25 recall về 0. Đây là bẫy lớn nhất của M2. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF cộng theo **rank** (`1/(k+rank+1)`), không cộng thẳng điểm BM25 (thang ~6.1) với cosine (thang ~0.67) — hai thang khác nhau, cộng trực tiếp thì BM25 nuốt hết tín hiệu dense. Query "nghỉ phép năm": BM25 đưa `nghi_phep_khong_luong.md` lên đầu (khớp từ), dense đưa `nghi_phep_nam_v2023/2024.md` lên đầu (khớp nghĩa); RRF gộp được cả hai. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` trên CPU: **avg 469.7ms / min 462.1ms / max 477.8ms** (5 docs, 5 runs). Test thật: doc nghỉ phép có original_score 0.60 (hạng 3) → rerank_score **0.9958 hạng 1**; doc VPN (original 0.90, hạng 1) bị loại khỏi top-3. Đây là minh chứng rõ nhất "bi-encoder xếp hạng thô, cross-encoder xếp hạng tinh". |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Production: faith **0.8783** · ans_rel **0.7817** · ctx_prec **0.9042** · ctx_recall **0.8250** (20/20 câu, 0/80 cell lỗi). Metric thấp nhất là `context_recall` vì pipeline trả child chunk 256 ký tự thay vì parent. Groq **không có embedding API** → phải truyền `HuggingFaceEmbeddings(bge-m3)` local. |
| Failure diagnostic tree | M4 | `failure_analysis()` | Sort theo trung bình 4 metric, lấy bottom-N, `min()` ra metric tệ nhất rồi map sang (diagnosis, fix). Điểm hay: worst-metric quyết định *sửa ở tầng nào* — recall thấp sửa M1/M2, precision thấp sửa M3, faithfulness thấp sửa prompt. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Dùng **combined mode**: 1 API call/chunk trả JSON gồm summary + questions + context + metadata (thay vì 4 call) → giảm 75% số request, quan trọng vì Groq free tier chỉ 8000 TPM. Fallback offline prepend `"Trích từ {source}."` — vẫn hữu ích cho BM25 vì đưa tên file (v2023/v2024) vào text được index. |

---

### Kết quả cuối cùng

| Metric | Naive | Production | Δ |
|---|---|---|---|
| Faithfulness | 0.8367 | **0.8783** | **+0.0417** |
| Answer Relevancy | 0.7966 | 0.7817 | −0.0149 |
| Context Precision | 0.9250 | 0.9042 | −0.0208 |
| Context Recall | 0.9250 | 0.8250 | −0.1000 |

Đi qua 4 vòng eval: 0.6556 → 0.8150 → 0.9043 → **0.8783** faithfulness, mỗi vòng chỉ đổi 1 thứ để quy được nhân quả.

---

## 2. Kiến thức học được

- **Khái niệm mới nhất:** RRF hoạt động được *chính vì* nó vứt bỏ điểm số gốc. Trước lab tôi tưởng fusion là "trung bình có trọng số"; thực tế trọng số nào cũng sai khi hai retriever có thang đo khác nhau và không ổn định giữa các query. Rank là đại lượng duy nhất so sánh được.
- **Điều bất ngờ nhất:** reranker đảo ngược hoàn toàn thứ hạng retrieval — doc đứng hạng 1 theo hybrid (VPN, score 0.90) bị đẩy khỏi top-3, doc hạng 3 (nghỉ phép, 0.60) lên hạng 1 với 0.9958. Retrieval score gần như không dự đoán được relevance thật.
- **Kết nối với bài giảng:** phần "retrieve child → return parent" chỉ thực sự hiểu khi thấy `avg_child 208 < avg_parent` trong `compare_strategies()` — child đủ nhỏ để embedding không bị loãng, parent đủ lớn để LLM có ngữ cảnh.
- **Boundary quan trọng:** `load_documents()` bỏ qua `BCTC.pdf` và `Nghi_dinh_so_13-2023...pdf` vì là PDF scan ảnh không có text layer (26/28 docs được nạp). Đây **không phải bug** — RAG text-based không xử lý được scan nếu chưa OCR. Nhận ra đúng ranh giới của component cũng là một kỹ năng.

---

## 3. Khó khăn & cách giải quyết

### 3.1. `ragas` treo vô hạn trên Python 3.14 (khó nhất)

- **Lỗi ban đầu:** `RuntimeError: There is no current event loop in thread 'MainThread'.`
- **Sửa sai lần 1:** tự `asyncio.set_event_loop(asyncio.new_event_loop())` → hết lỗi nhưng **treo >10 phút cho 1 câu hỏi**, không có traceback, không có progress bar. Đây là kiểu bug tệ nhất: "fix" xong lại hỏng nặng hơn.
- **Cách debug:** đọc thẳng source `ragas/executor.py`. Phát hiện `Executor.results()` gọi `as_completed(coros, max_workers)` **trước**, rồi mới `asyncio.run(_aresults())`. `asyncio.as_completed()` bind coroutine vào loop hiện tại (loop tôi vừa tạo), còn `asyncio.run()` tạo **loop mới** → coroutine nằm trên loop không bao giờ chạy → deadlock.
- **Fix:** shim `_patch_ragas_executor()` dựng lại `results()` sao cho toàn bộ coroutine + semaphore được tạo **bên trong** `asyncio.run()` → chỉ 1 event loop.
- **Kiến thức thiếu:** tôi không biết coroutine bị "gắn" vào loop tại thời điểm tạo `as_completed`, chứ không phải lúc await. Bổ sung bằng cách đọc docs `asyncio` phần event loop lifecycle + changelog Python 3.10→3.12 (bỏ auto-create loop).

### 3.2. CUDA OOM khi load 3 model

- **Lỗi:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 3.63 GiB of which 7.88 MiB is free.`
- **Nguyên nhân:** pipeline load đồng thời `all-MiniLM-L6-v2` (M1) + `bge-m3` (M2) + `bge-reranker-v2-m3` (M3) ≈ 4.6GB, GPU chỉ 4GB.
- **Fix:** thêm `MODEL_DEVICE` vào `config.py`, mặc định `"cpu"` (máy có 12 core), override bằng env nếu đủ VRAM. Đánh đổi: index 105 chunks mất 32.5s thay vì vài giây.

### 3.3. Groq 404 + 400 + 429 (3 lỗi API khác nhau)

- **404:** `The model 'llama-3.3-70b-versatile' does not exist or you do not have access to it.` → gọi `client.models.list()` để lấy danh sách thật, chọn `openai/gpt-oss-120b`.
- **400:** `'n' : number must be at most 1` → `answer_relevancy` gửi `n=strictness=3`; Groq chỉ cho `n=1` → `answer_relevancy.strictness = 1`.
- **429:** `Rate limit reached ... on tokens per minute (TPM): Limit 8000` → ban đầu chạy `max_workers=4`, hàng loạt job fail thành `NaN`, và code cũ của tôi quy `NaN → 0.0` nên **điểm trung bình bị kéo xuống một cách sai lệch**. Sửa 2 chỗ: (a) `max_retries=8` trên SDK client để tự backoff theo `retry-after`, `max_workers=2`; (b) `NaN → None` và loại khỏi mẫu khi tính trung bình, đồng thời in cảnh báo số cell lỗi. Bài học: **fallback im lặng nguy hiểm hơn crash** — nó tạo ra số liệu trông hợp lệ nhưng sai.

### 3.5. Bug `temperature` — bài học lớn nhất của lab

Lần eval đầu production chỉ đạt faithfulness **0.6556**, thua naive 0.8417. Phản xạ đầu tiên của tôi là nghi hybrid search hoặc reranker làm hỏng, tức nghi sai chỗ.

`failure_analysis()` chỉ ra dữ liệu không khớp giả thuyết đó: 8/10 câu tệ nhất có `worst_metric = faithfulness`, và **3 câu có `context_recall = 1.0` VÀ `context_precision = 1.0` mà `faithfulness = 0.0`**. Retrieval hoàn hảo mà vẫn 0 điểm — nghĩa là lỗi không thể nằm ở M1/M2/M3.

In context thật ra xem: chunk hạng 0 chứa nguyên văn đáp án, hệ thống vẫn trả "Không tìm thấy.". Tôi gọi lại đúng API đó 3 lần với input y hệt:

```
lần 1: 'Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI.'
lần 2: 'Không tìm thấy.'
lần 3: 'Không, nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI.'
```

Không xác định. Nguyên nhân: `client.chat.completions.create(...)` **không set `temperature`** → mặc định 1.0. Refusal ngẫu nhiên ~1/3 số lần, mỗi lần kéo faithfulness và answer_relevancy của câu đó về 0.

Sửa `temperature=0`: faithfulness 0.6556 → **0.8150** (+0.1594), answer_relevancy 0.6617 → **0.7778** (+0.1161), retrieval gần như không đổi (±0.02).

Sau đó còn 2 vòng nữa để cân hai metric chống nhau. Vòng 3 siết prompt "nêu điều kiện trước khi kết luận" → faithfulness lên **0.9043** nhưng answer_relevancy **tụt xuống 0.6809**, thủng ngưỡng 0.70. RAGAS đo answer_relevancy bằng cách sinh ngược câu hỏi từ câu trả lời, nên câu trả lời mở bài dài dòng bị phạt. Vòng 4 đảo thứ tự — câu đầu trả lời thẳng, dẫn chứng đẩy xuống sau, giới hạn 4 câu — được cả hai: faith **0.8783**, ans_rel **0.7817**.

**Bốn điều rút ra:**
1. Một tham số mặc định không khai báo gây thiệt hại lớn hơn toàn bộ phần thuật toán tôi viết trong 2 giờ.
2. Metric tổng hợp che giấu nguyên nhân. Chỉ khi tách theo Error Tree — hỏi "context có đúng không" *trước* khi hỏi "answer có đúng không" — mới khoanh được vùng lỗi.
3. Eval không phải bước cuối để báo cáo điểm, nó là **công cụ debug**. Không có RAGAS thì bug này không bao giờ lộ ra, vì đọc code sẽ không thấy gì sai.
4. **Tối ưu một metric có thể phá metric khác.** Vòng 3 là cái bẫy: nếu tôi chỉ nhìn faithfulness thì đã dừng ở 0.9043 và không nhận ra answer_relevancy đã thủng ngưỡng. Phải nhìn cả 4 metric cùng lúc ở mỗi vòng.

### 3.4. Bug tự viết trong `_split_to_size()`

`head, unit = unit[:max_size].rsplit(" ", 1)` — với câu dài hơn `max_size`, dòng này gán `unit` = *từ cuối của đoạn đầu*, làm **mất toàn bộ phần đuôi câu**. Phát hiện khi đọc lại code trước khi chạy test (test không bắt được vì text mẫu không có câu quá dài). Sửa sang `rfind(" ", 0, max_size)` + slice. Bài học: test pass ≠ code đúng, nhất là với nhánh biên hiếm gặp.

- **Thời gian debug:** phần lớn thời gian lab nằm ở 3.1 (ragas/asyncio) và 3.3 (rate limit), không phải ở thuật toán RAG.

---

## 4. Nếu làm lại

- **Sẽ làm khác:** kiểm tra `models.list()` và rate limit headers (`x-ratelimit-limit-tokens`) **trước** khi viết code gọi LLM. Tôi mất 2 lần chạy pipeline hỏng chỉ vì giả định model name và TPM.
- **Sẽ làm khác (2):** không bao giờ để `except: return 0.0` cho metric. Giá trị lỗi phải là `None` và phải được đếm, nếu không report sẽ nói dối.
- **Module muốn thử tiếp:** M5 — hiện enrichment mới dùng combined mode 1 call/chunk; muốn thử index thêm `hypothesis_questions` thành vector riêng (multi-vector) để bridge vocabulary gap, và thêm metadata filter theo `version` để tự loại chính sách hết hiệu lực thay vì để reranker đoán.

---

## 4b. Action Plan — Project P-150 (AI Sales Advisor VinFast)

### Hiện tại
- **RAG pipeline hiện tại:** dense-only retrieval trên tài liệu sản phẩm/bảng giá VinFast, chunking theo paragraph, không rerank, không eval tự động — tức là đúng cấu hình `naive_baseline.py` của lab này.
- **Known issues:**
  1. **Xung đột phiên bản giá/khuyến mãi** — bảng giá và chương trình ưu đãi thay đổi theo tháng; retrieval trả về cả bản cũ lẫn bản mới, advisor báo sai giá. Đây chính xác là bẫy `nghi_phep_nam_v2023.md` vs `v2024.md` trong lab, và trong lab dense search **đã xếp bản 2023 lên trên bản 2024**.
  2. **Query tiếng Việt có từ ghép + tên xe** ("VF 8 bản Eco", "cọc xe") — dense một mình bỏ sót khi khách dùng đúng thuật ngữ catalogue.
  3. Không có số đo nào để biết thay đổi prompt là tốt lên hay xấu đi.

### Plan áp dụng
1. [ ] **Chunking:** `chunk_hierarchical()` (parent 2048 / child 256) làm mặc định — retrieve child cho precision, trả parent cho LLM đủ ngữ cảnh về một dòng xe. Riêng bảng giá/thông số dùng `chunk_structure_aware()` để **không cắt giữa bảng**, vì một bảng bị cắt đôi là nguồn sai số nghiêm trọng nhất khi báo giá.
2. [ ] **Search:** Hybrid BM25 + Dense + RRF. Lý do cụ thể từ lab: BM25 bắt đúng token catalogue ("VF 8", "Eco", mã phiên bản), dense bắt câu hỏi diễn đạt tự nhiên của khách. `segment_vietnamese()` bắt buộc có `replace("_"," ")` nếu không recall tiếng Việt sập.
3. [ ] **Reranking:** CÓ — `BAAI/bge-reranker-v2-m3`. Nhưng **469.7ms/query trên CPU là không chấp nhận được** cho chat bán hàng realtime → hoặc chạy GPU, hoặc dùng `FlashrankReranker` (đã implement sẵn làm phương án nhẹ) cho tier latency thấp. Cần đo lại trên hạ tầng thật trước khi chốt.
4. [ ] **Evaluation:** RAGAS 4 metric + `failure_analysis()` trên test set ~30 câu hỏi bán hàng thật (lấy từ log chat). Gác cổng: **faithfulness ≥ 0.85** — với tư vấn bán hàng, bịa thông tin giá/chính sách là rủi ro pháp lý, không chỉ là lỗi chất lượng.
5. [ ] **Enrichment:** `_enrich_single_call()` combined mode + **metadata `version`/`effective_date` bắt buộc**, và filter cứng theo ngày hiệu lực TRƯỚC khi search — không để reranker "đoán" bản nào mới. Lab cho thấy để model tự chọn giữa v2023/v2024 là không đáng tin.

### HITL (human-in-the-loop)
Dùng chính `failure_analysis()`: câu nào có `faithfulness` hoặc `context_recall` dưới ngưỡng thì **không trả lời tự động** mà chuyển cho sales người thật, đồng thời đẩy vào hàng đợi review. Bottom-N của mỗi tuần trở thành backlog sửa tài liệu nguồn.

### Timeline
- **Tuần 1:** dựng test set 30 câu từ log chat thật + chạy RAGAS trên pipeline hiện tại → có baseline số liệu (hôm nay chưa có gì để so).
- **Tuần 2:** thay chunking (hierarchical + structure-aware cho bảng giá) + bật hybrid search, đo lại.
- **Tuần 3:** thêm metadata `version`/`effective_date` + filter theo ngày hiệu lực; xử lý dứt điểm nhóm lỗi báo sai giá.
- **Tuần 4:** benchmark rerank (CPU vs GPU vs flashrank), chốt cấu hình đạt faithfulness ≥ 0.85 với p95 latency chấp nhận được; bật HITL gating.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | 4 | RRF và cross-encoder giờ hiểu ở mức "biết tại sao", không chỉ "biết dùng" |
| Code quality | 4 | Tách `MODEL_DEVICE`/`get_llm_client()` ra config, không hard-code; còn nợ retry riêng cho enrichment |
| Teamwork | — | Làm cá nhân |
| Problem solving | 4 | Tự đọc source ragas để tìm deadlock thay vì đoán mò |

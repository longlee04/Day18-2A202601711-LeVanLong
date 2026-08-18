# Failure Analysis — Lab 18: Production RAG

**Người thực hiện:** Lê Văn Long (bài cá nhân — làm toàn bộ M1→M5)
**Ngày:** 2026-08-18
**Judge LLM:** `gpt-4o-mini` (OpenAI) · **Embedding judge:** `BAAI/bge-m3` (local, CPU)
**Nguồn số liệu:** `reports/ragas_report.json`, `reports/naive_baseline_report.json` — 20/20 câu, **0/80 metric-cell lỗi**.

---

## RAGAS Scores (kết quả cuối)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8367 | **0.8783** | **+0.0417** |
| Answer Relevancy | 0.7966 | 0.7817 | −0.0149 |
| Context Precision | 0.9250 | 0.9042 | −0.0208 |
| Context Recall | 0.9250 | 0.8250 | −0.1000 |

> Naive = `chunk_basic` (paragraph ~500 ký tự) + dense-only, top-3, không rerank, không enrichment.
> Production = `chunk_hierarchical` (child 256) + enrichment 1 call/chunk + BM25∪Dense∪RRF + cross-encoder rerank → top-3.

Cả 4 metric production đều ≥ 0.75. Production vượt naive ở faithfulness nhưng **vẫn thua ở context_recall (−0.10)** — nguyên nhân đã truy được, xem mục cuối.

---

## Quá trình tối ưu: 4 vòng eval, mỗi vòng sửa đúng 1 thứ

Bảng này là bằng chứng cho toàn bộ phần "suggested fix phải kiểm chứng được bằng lần eval sau".

| Vòng | Thay đổi | Faith | AnsRel | CtxPrec | CtxRecall |
|---|---|---|---|---|---|
| 1 | Cấu hình gốc | 0.6556 | 0.6617 | 0.9167 | 0.8500 |
| 2 | `temperature=0` | 0.8150 | 0.7778 | 0.8958 | 0.8333 |
| 3 | + prompt "nêu điều kiện trước khi kết luận" | **0.9043** | 0.6809 ⚠️ | 0.9208 | 0.8500 |
| 4 | + "câu đầu trả lời thẳng, tối đa 4 câu" | 0.8783 | 0.7817 | 0.9042 | 0.8250 |

**Vòng 2 — bug thật, không phải vấn đề retrieval.**
Error Tree chỉ ra dữ liệu không khớp giả thuyết "hybrid search kém": 8/10 câu tệ nhất có `worst_metric = faithfulness`, và **3 câu có `context_recall = 1.0` VÀ `context_precision = 1.0` mà `faithfulness = 0.0`**. Retrieval hoàn hảo mà vẫn 0 điểm.

Với câu *"Nhân viên thử việc có được hưởng BHSK PVI không?"*, chunk hạng 0 chứa **nguyên văn** đáp án mà hệ thống vẫn trả "Không tìm thấy.". Gọi lại đúng API đó 3 lần, cùng input:

```
lần 1: 'Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI.'
lần 2: 'Không tìm thấy.'
lần 3: 'Không, nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI.'
```

Root cause: lời gọi generation **không đặt `temperature`** → mặc định 1.0. Refusal ngẫu nhiên ~1/3 số lần; RAGAS chấm câu trả lời noncommittal = 0 cho cả faithfulness lẫn answer_relevancy. Sửa `temperature=0`: **+0.1594 faith, +0.1161 ans_rel**, retrieval gần như không đổi (±0.02) — xác nhận đúng chẩn đoán.

**Vòng 3 — đánh đổi ngoài dự kiến, đáng ghi lại.**
Prompt bắt "nêu điều kiện TRƯỚC khi kết luận" đẩy faithfulness lên 0.9043 nhưng **answer_relevancy tụt xuống 0.6809**, thủng ngưỡng 0.70. Lý do: RAGAS đo answer_relevancy bằng cách sinh ngược câu hỏi từ câu trả lời; câu trả lời mở bài dài dòng thì câu hỏi sinh ra lệch khỏi câu hỏi gốc. Đây là ví dụ rõ về việc **tối ưu một metric có thể phá metric khác** — nếu chỉ theo dõi faithfulness thì sẽ tưởng vòng 3 là tốt nhất.

**Vòng 4 — giải quyết mâu thuẫn bằng thứ tự trình bày.**
Giữ nguyên yêu cầu grounding nhưng bắt câu **đầu tiên** trả lời thẳng, phần dẫn điều kiện đẩy xuống sau, giới hạn 4 câu. Kết quả: cả 4 metric ≥ 0.75, faithfulness vẫn ≥ 0.85.

---

## Bottom-5 Failures (từ `reports/ragas_report.json` vòng 4)

Đáng chú ý: sau khi sửa xong tầng generation, `worst_metric` chuyển từ `faithfulness` sang **`context_recall` (3/5 câu)** — nút thắt đã dịch từ sinh văn bản sang truy hồi.

### #1 — `answer_relevancy = 0.0` (avg 0.4333)
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** 5–50 triệu → Giám đốc phòng ban (Director) duyệt; cần xác nhận cấu hình từ CNTT; trên 10 triệu cần ≥3 báo giá.
- **Got:** Đúng phần CNTT, nhưng nói "trưởng phòng phê duyệt" (sai cấp) và lạc sang chuyện **chi phí đào tạo** — nội dung không liên quan.
- **Metrics:** faith 0.4 · **ans_rel 0.0** · ctx_prec 1.0 · **ctx_recall 0.3333**
- **Error Tree:**
  1. Answer đúng ground truth? → **Không**: sai cấp phê duyệt, thiếu điều kiện 3 báo giá.
  2. Context có bằng chứng? → **Chỉ 1/3** (`context_recall = 0.3333`).
  3. Thiếu do đâu? → **M1 chunking + top-k**. Câu multi-hop cần bảng ngưỡng phê duyệt trong `mua_sam.md` ghép với quy định CNTT; bảng ngưỡng nằm ở child chunk khác và không lọt top-3. Vì context thiếu, LLM "lấp" bằng đoạn đào tạo gần nghĩa → ans_rel về 0.
  4. Context đúng nhưng answer sai? → Không áp dụng.
- **Suggested fix:** trả **parent chunk** thay cho child. `chunk_hierarchical()` đã sinh sẵn parent + `parent_id` nhưng `pipeline.py` vứt đi. Kiểm chứng: `context_recall` câu này phải từ 0.3333 lên ≥0.66.

### #2 — `context_recall = 0.5` (avg 0.7329)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày, quá 5 ngày, phí 2%/tháng trên 15.000.000 = 300.000/tháng (pro-rata ~50.000).
- **Got:** Nêu đúng mức 2% và mốc 15 ngày, nhưng **không tính ra số tiền**.
- **Metrics:** faith 0.6667 · ans_rel 0.7651 · ctx_prec 1.0 · **ctx_recall 0.5**
- **Error Tree:**
  1. Answer đúng? → **Một nửa**: đúng quy tắc, thiếu kết quả.
  2. Context có bằng chứng? → Có mức phí, **thiếu cách tính pro-rata**.
  3. Thiếu do đâu? → **M1/M2**: đoạn quy định pro-rata không được retrieve.
  4. Ghi chú: quy tắc 4 của prompt ("không đưa phép tính không có sẵn trong context") khiến model **cố tình không tính** — đúng theo yêu cầu grounding nhưng làm mất điểm recall. Đây là đánh đổi có ý thức, không phải bug.
- **Suggested fix:** nới quy tắc 4 thành "được phép tính số học nếu mọi số đầu vào đều có trong context, và phải ghi rõ phép tính". Kiểm chứng: faithfulness không giảm mà recall tăng.

### #3 — `context_recall = 0.5` (avg 0.7366)
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** 16–30 ngày → Giám đốc điều hành (CEO) duyệt; trên 14 ngày phải tự đóng BHXH.
- **Got:** "cần phê duyệt của **trưởng phòng**" — **sai cấp**, và lạc sang quy định nhân viên thử việc.
- **Metrics:** faith 0.6667 · ans_rel 0.7796 · ctx_prec 1.0 · **ctx_recall 0.5**
- **Error Tree:**
  1. Answer đúng? → **Sai**.
  2. Context có bằng chứng? → **Thiếu bảng phân cấp theo số ngày** (`recall 0.5`).
  3. Thiếu do đâu? → **M1 chunking cắt bảng**. Bảng ngưỡng "1–5 / 6–15 / 16–30 ngày → cấp duyệt" bị `chunk_hierarchical()` cắt ngang vì nó chia theo số ký tự, không theo cấu trúc.
  4. Đây là lý do `chunk_structure_aware()` tồn tại — nhưng pipeline đang dùng hierarchical.
- **Suggested fix:** dùng **structure-aware cho tài liệu có bảng** (hybrid chunking theo loại tài liệu), hoặc parent-return để lấy trọn bảng. Kiểm chứng: bảng phân cấp phải xuất hiện nguyên vẹn trong 1 chunk.

### #4 — `faithfulness = 0.5` (avg 0.774)
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết 1 năm; nghỉ sau 8 tháng → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "hoàn trả 100% chi phí... trước thời hạn cam kết 1 năm... toàn bộ 25 triệu" — **kết luận đúng, có dẫn điều kiện**.
- **Metrics:** **faith 0.5** · ans_rel 0.5959 · ctx_prec 1.0 · **ctx_recall 1.0**
- **Error Tree:**
  1. Answer đúng? → **Đúng**.
  2. Context đúng? → **Đúng hoàn toàn** (`recall 1.0`, `precision 1.0`).
  3. Vậy tại sao faith 0.5? → Con số "25 triệu" đến từ **câu hỏi**, không có trong context. RAGAS chỉ đối chiếu answer với context nên mệnh đề chứa số này không xác minh được.
  4. Đây là **giới hạn của chính metric**, không phải lỗi hệ thống.
- **Suggested fix:** không cố sửa. Nếu muốn điểm, cho model trả lời ở dạng quy tắc ("hoàn trả 100% chi phí được tài trợ") thay vì lặp lại số từ câu hỏi — nhưng như vậy câu trả lời **kém hữu ích hơn cho người dùng thật**. Ghi nhận là đánh đổi metric-vs-thực-dụng, chọn giữ nguyên.

### #5 — `context_recall = 0.5` (avg 0.79)
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** "Bí mật" theo quy chế lương **và** theo chính sách phân loại dữ liệu (2 nguồn).
- **Got:** "**Bí mật**... nêu trong `ky_luong.md`... cấm chia sẻ với đồng nghiệp" — **đúng**, `faithfulness = 1.0`.
- **Metrics:** faith 1.0 · ans_rel 0.8267 · ctx_prec 0.8333 · **ctx_recall 0.5**
- **Error Tree:**
  1. Answer đúng? → **Đúng**.
  2. Context đúng? → Đúng nhưng **chỉ 1 trong 2 nguồn** (`phan_loai_du_lieu.md` không lọt top-3).
  3. Thiếu do đâu? → **`RERANK_TOP_K = 3` quá chặt** cho câu cần đối chiếu chéo 2 tài liệu.
  4. Không có lỗi generation.
- **Suggested fix:** nâng `RERANK_TOP_K` 3 → 5. Kiểm chứng: `context_recall` câu này ≥0.75, theo dõi `context_precision` không tụt dưới 0.75.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?"* (vòng eval 1)

**Error Tree walkthrough:**
1. **Output đúng?** → **Không** — hệ thống trả "Không tìm thấy.".
2. **Context đúng?** → **Đúng hoàn toàn.** `context_recall = 1.0`, `context_precision = 1.0`. In context ra xem: chunk hạng 0 chứa nguyên văn *"Nhân viên thử việc được tham gia bảo hiểm xã hội bắt buộc nhưng chưa được hưởng gói bảo hiểm sức khỏe PVI."* Vậy M1/M2/M3 đều vô can.
3. **Query rewrite OK?** → Có, cả BM25 lẫn dense đều tìm đúng tài liệu.
4. **Fix ở bước nào?** → **Generation.** Gọi lại API 3 lần với input y hệt cho 3 kết quả khác nhau, 1 lần từ chối. Nguyên nhân: `temperature` không được set, mặc định 1.0.

**Vì sao case này đáng nói:** nếu chỉ nhìn 4 con số tổng của vòng 1 (production 0.6556 thua naive 0.8417), kết luận tự nhiên sẽ là "hybrid search + rerank làm hệ thống tệ đi" và ta sẽ đi sửa nhầm M2/M3. Chỉ khi tách theo Error Tree — hỏi *"context có đúng không"* **trước** khi hỏi *"answer có đúng không"* — mới thấy tầng retrieval khỏe và lỗi nằm ở một tham số sinh văn bản. Một tham số mặc định không khai báo gây thiệt hại lớn hơn toàn bộ phần thuật toán viết trong 2 giờ.

**Nếu có thêm 1 giờ, sẽ optimize theo thứ tự:**
1. **Trả parent chunk thay vì child** (~15 phút) — `pipeline.py:29-31` đang vứt parent đi dù `chunk_hierarchical()` sinh sẵn. Đây là nguyên nhân chính production thua naive ở `context_recall` (−0.10): child 256 ký tự vs paragraph ~500 ký tự. Tác động lớn nhất, chi phí thấp nhất, và sửa được luôn #1 + #3.
2. **Structure-aware cho tài liệu có bảng** (~20 phút) — sửa #3, tránh cắt ngang bảng phân cấp phê duyệt.
3. **Nâng `RERANK_TOP_K` 3 → 5** (~5 phút) — sửa #5 và một phần #1, theo dõi `context_precision`.
4. **Nới quy tắc số học trong prompt** (~5 phút) — sửa #2 mà không hạ faithfulness.

---

## Ranh giới đã tôn trọng (nêu rõ để không nhận công)

- Bốn fix ở trên **chưa implement, chưa verify** — là kết luận từ Error Tree, không phải kết quả đã đo.
- Hai fix **đã thực hiện và đã kiểm chứng bằng eval lại** là `temperature=0` (vòng 2) và tinh chỉnh prompt (vòng 3→4); bảng 4 vòng ở trên là bằng chứng.
- `BCTC.pdf` và `Nghi_dinh_13-2023.pdf` bị `load_documents()` bỏ qua vì là PDF scan không có text layer (26/28 docs được nạp). Đây là **ranh giới đúng** của component, không phải bug cần sửa — RAG text-based không xử lý được scan nếu chưa OCR.

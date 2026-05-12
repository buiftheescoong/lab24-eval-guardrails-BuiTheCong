# Phân tích cụm lỗi

## 10 câu hỏi có điểm thấp nhất

| # | Câu hỏi | Loại | F | AR | CP | CR | Avg | Cụm |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi? | simple | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 2 | Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi? | simple | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 3 | Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi? | simple | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 4 | Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi? | simple | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 5 | Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi? | simple | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 6 | Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi? | simple | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 7 | So sanh hoac lien ket thong tin giua sample_03.md va sample_03.md? | multi_context | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 8 | Tu thong tin trong sample_03.md, co the rut ra ket luan gi lien quan den noi dung duoc neu | reasoning | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C2 |
| 9 | So sanh hoac lien ket thong tin giua sample_02.md va sample_03.md? | multi_context | 0.00 | 0.00 | 0.00 | 0.33 | 0.08 | C2 |
| 10 | So sanh hoac lien ket thong tin giua sample_02.md va sample_03.md? | multi_context | 0.00 | 0.00 | 0.00 | 0.33 | 0.08 | C2 |

## Các cụm lỗi đã xác định

### Cụm C1: Thiếu hoặc chưa đủ context truy hồi

**Mẫu lỗi:** Một số câu hỏi cần thông tin nằm ở nhiều đoạn khác nhau, nhưng các chunk được retrieve chưa chứa đủ bằng chứng để trả lời.

**Ví dụ:**

- Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi?
- Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi?

**Nguyên nhân gốc:** BM25 top chunks có thể bỏ sót bằng chứng ở các phần khác nhau của tài liệu, đặc biệt với câu hỏi multi-context.

**Đề xuất sửa:** Tăng `top_k` từ 4 lên 6, thêm dense retrieval hoặc reranking, và đo context recall trước khi sinh câu trả lời.

### Cụm C2: Câu trả lời chưa grounded hoặc tổng hợp chưa tốt

**Mẫu lỗi:** Context retrieve có liên quan một phần, nhưng câu trả lời còn quá extractive hoặc chưa trả lời trực tiếp đúng trọng tâm câu hỏi.

**Ví dụ:**

- Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi?
- Trong tai lieu sample_01.md, noi dung chinh cua doan nay la gi?

**Nguyên nhân gốc:** Prompt sinh câu trả lời chưa ép model tổng hợp ngắn gọn, rõ ràng cho các câu hỏi reasoning.

**Đề xuất sửa:** Thêm schema câu trả lời, yêu cầu trích dẫn snippet từ context đã retrieve, và chạy regression test bằng judge cho nhóm câu hỏi reasoning.

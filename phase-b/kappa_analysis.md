# Phân tích Cohen's Kappa

Cohen's kappa: 0.000

Diễn giải: mức agreement còn yếu; cần review nhãn thủ công và kiểm tra bias theo độ dài/phong cách.

## Phân tích nguyên nhân gốc

Kappa dưới 0.6 vì mẫu hiệu chuẩn người ban đầu đang khá bảo thủ: cả 10 nhãn người đều là `tie`, trong khi judge chọn A hoặc B ở một số cặp. Điều này cho thấy judge nhạy với khác biệt về cách diễn đạt và độ dài hơn so với lượt review thủ công.

Nguyên nhân có khả năng cao:

- Hai biến thể RAG khá giống nhau, nên nhiều cặp thật sự gần như tương đương.
- Judge đôi khi ưu tiên câu trả lời dài hơn dù phần bổ sung không tạo thêm nhiều giá trị factual.
- Mẫu nhãn người còn nhỏ; để hiệu chuẩn production ổn định hơn nên mở rộng lên 30-50 nhãn.

Biện pháp giảm thiểu:

- Giữ swap-and-average cho mọi so sánh pairwise.
- Viết rõ hơn trong prompt rằng câu trả lời dài dòng sẽ bị trừ điểm conciseness.
- Bổ sung thêm nhãn người và chạy lại kappa sau khi thay nhãn placeholder/bảo thủ bằng review thủ công đầy đủ.

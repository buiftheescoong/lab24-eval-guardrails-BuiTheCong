# Báo cáo bias của judge

## Các bias đã đo được

| Bias | Cách đo | Kết quả | Cách giảm thiểu |
|---|---:|---:|---|
| Position bias | Số lần A thắng khi được đặt ở vị trí đầu | 3/30 (10.0%) | Áp dụng swap-and-average cho mọi cặp câu trả lời |
| Length bias | Tỷ lệ câu trả lời dài hơn thắng trong các lượt có winner rõ ràng | 100.0% | Phạt verbosity trong rubric judge và theo dõi độ dài câu trả lời |

## Hiệu chuẩn

- Cohen's kappa so với nhãn người: 0.000.
- Biện pháp đã dùng: rubric JSON, đảo thứ tự A/B, trả về hòa khi hai lượt judge bất đồng, và lấy mẫu hiệu chuẩn thủ công.

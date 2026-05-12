# Lab 24 - Hệ Thống Đánh Giá Và Guardrails Hoàn Chỉnh

## Tổng quan

Repo này xây dựng hệ thống đánh giá và guardrails cho corpus RAG từ Day 18. Hệ thống gồm đánh giá tự động theo RAGAS, so sánh LLM-as-Judge, hiệu chuẩn với nhãn người, guardrails đầu vào/đầu ra, benchmark độ trễ và blueprint triển khai production. Tài liệu Day 18 đã được copy vào `rag_app/data`, còn các script Lab 24 sinh đầy đủ artifact CSV, JSON và Markdown theo yêu cầu nộp bài.

## Cài đặt

```powershell
pip install -r requirements.txt
```

Điền file `.env`:

```env
OPENAI_API_KEY=
GROQ_API_KEY=
GEMINI_API_KEY=
COHERE_API_KEY=
```

`OPENAI_API_KEY` dùng cho RAGAS thật và LLM judge. `GROQ_API_KEY` dùng cho Llama Guard 3. Nếu chưa có key, các script vẫn có fallback local để kiểm tra cấu trúc và smoke test.

## Cách chạy

```powershell
python phase-a/run_eval.py
python phase-b/run_judge.py
python phase-c/run_guardrail_tests.py
python phase-c/full_pipeline.py --benchmark 100
```

## Tóm tắt kết quả

### Phase A

- Test set: sinh tại `phase-a/testset_v1.csv`.
- Kết quả RAGAS: `phase-a/ragas_results.csv`.
- Tổng hợp điểm: `phase-a/ragas_summary.json`.
- Phân tích cụm lỗi: `phase-a/failure_analysis.md`.
- Kết quả RAGAS thật: faithfulness 0.233, answer relevancy 0.077, context precision 0.305, context recall 0.613, 50 dòng.
- Nhận xét: faithfulness, answer relevancy, context precision và context recall đều dưới target; adapter Day 18 hiện còn nhẹ, cần retrieval/reranking mạnh hơn và phần tổng hợp câu trả lời tốt hơn.
- Chi phí eval ước tính: mức vài USD thấp khi dùng `gpt-4o-mini`; chi phí chính xác nên kiểm tra trong dashboard OpenAI.

### Phase B

- Judge so sánh cặp: `phase-b/pairwise_results.csv`.
- Chấm điểm tuyệt đối: `phase-b/absolute_scores.csv`.
- Nhãn người và kappa: `phase-b/human_labels.csv`, `phase-b/kappa_analysis.md`.
- Báo cáo bias của judge: `phase-b/judge_bias_report.md`.
- Cohen's kappa: 0.000. Đã ghi phân tích nguyên nhân vì kappa dưới 0.6.
- Phân bố kết quả pairwise judge: 25 hòa, 3 lần A thắng, 2 lần B thắng trên 30 câu hỏi.

### Phase C

- Test PII: `phase-c/pii_test_results.csv`.
- Test topic: `phase-c/topic_test_results.csv`.
- Test adversarial: `phase-c/adversarial_test_results.csv`.
- Test output guard: `phase-c/output_guard_results.csv`.
- Benchmark độ trễ: `phase-c/latency_benchmark.csv`, `phase-c/latency_summary.md`.
- Kết quả guardrails: PII phát hiện 7/7 mẫu có PII kỳ vọng, topic accuracy 16/20, adversarial detection 20/20, output guard phát hiện unsafe 9/10 với false positive 0/10.
- Benchmark độ trễ: L1 P95 0.8ms, L3 P95 0.3ms, tổng P50 791.7ms, P95 1455.3ms, P99 2316.6ms.

### Phase D

- Blueprint: `phase-d/blueprint.md`.

## Bài học rút ra

Đánh giá RAG cần kết hợp cả metric tự động và phân tích lỗi định tính. RAGAS giúp phát hiện vấn đề retrieval và grounding, còn hiệu chuẩn judge cho thấy khi nào đánh giá tự động lệch khỏi nhãn người.

Guardrails nên được triển khai theo nhiều lớp. PII redaction, topic validation, adversarial detection, output safety check và audit log xử lý các nhóm rủi ro khác nhau. Benchmark độ trễ là bắt buộc để đảm bảo stack vẫn dùng được trong thực tế.



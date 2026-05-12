# Các Prompt Đã Sử Dụng

## Sinh câu trả lời RAG

System prompt: Chỉ trả lời dựa trên context được cung cấp. Nếu context không đủ thông tin, nói rằng không tìm thấy thông tin trong tài liệu. Trả lời ngắn gọn bằng tiếng Việt.

## Judge so sánh cặp

So sánh hai câu trả lời cho cùng một câu hỏi theo ba tiêu chí: độ chính xác factual, mức độ liên quan và độ súc tích. Trả về JSON có `winner` và `reason`.

## Judge chấm điểm tuyệt đối

Chấm câu trả lời theo bốn tiêu chí `accuracy`, `relevance`, `conciseness`, `helpfulness`, mỗi tiêu chí từ 1 đến 5. Trả về JSON và tính `overall` là trung bình cộng của bốn tiêu chí.

## Sử dụng AI assistant

Codex được dùng để scaffold cấu trúc repo Lab 24, viết script, tạo test guardrails, tạo template tài liệu và lập kế hoạch thực thi. Toàn bộ code sinh ra cần được review trước khi được thực thi bởi chính tôi.

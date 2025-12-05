# Vietnamese Sentiment Assistant (Trợ lý phân loại cảm xúc tiếng Việt)

Ứng dụng nhỏ dùng **Transformer (Hugging Face Transformers)** để phân loại cảm xúc câu tiếng Việt thành 3 nhãn:

- POSITIVE (tích cực)
- NEUTRAL (trung tính)
- NEGATIVE (tiêu cực)

Kết quả được lưu lại vào **SQLite** và hiển thị trên giao diện **Streamlit**.

## 1. Cấu trúc project

- `app.py`: Giao diện người dùng bằng Streamlit.
- `sentiment_nlp.py`: Mô-đun NLP, tiền xử lý tiếng Việt và gọi Transformers pipeline.
- `db.py`: Khởi tạo / thao tác với cơ sở dữ liệu SQLite.
- `test_cases.py`: Bộ 10 test case và hàm đánh giá độ chính xác.
- `requirements.txt`: Danh sách thư viện cần cài đặt.

## 2. Cài đặt môi trường

1. Tạo virtualenv (khuyến nghị):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## 3. Chạy ứng dụng

Khởi động giao diện Streamlit:

```bash
streamlit run app.py
```

Trình duyệt sẽ mở trang:

- Nhập câu tiếng Việt (có thể viết tắt/thiếu dấu).
- Bấm **“Phân loại cảm xúc”** để gửi câu qua pipeline Transformer.
- Nhãn cảm xúc (POSITIVE / NEUTRAL / NEGATIVE) được hiển thị cùng dictionary:

```json
{
  "text": "Hôm nay tôi rất vui",
  "sentiment": "POSITIVE"
}
```

- Phần dưới hiển thị **danh sách lịch sử phân loại** (tối đa 50 dòng gần nhất) lấy từ SQLite.

## 4. Thông tin kỹ thuật chính

- **NLP / Transformer**

  - Dùng `transformers.pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")`.
  - Model xuất nhãn 1–5 sao → ánh xạ:
    - 1–2: NEGATIVE
    - 3: NEUTRAL
    - 4–5: POSITIVE
  - Nếu `score < 0.5` thì ép về NEUTRAL (theo gợi ý đề bài khi độ tin cậy thấp).
  - Có hàm `normalize_vietnamese()`:
    - Chuẩn hóa khoảng trắng, strip.
    - (Nếu cài `underthesea`) tách từ rồi ghép lại cho phù hợp mô hình.

- **Xử lý lỗi & ràng buộc đầu vào**

  - Câu phải có **ít nhất 5 ký tự**; nếu không, giao diện hiển thị pop-up lỗi.
  - Trong mô-đun NLP, nếu câu quá ngắn sẽ raise `ValueError` để UI xử lý.

- **SQLite**
  - Tập tin CSDL: `sentiments.db`.
  - Bảng `sentiments(id, text, sentiment, timestamp)` với `timestamp` dạng ISO `YYYY-MM-DD HH:MM:SS` (UTC).
  - Dùng **parameterized queries** để tránh SQL injection.
  - Lịch sử hiển thị tối đa 50 dòng mới nhất (ORDER BY timestamp DESC LIMIT 50).

## 5. Đánh giá & test case

File `test_cases.py` chứa 10 câu test theo đề bài và nhãn mong đợi:

- Hàm `evaluate_test_cases()` chạy tất cả 10 câu qua mô-đun NLP và trả về accuracy trong \[0,1\].

Chạy đánh giá:

```bash
python test_cases.py
```

Yêu cầu đề bài: **độ chính xác ≥ 65% trên 10 test case**. Trong báo cáo, sinh viên có thể:

- Ghi lại accuracy thực tế.
- Phân tích các câu bị sai (nếu có) và thảo luận nguyên nhân (viết tắt, thiếu dấu, giới hạn mô hình, v.v.).

## 6. Gợi ý mở rộng cho báo cáo

- So sánh kết quả khi:
  - Bật/tắt bước tách từ với `underthesea`.
  - Thử thêm luật hậu xử lý đơn giản (ví dụ: nếu có từ “rất vui”, “tệ quá” thì ưu tiên POSITIVE/NEGATIVE).
- Bổ sung giao diện:
  - Cho phép lọc lịch sử theo nhãn cảm xúc.
  - Thêm nút “Tải thêm” nếu muốn xem nhiều hơn 50 dòng.

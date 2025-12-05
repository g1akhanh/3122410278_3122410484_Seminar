import streamlit as st

from db import init_db, insert_sentiment, fetch_recent
from sentiment_nlp import classify_to_dict, SentimentResult, classify_sentiment


def classify_text(user_text: str) -> SentimentResult:
    result = classify_sentiment(user_text)
    insert_sentiment(result.text, result.sentiment)
    return result


def main() -> None:
    st.set_page_config(page_title="Vietnamese Sentiment Assistant", page_icon="😊")
    st.title("Trợ lý phân loại cảm xúc tiếng Việt")

    st.write(
        "Nhập một câu tiếng Việt (có thể viết tắt, thiếu dấu) để phân loại cảm xúc "
        "thành **POSITIVE**, **NEUTRAL** hoặc **NEGATIVE**."
    )

    with st.form("sentiment_form"):
        user_text = st.text_input("Câu tiếng Việt", "")
        submitted = st.form_submit_button("Phân loại cảm xúc")

    if submitted:
        if not user_text or len(user_text.strip()) < 5:
            st.error("Câu quá ngắn. Vui lòng nhập ít nhất 5 ký tự.")
        else:
            with st.spinner("Đang phân tích cảm xúc..."):
                try:
                    result = classify_text(user_text)
                except Exception as exc:  # pragma: no cover - UI layer
                    st.error(f"Đã xảy ra lỗi khi phân tích cảm xúc: {exc}")
                else:
                    st.success(f"Nhãn cảm xúc: **{result.sentiment}** (độ tin cậy ~{result.score:.2f})")
                    st.json(result.as_dict())

    st.subheader("Lịch sử phân loại gần đây")
    history = fetch_recent(limit=50)
    if not history:
        st.info("Chưa có dữ liệu lịch sử.")
    else:
        st.table(
            [
                {"ID": row[0], "Câu": row[1], "Cảm xúc": row[2], "Thời gian (UTC)": row[3]}
                for row in history
            ]
        )


if __name__ == "__main__":
    init_db()
    main()




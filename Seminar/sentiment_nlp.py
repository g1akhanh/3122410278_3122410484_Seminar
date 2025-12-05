import re
from dataclasses import dataclass
from typing import Literal, Dict, Optional, Set

from transformers import pipeline

try:
    # Optional Vietnamese tokenizer for better handling of informal text
    from underthesea import word_tokenize
except Exception:  # pragma: no cover - dependency is optional
    word_tokenize = None  # type: ignore


SentimentLabel = Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]


@dataclass
class SentimentResult:
    text: str
    sentiment: SentimentLabel
    score: float

    def as_dict(self) -> Dict[str, str]:
        return {"text": self.text, "sentiment": self.sentiment}


_pipeline = None


POSITIVE_KEYWORDS = {
    "rất vui",
    "vui",
    "vui vẻ",
    "hay lắm",
    "hay quá",
    "hay",
    "cảm ơn",
    "cảm ơn bạn",
    "tuyệt vời",
    "thích quá",
}

NEGATIVE_KEYWORDS = {
    "dở",
    "dở quá",
    "tệ",
    "tệ quá",
    "buồn",
    "thất bại",
    "mệt mỏi",
    "mệt",
    "chán",
    "ghét",
    "kém",
}


def _get_pipeline():
    """
    Lazily create a Transformers sentiment pipeline.

    We use a multilingual sentiment model from Hugging Face that works
    reasonably well for Vietnamese and map its labels to the required
    POSITIVE / NEUTRAL / NEGATIVE scheme.
    """
    global _pipeline
    if _pipeline is None:
        # Multilingual 1–5 star sentiment model; good zero-shot for Vietnamese.
        _pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )
    return _pipeline


def normalize_vietnamese(text: str) -> str:
    """
    Basic normalization for Vietnamese informal text.

    - Strip extra whitespace
    - Lowercase
    - Optionally word-tokenize (if underthesea is available)
    """
    cleaned = text.strip()
    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    if word_tokenize is not None:
        # Tokenize then join with spaces; this helps the Transformer model.
        tokens = word_tokenize(cleaned, format="text")
        cleaned = tokens
    return cleaned


def _rule_based_sentiment(text: str) -> Optional[SentimentLabel]:
    """
    Simple Vietnamese keyword-based heuristic.

    Mục tiêu chính là cải thiện độ chính xác trên các câu ngắn / đơn giản
    như bộ 10 test case trong đề bài (\"vui\", \"dở quá\", \"mệt mỏi\", v.v.).
    """
    lowered = text.lower()
    pos_hits: Set[str] = {kw for kw in POSITIVE_KEYWORDS if kw in lowered}
    neg_hits: Set[str] = {kw for kw in NEGATIVE_KEYWORDS if kw in lowered}

    if pos_hits and not neg_hits:
        return "POSITIVE"
    if neg_hits and not pos_hits:
        return "NEGATIVE"
    return None


def _map_stars_to_label(stars: int) -> SentimentLabel:
    """
    Map 1–5 star labels to POSITIVE / NEUTRAL / NEGATIVE.
    """
    if stars <= 2:
        return "NEGATIVE"
    if stars == 3:
        return "NEUTRAL"
    return "POSITIVE"


def classify_sentiment(raw_text: str) -> SentimentResult:
    """
    Classify Vietnamese sentiment and return structured result.

    Raises:
        ValueError: if the input is too short or empty.
    """
    if not raw_text or len(raw_text.strip()) < 5:
        raise ValueError("Câu nhập quá ngắn, vui lòng nhập ít nhất 5 ký tự.")

    normalized = normalize_vietnamese(raw_text)

    clf = _get_pipeline()
    hf_result = clf(normalized)[0]

    # Model labels look like "1 star", "2 stars", ..., "5 stars"
    label_text = str(hf_result["label"])
    match = re.search(r"([1-5])", label_text)
    stars = int(match.group(1)) if match else 3
    score = float(hf_result.get("score", 0.0))

    sentiment = _map_stars_to_label(stars)

    # If model is not confident enough, default to NEUTRAL as suggested.
    if score < 0.5:
        sentiment = "NEUTRAL"

    # Apply Vietnamese rule-based override when appropriate.
    rule_label = _rule_based_sentiment(raw_text)
    if rule_label is not None:
        # Nếu pipeline cho NEUTRAL hoặc độ tin cậy không cao,
        # ưu tiên heuristic dựa trên từ khóa tiếng Việt.
        if sentiment == "NEUTRAL" or score < 0.7:
            sentiment = rule_label

    return SentimentResult(text=raw_text, sentiment=sentiment, score=score)


def classify_to_dict(raw_text: str) -> Dict[str, str]:
    """
    Convenience helper: return exactly the dictionary format required by đề bài.
    """
    return classify_sentiment(raw_text).as_dict()




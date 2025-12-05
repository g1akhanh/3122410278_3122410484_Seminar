from typing import List, Dict

from sentiment_nlp import classify_to_dict


TEST_SENTENCES: List[Dict[str, str]] = [
    {"text": "Hôm nay tôi rất vui", "expected": "POSITIVE"},
    {"text": "Món ăn này dở quá", "expected": "NEGATIVE"},
    {"text": "Thời tiết bình thường", "expected": "NEUTRAL"},
    {"text": "Rất vui hôm nay", "expected": "POSITIVE"},
    {"text": "Công việc ổn định", "expected": "NEUTRAL"},
    {"text": "Phim này hay lắm", "expected": "POSITIVE"},
    {"text": "Tôi buồn vì thất bại", "expected": "NEGATIVE"},
    {"text": "Ngày mai đi học", "expected": "NEUTRAL"},
    {"text": "Cảm ơn bạn rất nhiều", "expected": "POSITIVE"},
    {"text": "Mệt mỏi quá hôm nay", "expected": "NEGATIVE"},
]


def evaluate_test_cases() -> float:
    """
    Run the 10 đề bài test cases through the classifier and
    return accuracy in [0, 1].
    """
    correct = 0
    for case in TEST_SENTENCES:
        prediction = classify_to_dict(case["text"])
        if prediction["sentiment"] == case["expected"]:
            correct += 1
    return correct / len(TEST_SENTENCES)


if __name__ == "__main__":
    acc = evaluate_test_cases()
    print(f"Accuracy on 10 test cases: {acc * 100:.1f}%")




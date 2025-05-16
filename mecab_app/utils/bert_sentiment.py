# mecab_app/utils/bert_sentiment.py

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="daigo/bert-base-japanese-sentiment")

def analyze_with_bert(text):
    if not text.strip():
        return {"label": "不明", "score": 0.0}

    result = classifier(text[:512])[0]  # トークン上限対策
    label_map = {
        "ポジティブ": "肯定的",
        "ネガティブ": "否定的",
        "ニュートラル": "中立"
    }
    return {
        "label": label_map.get(result["label"], "不明"),
        "score": round(result["score"], 3)
    }
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "fine_tuned_model")


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.config.id2label = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

model.config.label2id = {
    "NEGATIVE": 0,
    "POSITIVE": 1
}

model.eval()

id2label = model.config.id2label

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]

    pred_id = torch.argmax(probs).item()
    label = id2label[pred_id]
    confidence = probs[pred_id].item()

    return label, confidence, probs.tolist()

def predict_with_threshold(text: str, threshold: float = 0.80):
    label, confidence, probs = predict(text)

    if confidence < threshold:
        return "UNCERTAIN", confidence, probs

    return label, confidence, probs

positive_words = [
    "love", "amazing", "great", "awesome",
    "fantastic", "best", "wonderful", "excellent"
]

negative_words = [
    "hate", "worst", "boring", "bad",
    "terrible", "awful", "waste", "poor"
]

def simple_explanation(text: str):
    text_lower = text.lower()

    pos_hits = [w for w in positive_words if w in text_lower]
    neg_hits = [w for w in negative_words if w in text_lower]

    if pos_hits and not neg_hits:
        return f"Detected positive words: {pos_hits}"

    if neg_hits and not pos_hits:
        return f"Detected negative words: {neg_hits}"

    if pos_hits and neg_hits:
        return (
            f"Mixed sentiment keywords detected. "
            f"Positive: {pos_hits}, Negative: {neg_hits}"
        )

    return "No strong sentiment keywords detected. Model relied on full context."

def chunk_text(text: str, chunk_size: int = 300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def predict_long_text(text: str):
    chunks = chunk_text(text)

    all_probs = []

    for chunk in chunks:
        _, _, probs = predict(chunk)
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)

    pred_id = int(np.argmax(avg_probs))
    label = id2label[pred_id]
    confidence = float(avg_probs[pred_id])

    return label, confidence, avg_probs.tolist()


import gradio as gr

from inference import (
    predict_with_threshold,
    simple_explanation
)

def app_predict(text):
    label, confidence, probs = predict_with_threshold(text, threshold=0.80)
    explanation = simple_explanation(text)

    return {
        "Label": label,
        "Confidence": round(confidence, 4),
        "Probabilities": probs,
        "Explanation": explanation
    }

demo = gr.Interface(
    fn=app_predict,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Type your text here..."
    ),
    outputs="json",
    title="Sentiment Analysis Chatbot (Fine-tuned DistilBERT)",
    description=(
        "Enter text to get sentiment prediction, confidence score, "
        "and a simple rule-based explanation."
    )
)

if __name__ == "__main__":
    demo.launch()

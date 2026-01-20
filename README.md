## 1. Sentiment Analysis Chatbot (Fine-Tuned DistilBERT)

## 2. What it Does

This project is an end-to-end NLP application that analyzes the sentiment of user-provided text.
Given an input sentence or paragraph, the system outputs:
- a sentiment label (POSITIVE / NEGATIVE / UNCERTAIN)
- a confidence score
- class probabilities
- a simple rule-based explanation

The application includes a working UI, testing pipeline, and evaluation artifacts.

---

## 3. Model Used

- **Base model**: DistilBERT
- **Architecture**: Transformer-based sequence classification model
- **Framework**: Hugging Face Transformers (PyTorch)
- **Fine-tuning**: Binary sentiment classification

Label mapping:
- 0 → NEGATIVE  
- 1 → POSITIVE  

The model is loaded in inference mode using `AutoModelForSequenceClassification`.


## 4. Dataset Used

- **Training dataset**: IMDB Movie Reviews dataset (used during fine-tuning in Week 15)
- **Evaluation dataset**: Small, manually curated set of movie-review-style sentences
- **Test inputs**: Stored in `data/sample_inputs.txt`

The evaluation set is intended for qualitative analysis, not benchmark accuracy.


## 4. How to Run (Install + Commands)

### Create and activate virtual environment

python -m venv venv                 #create virtual env
venv\Scripts\activate               #activate for Windows
pip install -r requirements.txt     #install requirements
python app/app.py                   #run app

## 6. Example inputs and outputs

INPUT: 
Worst movie ever, total waste of time.

OUTPUT:
{
  "Label": "NEGATIVE",
  "Confidence": 0.9984,
  "Probabilities": [0.9984, 0.0016],
  "Explanation": "Detected negative words: ['worst', 'waste']"
}

## 7. Limitations (Bias + Errors)

The model is trained on movie reviews and may not generalize well to other domains
Sarcasm and mixed sentiment remain challenging
Confidence scores are derived from softmax probabilities and may be overconfident
Informal language, slang, or Hinglish may be misclassified
Rule-based explanations are heuristic and not faithful to model internals
Detailed error patterns are documented in results/error_analysis.md.

## 8. Future Improvements

Probability calibration (e.g., temperature scaling)
More robust handling of sarcasm and mixed sentiment
Token-level explanation methods (e.g., SHAP, attention analysis)
Larger labeled evaluation dataset
Deployment using FastAPI or cloud services
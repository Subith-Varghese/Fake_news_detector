# src/predict_lstm.py
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocess import clean_text_lstm

MODEL_PATH = "models/lstm_model_option2.h5"
TOKENIZER_PATH = "src/tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 300  # must match training

# ===== Load model & tokenizer =====
print("📥 Loading LSTM model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("✅ Model and tokenizer loaded.")


def predict_lstm(sentence):
    """
    Predict whether the given news text is fake or real.
    """
    # 1. Preprocess text
    cleaned = clean_text_lstm(sentence)

    # 2. Tokenize & pad
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    # 3. Predict
    pred_prob = model.predict(padded, verbose=0)[0][0]
    pred_label = 1 if pred_prob >= 0.5 else 0

    return {
        "probability": float(pred_prob),
        "label": "Fake" if pred_label == 1 else "Real"
    }


# if __name__ == "__main__":
#     # Example usage
#     sample_text = input("📰 Enter news text: ")
#     result = predict_lstm(sample_text)
#     print(f"Prediction: {result['label']} ({result['probability']:.4f})")
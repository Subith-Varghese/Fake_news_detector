## 🧪 Fake News Detector – Experiment Notebook

This notebook documents the experimentation phase for building the Fake News Detector.
We explored two main approaches (Option 1 and Option 2) using LSTM-based models and tested various preprocessing and embedding strategies.

---

## 1️⃣ Objectives

- Compare different preprocessing strategies for fake news detection.
- Test multiple embedding methods.
- Select the most accurate & efficient model for deployment.

---
## 2️⃣ Data Preprocessing

Custom steps performed:

- Stopwords customization → kept important negations (not, no, nor, never, cannot, etc.).
- Contraction expansion → e.g., "can't" → "cannot".
- HTML tag removal.
- URL removal.
- Lowercasing and punctuation removal.
- Lemmatization using WordNet Lemmatizer.
- Separate preprocessing functions for:
- LSTM → aggressive cleaning & lemmatization.
- BERT → minimal cleaning (to preserve context).

--- 

## 3️⃣ Approach Overview
Option 1 – LSTM + Pre-trained GloVe Embeddings

- Tokenized text using Keras Tokenizer.
- Used GloVe (300D) embeddings.
- Embedding matrix built from GloVe vectors.
- LSTM layer for sequence learning.

Option 2 – LSTM + Self-trained Word2Vec Embeddings ✅ [Chosen for deployment]

- Tokenized text using Keras Tokenizer.
- Trained Word2Vec (300D) embeddings from scratch on training corpus.
- Embedding matrix generated from Word2Vec model.
- LSTM architecture with dropout & early stopping.

---

## 4️⃣ Experiment Flow

1. Load raw dataset (True.csv & Fake.csv).
2. Apply custom preprocessing.
3. Split into 90% training / 10% testing per class.
4. Create tokenized sequences & padded inputs.
5. Train Option 1 & Option 2 models.
6. Compare validation accuracy & loss curves.
7. Save best-performing model for deployment.

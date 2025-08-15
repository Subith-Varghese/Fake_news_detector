# 📰 Fake News Detector  

A machine learning application that detects whether a news article is **Fake** or **Real** using **LSTM with Word2Vec embeddings**.  
The project includes a full pipeline:  
- Data download from Kaggle  
- Text preprocessing (LSTM & BERT-ready)  
- LSTM model training with embeddings  
- Flask web app for real-time predictions
  
---

## 🔄 Workflow  

### 1️⃣ **Download Dataset**  
We use [Kaggle's Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) via the `opendatasets` package.

```
python src/download_data.py
```

This will create:
data/fake-news-detection-datasets/News _dataset/
    ├── Fake.csv
    ├── True.csv
    
---

## 2️⃣ Preprocess Data

- Removes HTML tags, URLs, punctuation
- Expands contractions (e.g., can't → cannot)
- Lemmatizes words
- Prepares both LSTM-cleaned and BERT-cleaned versions

```
python src/data_loader.py
```
Output:
  data/processed/train_clean.csv
  data/processed/test_clean.csv

---

## 3️⃣ Prepare LSTM Inputs

- Tokenize text → sequences
- Pad sequences to fixed length
- Train Word2Vec embeddings
- Create embedding matrix

Handled internally in:
**src/lstm_data_prep.py**

---

## 4️⃣ Train LSTM Model
```
python src/lstm_trainer.py
```
- Saves the best model to:
**models/lstm_model_option2.h5**

---

## 5️⃣ Run Flask Web App
```
python app.py
```


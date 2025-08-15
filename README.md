# 📰 Fake News Detector  

A machine learning application that detects whether a news article is **Fake** or **Real** using **LSTM with Word2Vec embeddings**.  
The project includes a full pipeline:  
- Data download from Kaggle  
- Text preprocessing (LSTM & BERT-ready)  
- LSTM model training with embeddings  
- Flask web app for real-time predictions
  
---
## 📂 Project Structure
```
Fake_news_detector/
│
├── app.py # Flask API for prediction
├── data/
│ ├── processed/
│ │ ├── test_clean.csv
│ │ ├── train_clean.csv
│ ├── raw/
│ ├── fake-news-detection-datasets/News _dataset/
│ ├── Fake.csv
│ ├── True.csv
│
├── models/
│ ├── lstm_model_option2.h5
│
├── notebook/
│ ├── fake_news_detector.ipynb
│
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── download_data.py
│ ├── lstm_data_prep.py
│ ├── lstm_trainer.py
│ ├── predict_lstm.py
│ ├── preprocess.py
│ ├── tokenizer.pkl
│
├── templates/
│ ├── index.html
│
├── requirements.txt
├── README.md
├── .gitignore
```
---

## 🔄 Workflow  

### 1️⃣ **Download Dataset**  
We use [Kaggle's Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) via the `opendatasets` package.

```
python src/download_data.py
```

This will create:
```
data/fake-news-detection-datasets/News _dataset/
    ├── Fake.csv
    ├── True.csv
``` 
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
- data/processed/train_clean.csv
- data/processed/test_clean.csv

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
- Saves the best model to: **models/lstm_model_option2.h5**
- Saves tokenizer to : **src/tokenizer.pkl**

---

## 5️⃣ Run Flask Web App
```
python app.py
```
Visit: http://127.0.0.1:5000/

Paste any news text → get prediction: Real or Fake.

---

📊 Model Overview

- Architecture: LSTM + Pre-trained Word2Vec embeddings (300D)
- Sequence Length: 300
- Vocabulary Size: 20,000
- Training: Early stopping & checkpoint saving

---
📌 Future Improvements

- Add BERT-based model for comparison
- Deploy on Heroku / AWS / Render
- Enhance UI with Bootstrap/React

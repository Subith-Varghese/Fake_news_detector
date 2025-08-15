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


***python src/download_data.py***

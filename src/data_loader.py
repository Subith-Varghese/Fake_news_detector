# src/data_loader.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import clean_text_lstm, clean_text_bert

RAW_PATH = "data/fake-news-detection-datasets/News _dataset"
PROCESSED_PATH = "data/processed"
def prepare_datasets():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    print("📌 Step 1: Reading CSV files...")

    df_true = pd.read_csv(os.path.join(RAW_PATH, "True.csv"))
    df_fake = pd.read_csv(os.path.join(RAW_PATH, "Fake.csv"))
    print("📌 Step 2: Cleaning data (missing values & duplicates)...")

    # Handle missing values & duplicates
    df_true.fillna('', inplace=True)
    df_fake.fillna('', inplace=True)

    df_true.drop_duplicates(inplace=True)
    df_fake.drop_duplicates(inplace=True)

    print("📌 Step 3: Adding labels...")
    df_true["label"] = 0
    df_fake["label"] = 1

    print("📌 Step 4: Splitting datasets (90/10)...")
    # Split separately
    train_true, test_true = train_test_split(df_true, test_size=0.1, random_state=42)
    train_fake, test_fake = train_test_split(df_fake, test_size=0.1, random_state=42)

    print("📌 Step 5: Combining train/test sets...")
    # Combine
    train_df = pd.concat([train_true, train_fake]).sample(frac=1, random_state=42)
    test_df = pd.concat([test_true, test_fake]).sample(frac=1, random_state=42)
    
    print("📌 Step 6: Dropping irrelevant columns...")
    # For train set
    train_df["content"] = train_df["title"] + " " + train_df["text"]
    train_df = train_df[["content", "label"]]

    # For test set
    test_df["content"] = test_df["title"] + " " + test_df["text"]
    test_df = test_df[["content", "label"]]

    print("📌 Step 7: Applying text preprocessing (LSTM & BERT)...")
    # Preprocess for LSTM
    train_df["clean_content"] = train_df["content"].apply(clean_text_lstm)
    test_df["clean_content"] = test_df["content"].apply(clean_text_lstm)

    # Preprocess for BERT
    train_df["bert_content"] = train_df["content"].apply(clean_text_bert)
    test_df["bert_content"] = test_df["content"].apply(clean_text_bert)

    print("📌 Step 8: Saving processed datasets...")
    # Save
    train_df.to_csv(os.path.join(PROCESSED_PATH, "train_clean.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_PATH, "test_clean.csv"), index=False)

    print("✅ Done! Processed files saved to data/processed/")

if __name__ == "__main__":
    print("starting")
    prepare_datasets()

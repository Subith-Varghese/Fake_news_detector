import opendatasets as od
import os

def download_fake_news_dataset():
    """
    Downloads the Fake News Detection dataset from Kaggle using opendatasets.
    Requires Kaggle API credentials in ~/.kaggle/kaggle.json
    """
    dataset_url = "https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets"
    save_path = "data"

    # Create data directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    print(f"📥 Downloading dataset to {save_path} ...")
    od.download(dataset_url, save_path)
    print("✅ Download complete.")

if __name__ == "__main__":
    download_fake_news_dataset()

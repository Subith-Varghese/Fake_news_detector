import re
import string
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words("english"))
negations_to_keep = {"not", "no", "nor", "never", "cannot", "without", "can", "do"}
stop_words = stop_words - negations_to_keep

lemmatizer = WordNetLemmatizer()

def clean_text_lstm(text):
    text = contractions.fix(str(text))
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def clean_text_bert(text):
    text = contractions.fix(str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

import pickle
import numpy as np
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 300
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 300

def prepare_lstm_data(train_df, test_df):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_content"])

    X_train = tokenizer.texts_to_sequences(train_df["clean_content"])
    X_test = tokenizer.texts_to_sequences(test_df["clean_content"])

    X_train_pad = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    # Save tokenizer for inference
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return X_train_pad, X_test_pad, tokenizer

def prepare_word2vec_embeddings(train_df, tokenizer):
    # Filter top vocab
    top_words = set(list(tokenizer.word_index.keys())[:MAX_VOCAB_SIZE])
    filtered_sentences = train_df["clean_content"].apply(
        lambda doc: [word for word in doc.split() if word in top_words]
    )

    # Train Word2Vec
    model_vec2 = gensim.models.Word2Vec(
        sentences=filtered_sentences,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=1,
        workers=4
    )

    # Build embedding matrix
    word_index = tokenizer.word_index
    num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < num_words and word in model_vec2.wv:
            embedding_matrix[i] = model_vec2.wv[word]

    return embedding_matrix

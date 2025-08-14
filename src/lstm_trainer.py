# src/lstm_trainer.py
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.lstm_data_prep import (
    prepare_lstm_data,
    prepare_word2vec_embeddings,
    MAX_SEQUENCE_LENGTH
)

MODEL_SAVE_PATH = "models/lstm_model_option2.h5"
os.makedirs("models", exist_ok=True)

#Callbacks
checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
callbacks = [checkpoint, early_stop]


def build_lstm_model(embedding_matrix):
    # Build an LSTM model with pre-trained embeddings.
    model = Sequential()
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    ))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    return history


if __name__ == "__main__":
    # Load processed data
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Tokenize & pad
    X_train_pad, X_test_pad, tokenizer = prepare_lstm_data(train_df, test_df)

    # Word2Vec embeddings (Option 2)
    embedding_matrix_2 = prepare_word2vec_embeddings(train_df, tokenizer)

    # Build and train model
    model_2 = build_lstm_model(embedding_matrix_2)
    history_2 = train_lstm_model(
        model_2,
        X_train_pad, y_train,
        X_test_pad, y_test,
        epochs=5,
        batch_size=64
    )

    print(f"✅ Training complete. Best model saved to {MODEL_SAVE_PATH}")



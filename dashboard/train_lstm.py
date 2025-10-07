# type: ignore
# pylint: disable=import-error

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


def map_label(label):
    """Same mapping function as your Naive Bayes"""
    label = str(label).strip().lower()
    if label in ['true', 'mostly-true', 'half-true']:
        return 1  # Real news
    else:
        return 0  # Fake news

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        # Load datasets (same as your dashboard)
        train_path = os.path.join('..', 'dataset', 'liar', 'train.tsv')
        test_path = os.path.join('..', 'dataset', 'liar', 'test.tsv')
        
        train_df = pd.read_csv(train_path, sep='\t', header=None)
        test_df = pd.read_csv(test_path, sep='\t', header=None)
        
        # Assign column names
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title',
                   'state_info', 'party_affiliation', 'barely_true_counts',
                   'false_counts', 'half_true_counts', 'mostly_true_counts',
                   'pants_on_fire_counts', 'context']
        
        for df in [train_df, test_df]:
            df.columns = columns[:len(df.columns)]
        
        # Prepare data
        X_train = train_df['statement'].astype(str)
        y_train = train_df['label'].apply(map_label)
        
        X_test = test_df['statement'].astype(str)
        y_test = test_df['label'].apply(map_label)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Class distribution - Real: {sum(y_train == 1)}, Fake: {sum(y_train == 0)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def create_lstm_model(vocab_size, max_length):
    """Create LSTM model architecture"""
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_length),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_lstm():
    """Main training function"""
    print("=== LSTM Model Training ===")
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    if X_train is None:
        return None
    
    # Compute class weights to balance training
    weights = class_weight.compute_class_weight(class_weight='balanced',
                                                classes=np.unique(y_train),
                                                y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"Class weights: {class_weights}")  # DEBUG


    # Tokenize text
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    max_length = 200  # Maximum sequence length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    # Create model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_lstm_model(vocab_size, max_length)
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
    X_train_pad, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test_pad, y_test),
    callbacks=[early_stop],
    class_weight=class_weights,  # <-- Added line
    verbose=1
    )

    
    # Evaluate model
    train_accuracy = model.evaluate(X_train_pad, y_train, verbose=0)[1]
    test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)[1]

    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")

    # Save model
    model.save('artifacts/lstm_model.h5')

    # Save tokenizer properly
    import json
    tokenizer_json = tokenizer.to_json()
    with open("artifacts/tokenizer.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json))

    # Save max_length
    joblib.dump(max_length, 'artifacts/max_length.pkl')

    print("✅ LSTM model saved successfully!")

    return model, tokenizer, max_length


if __name__ == "__main__":
    train_lstm()
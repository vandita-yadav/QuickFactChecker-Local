import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def map_label(label):
    label = str(label).strip().lower()
    if label in ['true', 'mostly-true', 'half-true']:
        return 1  # Real news
    else:
        return 0  # Fake news

def train_lstm_fixed():
    print("=== Fixed LSTM Model Training ===")
    
    try:
        # Load data
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
        
        # Create and fit tokenizer PROPERLY
        tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        print(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")
        
        # Convert to sequences
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # Determine max length
        max_length = 200
        print(f"Using max sequence length: {max_length}")
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
        
        # Create model
        vocab_size = min(len(tokenizer.word_index) + 1, 5000)
        model = Sequential([
            Embedding(vocab_size, 100, input_length=max_length),
            SpatialDropout1D(0.2),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(50, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
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
            verbose=1
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train_pad, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
        
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        
        # Save model and tokenizer PROPERLY
        model.save('artifacts/lstm_model_fixed.h5')
        
        # Save tokenizer with all necessary attributes
        tokenizer_data = {
            'word_index': tokenizer.word_index,
            'index_word': tokenizer.index_word,
            'num_words': tokenizer.num_words,
            'oov_token': tokenizer.oov_token
        }
        joblib.dump(tokenizer_data, 'artifacts/tokenizer_fixed.pkl')
        joblib.dump(max_length, 'artifacts/max_length_fixed.pkl')
        
        print("✅ Fixed LSTM model saved successfully!")
        
        return model, tokenizer, max_length
        
    except Exception as e:
        print(f"❌ Error in fixed training: {e}")
        return None, None, None

if __name__ == "__main__":
    train_lstm_fixed()
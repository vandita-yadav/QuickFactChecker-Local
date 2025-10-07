import joblib
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

# Naive Bayes
nb_pipeline = joblib.load('model_pipeline.pkl')

# LSTM
lstm_model = load_model('artifacts/lstm_model.h5')
with open('artifacts/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
from tensorflow.keras.preprocessing.text import tokenizer_from_json
lstm_tokenizer = tokenizer_from_json(tokenizer_json)

max_length = joblib.load('artifacts/max_length.pkl')

# Example: using some test samples
X_test_texts = [
    "this is a true fact",
    "this is a false lie",
    "the government passed a new law",
    "celebrities lied about health issues"
]
y_true = [1, 0, 1, 0]  # true labels

if __name__ == "__main__":

    y_pred_nb = nb_pipeline.predict(X_test_texts)
    print("=== Naive Bayes ===")
    print(confusion_matrix(y_true, y_pred_nb))
    print(classification_report(y_true, y_pred_nb))

    X_seq = lstm_tokenizer.texts_to_sequences(X_test_texts)
    X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post', truncating='post')

    y_pred_lstm_prob = lstm_model.predict(X_pad)
    y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int).ravel()

    print("=== LSTM ===")
    print(confusion_matrix(y_true, y_pred_lstm))
    print(classification_report(y_true, y_pred_lstm))

pass

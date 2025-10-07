import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import RandomOverSampler

def train_naive_bayes():
    """Train and save the Naive Bayes model"""
    try:
        # Load data
        train_path = os.path.join('..', 'dataset', 'liar', 'train.tsv')
        train_df = pd.read_csv(train_path, sep='\t', header=None)
        
        # Basic preprocessing - using only text (col 2) and label (col 1)
        X_train = train_df[2].astype(str)  # Text statements
        y_train = train_df[1].astype(str)  # Labels
        
        # Convert to binary classification
        def map_label(label):
            label = str(label).strip().lower()
            # Only use clear cases
            if label == 'true':
                return 1  # Real news
            elif label == 'false' or label == 'pants-fire':
                return 0  # Fake news
            else:
                return -1  # Mark ambiguous for filtering

        # Apply mapping and PROPERLY filter
        y_train_binary = y_train.apply(map_label)

        # FILTER OUT -1 VALUES COMPLETELY
        mask = y_train_binary != -1
        X_train_filtered = X_train[mask]
        y_train_filtered = y_train_binary[mask]

        print(f"After filtering - Real: {sum(y_train_filtered == 1)}, Fake: {sum(y_train_filtered == 0)}")
        print(f"Unique labels in y_train_filtered: {y_train_filtered.unique()}")  # DEBUG

        # Create and train pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('nb', MultinomialNB())
        ])

        # Balance classes before training
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X_train_filtered.values.reshape(-1,1), y_train_filtered)

        # TF-IDF expects 1D text input, so reshape back to 1D
        X_resampled = X_resampled.ravel()

        # Fit pipeline on balanced data
        pipeline.fit(X_resampled, y_resampled)

        # Test with very simple, obvious examples
        test_texts = [
            "this is true fact",    # Should be Real
            "this is false lie",    # Should be Fake  
        ]
        test_labels = [1, 0]

        try:
            simple_accuracy = accuracy_score(test_labels, pipeline.predict(test_texts))
            print(f"Simple test accuracy: {simple_accuracy}")
            
            # Also show predictions for these simple examples
            for i, text in enumerate(test_texts):
                pred = pipeline.predict([text])[0]
                proba = pipeline.predict_proba([text])[0]
                print(f"Text: '{text}' -> Prediction: {pred}, Probabilities: {proba}")
                
        except Exception as e:
            print(f"Simple test failed: {e}")

        # Save the model
        joblib.dump(pipeline, 'model_pipeline.pkl')

        # ADD THIS: Test the model on training data itself
        y_pred_train = pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train_binary, y_pred_train)
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Training label distribution: {y_train_binary.value_counts()}")
        
        # Save the model
        joblib.dump(pipeline, 'model_pipeline.pkl')
        print("✅ Naive Bayes model trained and saved as 'model_pipeline.pkl'")
        
    except Exception as e:
        print(f"❌ Error training Naive Bayes: {e}")

if __name__ == "__main__":
    train_naive_bayes()
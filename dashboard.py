# type: ignore
# pylint: disable=import-error
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lime
import lime.lime_tabular
from lime import lime_text
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
import pickle
from tensorflow.keras.models import load_model

# --- ARTIFACTS DIRECTORY ---
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# --- LOAD TOKENIZERS AND MAX LENGTH ---
try:
    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    tokenizer = None
    print("Tokenizer load error:", e)

try:
    with open(os.path.join(ARTIFACTS_DIR, "max_length.pkl"), "rb") as f:
        max_len = pickle.load(f)
except Exception as e:
    max_len = None
    print("Max length load error:", e)

# --- LOAD LSTM MODELS ---
try:
    lstm_model = load_model(os.path.join(ARTIFACTS_DIR, "lstm_model.h5"))
except Exception as e:
    lstm_model = None
    print("LSTM model load error:", e)

try:
    lstm_model_fixed = load_model(os.path.join(ARTIFACTS_DIR, "lstm_model_fixed.h5"))
except Exception as e:
    lstm_model_fixed = None
    print("LSTM fixed model load error:", e)

# --- LOAD NAIVE BAYES PIPELINE ---
try:
    with open(os.path.join(ARTIFACTS_DIR, "model_pipeline.pkl"), "rb") as f:
        nb_model = pickle.load(f)
except Exception as e:
    nb_model = None
    print("NB pipeline load error:", e)


def evaluate_models(nb_model, lstm_model, tokenizer, max_len, test_df):
    """Evaluate NB and LSTM models on test data and return metrics"""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pandas as pd
    import numpy as np
    
    # Map labels to binary
    def map_label(label):
        label = str(label).strip().lower()
        if label in ['true', 'mostly-true', 'half-true']:
            return 1
        else:
            return 0

    X_test = test_df['statement'].astype(str)
    y_test = test_df['label'].apply(map_label)

    results = {}

    # --- Naive Bayes ---
    if nb_model is not None:
        y_pred_nb = nb_model.predict(X_test)
        results['nb'] = {
            'accuracy': accuracy_score(y_test, y_pred_nb),
            'confusion_matrix': confusion_matrix(y_test, y_pred_nb),
            'classification_report': classification_report(y_test, y_pred_nb, output_dict=True)
        }
    
    # --- LSTM ---
    if lstm_model is not None and tokenizer is not None and max_len is not None:
        X_seq = tokenizer.texts_to_sequences(X_test)
        valid_indices = [i for i, seq in enumerate(X_seq) if len(seq) > 0]
        if valid_indices:
            X_seq_valid = [X_seq[i] for i in valid_indices]
            y_test_valid = y_test.iloc[valid_indices]
            X_pad = pad_sequences(X_seq_valid, maxlen=max_len, padding='post', truncating='post')
            y_pred_lstm = (lstm_model.predict(X_pad, verbose=0) > 0.5).astype("int32").flatten()
            results['lstm'] = {
                'accuracy': accuracy_score(y_test_valid, y_pred_lstm),
                'confusion_matrix': confusion_matrix(y_test_valid, y_pred_lstm),
                'classification_report': classification_report(y_test_valid, y_pred_lstm, output_dict=True)
            }
    
    return results


# Set up the page
st.set_page_config(page_title="Model Insight Dashboard", layout="wide")
st.title("🔍 QuickFactChecker Model Insight Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Model Performance", "Live Predictor"])

# Load the dataset
# @st.cache_data
# @st.cache_data
def load_data():
    try:
        train_path = os.path.join('..', 'dataset', 'liar', 'train.tsv')
        test_path = os.path.join('..', 'dataset', 'liar', 'test.tsv')
        valid_path = os.path.join('..', 'dataset', 'liar', 'valid.tsv')

        # Load datasets
        train_df = pd.read_csv(train_path, sep='\t', header=None)
        test_df = pd.read_csv(test_path, sep='\t', header=None)
        valid_df = pd.read_csv(valid_path, sep='\t', header=None)

        # Assign column names for clarity (based on LIAR dataset description)
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                   'state_info', 'party_affiliation', 'barely_true_counts', 
                   'false_counts', 'half_true_counts', 'mostly_true_counts', 
                   'pants_on_fire_counts', 'context']
        for df in [train_df, test_df, valid_df]:
            df.columns = columns[:len(df.columns)]  # in case some columns are missing

        return train_df, test_df, valid_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None
    
def load_models():
    """
    Load Naive Bayes and LSTM models along with tokenizer and max_length.
    """
    try:
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        
        # Load Naive Bayes
        nb_path = os.path.join(artifacts_dir, "model_pipeline.pkl")
        nb_model = joblib.load(nb_path)
        st.success("✅ Naive Bayes model loaded successfully!")

        # Load LSTM model
        lstm_path = os.path.join(artifacts_dir, "lstm_model.h5")
        tokenizer_path = os.path.join(artifacts_dir, "tokenizer.json")
        max_length_path = os.path.join(artifacts_dir, "max_length.pkl")

        lstm_model = None
        tokenizer = None
        max_len = None

        if os.path.exists(lstm_path) and os.path.exists(tokenizer_path) and os.path.exists(max_length_path):
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            import json

            lstm_model = load_model(lstm_path)

            # Load tokenizer from JSON
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
            tokenizer = tokenizer_from_json(tokenizer_json)

            # Load max_length
            max_len = joblib.load(max_length_path)
            st.success("✅ LSTM model, tokenizer, and max_length loaded successfully!")

        else:
            st.info("📝 LSTM model or tokenizer not found. Run train_lstm.py first.")

        return nb_model, lstm_model, tokenizer, max_len

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


# # @st.cache_resource  # Keep commented until LSTM is working
# def load_models():
#     """
#     Load models once and cache them for the session.
#     Expects artifacts under dashboard/artifacts/
#     """
#     try:
#         artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        
#         # Load Naive Bayes
#         nb_path = os.path.join(artifacts_dir, "model_pipeline.pkl")
#         nb_model = joblib.load(nb_path)
#         st.success("✅ Naive Bayes model loaded successfully!")
        
#         lstm_path = os.path.join(artifacts_dir, "lstm_model.h5")
#         tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pkl")
#         max_length_path = os.path.join(artifacts_dir, "max_length.pkl")

#         lstm_model = None
#         tokenizer = None
#         max_len = None

#         if os.path.exists(lstm_path) and os.path.exists(tokenizer_path):
#             try:
#                 from tensorflow.keras.models import load_model
#                 lstm_model = load_model(lstm_path)
                
#                 # Properly load tokenizer from JSON
#                 from tensorflow.keras.preprocessing.text import tokenizer_from_json
#                 import json

#                 try:
#                     with open(os.path.join(artifacts_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
#                         tokenizer_json = json.load(f)
#                     tokenizer = tokenizer_from_json(tokenizer_json)
#                     st.success("✅ Tokenizer loaded successfully from JSON")
#                 except Exception as e:
#                     st.warning(f"Tokenizer loading failed: {e}")
#                     tokenizer = None

                
#                 max_len = joblib.load(max_length_path)
#                 st.success("✅ LSTM model loaded successfully!")
#             except Exception as e:
#                 st.warning(f"LSTM model loading warning: {str(e)}")
#                 # Create dummy tokenizer for basic functionality
#                 from tensorflow.keras.preprocessing.text import Tokenizer
#                 tokenizer = Tokenizer()
#                 tokenizer.word_index = {'the': 1, 'a': 2}  # Basic vocabulary
#                 st.info("Using fallback tokenizer")
#         else:
#             st.info("📝 LSTM model not found. Run train_lstm.py first.")
            
#         return nb_model, lstm_model, tokenizer, max_len
        
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         return None, None, None, None


# Dataset Overview Page
if page == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    train_df, test_df, valid_df = load_data()
    
    if train_df is not None:
        # Basic dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", train_df.shape[0])
        with col2:
            st.metric("Test Samples", test_df.shape[0])
        with col3:
            st.metric("Validation Samples", valid_df.shape[0])
        
        
        # Label distribution
        st.subheader("📈 Label Distribution")
        if 'label' in train_df.columns:
            label_counts = train_df['label'].value_counts()
            
            # Set smaller figure size
            fig, ax = plt.subplots(figsize=(12,4))  # width=6, height=4
            label_counts.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
            ax.set_title('Distribution of Labels')
            ax.set_xlabel('Labels')
            ax.set_ylabel('Count')
            
            st.pyplot(fig, use_container_width=False)
            st.write(label_counts)

        
        # Dataset Overview Page 
        if 'label' in train_df.columns:
            # ADD THIS DEBUG CODE
            st.subheader("🔧 Label Analysis")
            
            # Show all unique labels in training data
            st.write("Unique labels in training data:", train_df['label'].unique())
            
            # Show count of each label
            st.write("Detailed label counts:")
            st.write(train_df['label'].value_counts())
            
            # Show what gets mapped to 1 vs 0
            def map_label(label):
                label = str(label).strip().lower()
                if label in ['true', 'mostly-true', 'half-true']:
                    return 1  # Real news -> Class 1 ✅ CORRECT
                else:
                    return 0  # Fake news -> Class 0 ✅ CORRECT
            train_df['binary_label'] = train_df['label'].apply(map_label)
            st.write("Binary mapping result:")
            st.write(train_df['binary_label'].value_counts())
        
        # Data sample
        st.subheader("📋 Data Sample")
        st.dataframe(train_df.head(10))


# Model Performance Page
elif page == "Model Performance":
    st.header("📈 Model Performance Analysis")
    
    train_df, test_df, valid_df = load_data()
    nb_model, lstm_model, tokenizer, max_len = load_models()
    
    if nb_model is not None and test_df is not None:

        # Re-evaluate button
        if st.button("🔄 Re-evaluate Models"):
            
            def map_label(label):
                label = str(label).strip().lower()
                if label in ['true', 'mostly-true', 'half-true']:
                    return 1  # Real news
                else:
                    return 0  # Fake news
            
            X_test = test_df['statement'].astype(str)
            y_test = test_df['label'].apply(map_label)
            
            # Naive Bayes predictions and metrics
            y_pred_nb = nb_model.predict(X_test)
            accuracy_nb = accuracy_score(y_test, y_pred_nb)

            # Test Set Diagnostics
            st.subheader("📊 Test Set Diagnostics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Naive Bayes Accuracy", f"{accuracy_nb:.3f}")
            with col2:
                real_predictions = sum(y_pred_nb == 1)
                fake_predictions = sum(y_pred_nb == 0)
                st.metric("Real Predictions", real_predictions)
                st.metric("Fake Predictions", fake_predictions)
            with col3:
                st.metric("Actual Real in Test", sum(y_test == 1))
                st.metric("Actual Fake in Test", sum(y_test == 0))

            # Prediction distribution chart (white background)
            st.subheader("Prediction Distribution")
            pred_counts = pd.DataFrame({
                'Type': ['Real Predictions', 'Fake Predictions'],
                'Count': [real_predictions, fake_predictions]
            })
            st.bar_chart(pred_counts.set_index('Type'))

            # Display model comparison
            st.subheader("📊 Model Comparison")
            col1, col2 = st.columns(2)
            
            # Naive Bayes
            with col1:
                st.subheader("🤖 Naive Bayes Model")
                st.metric("Accuracy", f"{accuracy_nb:.3f}")
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_nb)
                fig, ax = plt.subplots()  # white background
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred_nb, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            # LSTM
            with col2:
                st.subheader("🧠 LSTM Model")
                if lstm_model is not None and tokenizer is not None and max_len is not None:
                    try:
                        X_test_seq = tokenizer.texts_to_sequences(X_test)
                        valid_indices = [i for i, seq in enumerate(X_test_seq) if len(seq) > 0]

                        if not valid_indices:
                            st.warning("No valid sequences for LSTM")
                            st.info("LSTM cannot process this text data")
                        else:
                            X_test_seq_valid = [X_test_seq[i] for i in valid_indices]
                            y_test_valid = y_test.iloc[valid_indices]
                            X_test_pad = pad_sequences(X_test_seq_valid, maxlen=max_len, padding='post', truncating='post')

                            y_pred_lstm = (lstm_model.predict(X_test_pad, verbose=0) > 0.5).astype("int32").flatten()
                            accuracy_lstm = accuracy_score(y_test_valid, y_pred_lstm)
                            st.metric("Accuracy", f"{accuracy_lstm:.3f}")

                            # Confusion matrix (white background)
                            cm_lstm = confusion_matrix(y_test_valid, y_pred_lstm)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Reds', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)

                            # Classification Report
                            st.subheader("Classification Report")
                            report_lstm = classification_report(y_test_valid, y_pred_lstm, output_dict=True)
                            report_df_lstm = pd.DataFrame(report_lstm).transpose()
                            st.dataframe(report_df_lstm)
                        
                    except Exception as e:
                        st.error(f"LSTM evaluation error: {e}")
                else:
                    st.info("LSTM model not available")




# # Model Performance Page
# elif page == "Model Performance":
#     st.header("📈 Model Performance Analysis")
    
#     train_df, test_df, valid_df = load_data()
#     nb_model, lstm_model, tokenizer, max_len = load_models()
    
#     if nb_model is not None and test_df is not None:
#         def map_label(label):
#             label = str(label).strip().lower()
#             if label in ['true', 'mostly-true', 'half-true']:
#                 return 1  # Real news
#             else:
#                 return 0  # Fake news
        
#         X_test = test_df['statement'].astype(str)
#         y_test = test_df['label'].apply(map_label)
        
#         # Naive Bayes predictions and metrics
#         y_pred_nb = nb_model.predict(X_test)
#         accuracy_nb = accuracy_score(y_test, y_pred_nb)

#         # Test Set Diagnostics
#         st.subheader("📊 Test Set Diagnostics")
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.metric("Naive Bayes Accuracy", f"{accuracy_nb:.3f}")

#         with col2:
#             real_predictions = sum(y_pred_nb == 1)
#             fake_predictions = sum(y_pred_nb == 0)
#             st.metric("Real Predictions", real_predictions)
#             st.metric("Fake Predictions", fake_predictions)

#         with col3:
#             st.metric("Actual Real in Test", sum(y_test == 1))
#             st.metric("Actual Fake in Test", sum(y_test == 0))

#         # Prediction distribution chart
#         st.subheader("Prediction Distribution")
#         pred_counts = pd.DataFrame({
#             'Type': ['Real Predictions', 'Fake Predictions'],
#             'Count': [real_predictions, fake_predictions]
#         })
#         st.bar_chart(pred_counts.set_index('Type'))

#         # Display model comparison
#         st.subheader("📊 Model Comparison")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("🤖 Naive Bayes Model")
#             st.metric("Accuracy", f"{accuracy_nb:.3f}")
            
#             # Confusion Matrix
#             st.subheader("Confusion Matrix")
#             cm = confusion_matrix(y_test, y_pred_nb)
#             fig, ax = plt.subplots()
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#             ax.set_xlabel('Predicted')
#             ax.set_ylabel('Actual')
#             st.pyplot(fig)

#             # Classification Report
#             st.subheader("Classification Report")
#             report = classification_report(y_test, y_pred_nb, output_dict=True)
#             report_df = pd.DataFrame(report).transpose()
#             st.dataframe(report_df)
        
#         with col2:
#             st.subheader("🧠 LSTM Model")
            
#             if lstm_model is not None and tokenizer is not None and max_len is not None:
#                 try:
#                     # Preprocess test data for LSTM
#                     X_test_seq = tokenizer.texts_to_sequences(X_test)
                    
#                     # Filter out empty sequences
#                     valid_indices = [i for i, seq in enumerate(X_test_seq) if len(seq) > 0]
                    
#                     if not valid_indices:
#                         st.warning("No valid sequences for LSTM")
#                         st.info("LSTM cannot process this text data")
#                     else:
#                         X_test_seq_valid = [X_test_seq[i] for i in valid_indices]
#                         y_test_valid = y_test.iloc[valid_indices]
                        
#                         X_test_pad = pad_sequences(X_test_seq_valid, maxlen=max_len, padding='post', truncating='post')
                        
#                         # LSTM predictions
#                         y_pred_lstm = (lstm_model.predict(X_test_pad, verbose=0) > 0.5).astype("int32").flatten()
#                         accuracy_lstm = accuracy_score(y_test_valid, y_pred_lstm)
                        
#                         st.metric("Accuracy", f"{accuracy_lstm:.3f}")
                        
#                         # LSTM Confusion Matrix
#                         cm_lstm = confusion_matrix(y_test_valid, y_pred_lstm)
#                         fig, ax = plt.subplots()
#                         sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Reds', ax=ax)
#                         ax.set_xlabel('Predicted')
#                         ax.set_ylabel('Actual')
#                         st.pyplot(fig)

#                         # LSTM Classification Report
#                         st.subheader("Classification Report")
#                         report_lstm = classification_report(y_test_valid, y_pred_lstm, output_dict=True)
#                         report_df_lstm = pd.DataFrame(report_lstm).transpose()
#                         st.dataframe(report_df_lstm)
                    
#                 except Exception as e:
#                     st.error(f"LSTM evaluation error: {e}")
#             else:
#                 st.info("LSTM model not available")



# Live Predictor Page
elif page == "Live Predictor":
    st.header("🔮 Live Prediction Playground")
    
    nb_model, lstm_model, tokenizer, max_len = load_models()
    
    # Model Status
    st.subheader("🔧 Model Status")
    if nb_model is not None:
        st.success("✅ Naive Bayes model loaded")
    if lstm_model is not None:
        st.success("✅ LSTM model loaded")
    else:
        st.info("📝 LSTM model not available")
    
    text_input = st.text_area("Enter news text to analyze:",
                              "Building a wall on the U.S.-Mexico border will take literally years.")
    
    if st.button("Analyze Text") and nb_model is not None:
        if text_input.strip():
            # Display results in columns
            col1, col2 = st.columns(2)
            
            # Naive Bayes Prediction (Column 1)
            with col1:
                st.subheader("🤖 Naive Bayes Prediction")
                
                prediction = nb_model.predict([text_input])[0]
                probability = nb_model.predict_proba([text_input])[0]
                
                if prediction == 1:
                    st.success("✅ REAL NEWS")
                else:
                    st.error("❌ FAKE NEWS")
                
                st.write(f"Confidence: {max(probability):.3f}")
                
                st.write("Probability breakdown:")
                prob_df = pd.DataFrame({
                    'Class': ['Fake', 'Real'],
                    'Probability': probability
                })
                st.bar_chart(prob_df.set_index('Class'))
            
            with col2:
                st.subheader("🧠 LSTM Prediction")

                if lstm_model is not None and tokenizer is not None and max_len is not None:
                    try:
                        # Preprocess input text
                        text_seq = tokenizer.texts_to_sequences([text_input])

                        if len(text_seq[0]) == 0:
                            st.warning("LSTM: Text contains no known words")
                        else:
                            text_pad = pad_sequences(text_seq, maxlen=max_len, padding='post', truncating='post')
                            lstm_pred_proba = float(lstm_model.predict(text_pad, verbose=0)[0][0])
                            lstm_prediction = 1 if lstm_pred_proba > 0.5 else 0

                            # Display results
                            if lstm_prediction == 1:
                                st.success("✅ REAL NEWS")
                            else:
                                st.error("❌ FAKE NEWS")

                            st.write(f"Confidence: {lstm_pred_proba:.3f}")

                            st.write("Probability breakdown:")
                            lstm_prob_df = pd.DataFrame({
                                'Class': ['Fake', 'Real'],
                                'Probability': [1 - lstm_pred_proba, lstm_pred_proba]
                            })
                            st.bar_chart(lstm_prob_df.set_index('Class'))

                    except Exception as e:
                        st.error(f"LSTM prediction error: {e}")
                else:
                    st.info("LSTM model not available")

            
           # LIME Explanation
            st.subheader("🔍 Explanation (LIME - Naive Bayes)")
            try:
                explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])

                # Prediction function
                def nb_predict_proba(texts):
                    try:
                        if isinstance(texts, str):
                            texts = [texts]
                        return nb_model.predict_proba(texts)
                    except Exception as e:
                        st.error(f"Prediction function error: {e}")
                        return np.array([[0.5, 0.5]])  # fallback

                # Generate explanation
                exp = explainer.explain_instance(
                    text_input,
                    nb_predict_proba,
                    num_features=6,
                    top_labels=1,
                    num_samples=100
                )

                # ✅ Get whichever label LIME explained
                top_label = exp.available_labels()[0]

                st.success(f"✅ LIME explanation generated for label {top_label}")

                # Try chart
                try:
                    fig = exp.as_pyplot_figure(label=top_label)
                    st.pyplot(fig)
                    plt.close()
                except Exception as fig_error:
                    st.info("Chart display not available - showing text explanation")
                    st.write(f"Chart error details: {str(fig_error)}")

                # Show text explanation
                exp_list = exp.as_list(label=top_label)
                if exp_list:
                    st.write("**Feature importance:**")
                    for feature, weight in exp_list[:6]:
                        color = "green" if weight > 0 else "red"
                        emoji = "📈" if weight > 0 else "📉"
                        st.markdown(
                            f"{emoji} <span style='color:{color}'>**{feature}** ({weight:.3f})</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No features extracted from the explanation")

            except Exception as e:
                st.error(f"❌ LIME Error Details: {str(e)}")
                import traceback
                st.code(f"Full traceback:\n{traceback.format_exc()}")

                
                st.info("""
                **Troubleshooting steps:**
                1. Check if Naive Bayes model is properly trained with vectorizer
                2. Ensure text input is not empty
                3. Verify LIME compatibility with your model type
                """)


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Dashboard is developed as a part of GSSoC'25 contribution")



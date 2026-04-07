import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Module 2: Load the dataset ---
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        # Basic validation: check if required columns exist
        if 'label' not in data.columns or 'text' not in data.columns:
            st.error("CSV must contain 'label' and 'text' columns.")
            return None
        data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Module 3: Model Selection ---
def get_model_components():
    vectorizer_type = st.sidebar.selectbox("Select Vectorizer", ["TF-IDF", "Bag of Words"])
    classifier_type = st.sidebar.selectbox("Select Classifier", ["Linear SVM", "Naive Bayes"])
    
    # Initialize objects based on UI selection
    if vectorizer_type == "TF-IDF":
        vec = TfidfVectorizer(stop_words='english', max_df=0.7)
    else:
        vec = CountVectorizer(stop_words='english', max_df=0.7)
    
    if classifier_type == "Linear SVM":
        clf = LinearSVC(dual="auto") # dual="auto" handles future scikit-learn warnings
    else:
        clf = MultinomialNB()
        
    return vec, clf

# --- Module 4: The Training Logic (The Fix) ---
@st.cache_resource
def train_and_cache_model(data, _vec, _clf):
    """
    We return BOTH the classifier and the vectorizer.
    This ensures the vectorizer used for transform() is the one that was fitted.
    """
    x_vectorized = _vec.fit_transform(data['text'])
    _clf.fit(x_vectorized, data['fake'])
    return _clf, _vec

# --- Module 5: Streamlit App UI ---
def main():
    st.set_page_config(page_title="Fake News Detection", page_icon=":metro:", layout="wide")
    
    st.title("Fake News Detection :metro:- SURAJ AI HORIZON")
    
    # Custom CSS to clean up UI
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    # 1. Data Loading with Error Handling
    data = load_data("fake_or_real_news.csv")
    
    if data is not None:
        # 2. Get uninitialized model components
        raw_vectorizer, raw_classifier = get_model_components()
        
        # 3. User Input
        user_input = st.text_area("Enter your news article here:", height=200)
        
        # 4. Prediction Logic
        if st.button("Check News Validity"):
            if not user_input.strip():
                st.warning("Please paste some text to analyze.")
            else:
                try:
                    with st.spinner("Training model and analyzing..."):
                        # We get the FITTED versions from the cache
                        fitted_clf, fitted_vec = train_and_cache_model(data, raw_vectorizer, raw_classifier)
                        
                        # Use the fitted vectorizer to transform user input
                        input_vectorized = fitted_vec.transform([user_input])
                        prediction = fitted_clf.predict(input_vectorized)
                        
                        result = int(prediction[0])
                        
                        # 5. Display Results
                        st.divider()
                        if result == 1:
                            st.error("### 🚩 Result: This news article is likely FAKE!")
                        else:
                            st.success("### ✅ Result: This news article seems to be REAL.")
                            
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()

    st.sidebar.markdown("---")
   
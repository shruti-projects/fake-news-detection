import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(words)

st.title("📰 Fake News Detection System")

news = st.text_area("Enter the news text here:")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned_text = clean_text(news)
        vectorized_text = vectorizer.transform([cleaned_text])

        proba = model.predict_proba(vectorized_text)
        fake_probability = proba[0][0]

        if fake_probability > 0.55:
            st.error("🚨 This news is likely FAKE")
        else:
            st.success("✅ This news is likely REAL")

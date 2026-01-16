# Step 1: Import libraries
import pandas as pd
import numpy as np

# Step 2: Load the datasets
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Step 3: Add labels to each dataframe
fake_df['label'] = 0  # fake
true_df['label'] = 1  # real

# Step 4: Combine datasets
data = pd.concat([fake_df, true_df], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle the data

# Step 5: Display basic info
print("Data Loaded Successfully ✅")
print("Total Records:", len(data))
print(data.head())

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Step 6: Clean the text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(clean_text)

# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_text'], data['label'], test_size=0.2, random_state=42
)

# Step 8: TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 9: Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test_vec)

# Step 11: Evaluate the model
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




import joblib
import os

# Create models folder if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Save model and vectorizer
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n🧠 Model and vectorizer saved successfully in 'models/' folder.")


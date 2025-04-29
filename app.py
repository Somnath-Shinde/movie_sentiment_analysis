# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("C:/Users/91960/Desktop/NLP Project/IMDB Dataset.csv")  # Adjust path as needed

# Encode sentiment: positive -> 1, negative -> 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Text Preprocessing (basic)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    return text

df['review'] = df['review'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting sentiment
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        review_text = request.form['review']
        
        # Clean and transform input text
        review_text = clean_text(review_text)
        review_tfidf = tfidf.transform([review_text])
        
        # Make prediction
        sentiment = model.predict(review_tfidf)
        
        # Interpret result
        sentiment_result = 'Positive' if sentiment[0] == 1 else 'Negative'
        
        return render_template('index.html', prediction_text=f"The sentiment of the review is: {sentiment_result}")

if __name__ == '__main__':
    app.run(debug=True)

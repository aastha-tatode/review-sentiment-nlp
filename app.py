import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Load dataset
df = pd.read_csv("data/reviews.csv")
df = df[['reviews.text', 'reviews.rating']]
df.columns = ['review', 'rating']

# Create sentiment labels
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['rating'].apply(get_sentiment)

# Text preprocessing
stop_words = set(stopwords.words('english'))

# Keep important negation words
important_words = {'not','no','never'}
stop_words = stop_words - important_words

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# Train model
model = LogisticRegression(class_weight='balanced', max_iter=2000)
model.fit(X, y)

# Streamlit UI
st.title("AI Product Review Sentiment Analyzer")

review_input = st.text_area("Enter a product review:")

if st.button("Analyze Sentiment"):

    cleaned = clean_text(review_input)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    # VADER sentiment check
    score = sia.polarity_scores(review_input)["compound"]

    if score < -0.3:
        prediction = "Negative"
    elif score > 0.3:
        prediction = "Positive"

    st.subheader("Predicted Sentiment:")

    if prediction == "Positive":
        st.success("Positive 😊")

    elif prediction == "Negative":
        st.error("Negative 😡")

    else:
        st.warning("Neutral 😐")
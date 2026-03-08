import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
# load dataset
df = pd.read_csv("data/reviews.csv")

# keep only required columns
df = df[['reviews.text','reviews.rating']]

# rename columns
df.columns = ['review','rating']

# show first 5 rows
print(df.head())
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['rating'].apply(get_sentiment)

print(df.head())
print(df['sentiment'].value_counts())
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))

# keep negation words for sentiment
important_words = {'not','no','never'}
stop_words = stop_words - important_words


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

print(df[['review','clean_review']].head())
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X = vectorizer.fit_transform(df['clean_review'])

y = df['sentiment']

print("Feature matrix shape:", X.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(class_weight='balanced', max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='sentiment', data=df)

plt.title("Sentiment Distribution of Reviews")
plt.show()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(df['clean_review'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Most Common Words in Reviews")
plt.show()
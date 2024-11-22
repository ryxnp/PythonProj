import pandas as pd
import re
from textblob import TextBlob
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_dataset():
    try:
        # Load CSV file into a DataFrame
        df = pd.read_csv('iphone_reviews.csv')
        return df  # Return the DataFrame itself for processing
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Initializing the sentiment analyzer function
def analyze_sentiment(review):
    if isinstance(review, str):
        analysis = TextBlob(review)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    else:
        return 'Unknown'

# Load the dataset
df = load_dataset()

# Ensure the DataFrame is not empty before proceeding
if not df.empty:
    # Analyzing sentiments and adding a new column to the DataFrame
    df['Sentiment'] = df['reviewDescription'].apply(analyze_sentiment)

    positive_reviews = df[df['Sentiment'] == 'Positive']
    negative_reviews = df[df['Sentiment'] == 'Negative']

    # Prints how many reviews per category
    print(f"Number of Positive Reviews: {len(positive_reviews)}")
    print(f"Number of Negative Reviews: {len(negative_reviews)}")

    # Initializing lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        if isinstance(text, str):
            # Removing special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Converting to lowercase
            text = text.lower()
            # Tokenization and lemmatization
            tokens = text.split()
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            return ' '.join(tokens)

    # Apply preprocessing to the review descriptions
    df['cleaned_reviews'] = df['reviewDescription'].apply(preprocess_text)

    print(df[['reviewDescription', 'cleaned_reviews']].head())  # Only displays the first 5 cleaned reviews
else:
    print("No data available to process.")
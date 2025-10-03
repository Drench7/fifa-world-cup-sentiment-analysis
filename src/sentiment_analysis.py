import pandas as pd
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
    
    def get_textblob_sentiment(self, text):
        """Get sentiment using TextBlob"""
        analysis = TextBlob(text)
        
        # Classify based on polarity
        if analysis.sentiment.polarity > 0.1:
            return 'positive'
        elif analysis.sentiment.polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiments(self, df, text_column='cleaned_text'):
        """Perform sentiment analysis on the dataset"""
        print("Performing sentiment analysis...")
        
        # TextBlob analysis
        df['sentiment'] = df[text_column].apply(self.get_textblob_sentiment)
        df['polarity'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['subjectivity'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        # Print sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{sentiment}: {count} ({percentage:.2f}%)")
        
        return df
    
    def extract_sentiment_features(self, df, text_column='cleaned_text'):
        """Extract features for machine learning"""
        # Text length features
        df['text_length'] = df[text_column].apply(len)
        df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
        
        # Exclamation and question marks
        df['exclamation_count'] = df[text_column].apply(lambda x: x.count('!'))
        df['question_count'] = df[text_column].apply(lambda x: x.count('?'))
        
        # Capital letters (might indicate excitement)
        df['capital_ratio'] = df[text_column].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )
        
        return df
    
    def prepare_ml_data(self, df, text_column='cleaned_text'):
        """Prepare data for machine learning classification"""
        # Use TextBlob sentiment as labels for training
        X = df[text_column]
        y = df['sentiment']
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_vectorized = self.vectorizer.fit_transform(X)
        
        return X_vectorized, y
    
    def train_ml_model(self, X, y):
        """Train a machine learning model for sentiment classification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nMachine Learning Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model, X_test, y_test, y_pred

def analyze_team_sentiments(df, team_keywords):
    """Analyze sentiments for specific teams"""
    team_sentiments = {}
    
    for team, keywords in team_keywords.items():
        # Filter tweets mentioning the team
        pattern = '|'.join(keywords)
        team_tweets = df[df['cleaned_text'].str.contains(pattern, case=False, na=False)]
        
        if len(team_tweets) > 0:
            sentiment_dist = team_tweets['sentiment'].value_counts(normalize=True) * 100
            team_sentiments[team] = {
                'total_tweets': len(team_tweets),
                'sentiment_distribution': sentiment_dist.to_dict(),
                'average_polarity': team_tweets['polarity'].mean()
            }
    
    return team_sentiments

def analyze_sentiment(df, text_column='cleaned_text'):
    """
    Analyze sentiment of tweets - wrapper function for Streamlit app compatibility
    This function matches what the Streamlit app expects to import
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_sentiments(df, text_column)
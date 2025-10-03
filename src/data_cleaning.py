import pandas as pd
import re
import nltk
import ssl
import streamlit as st # Import Streamlit for deployment best practices

# Fix SSL certificate issues for NLTK download (for local runs)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Global NLTK Resource Management ---
# The original download_nltk_resources function is now removed, 
# as the new clean_text method handles the download attempts inside itself.

# Import after downloading with fallbacks
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    print("✅ NLTK modules imported successfully")
except ImportError as e:
    print(f"❌ NLTK not available: {e}")
    print("Using basic text processing without NLTK")
    NLTK_AVAILABLE = False


class DataCleaner:
    def __init__(self):
        # NOTE: The user's new clean_text method handles stopword and lemmatizer 
        # setup *inside* the cleaning function, making these properties redundant.
        # Original properties like self.stop_words, self.lemmatizer, and self.custom_stop_words are removed.
        pass
        
    # *** START OF USER'S REQUESTED REPLACEMENT CODE ***
    def clean_text(self, text):
        """Clean and preprocess text data with NLTK fallback"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize with robust error handling
        tokens = []
        try:
            # Ensure punkt is available
            # NOTE: This check/download logic runs on every call and is inefficient.
            nltk.data.find('tokenizers/punkt_tab')
            tokens = word_tokenize(text)
        except LookupError:
            try:
                # Try to download if missing
                nltk.download('punkt_tab', quiet=True)
                nltk.download('punkt', quiet=True)
                tokens = word_tokenize(text)
            except:
                # Fallback to simple tokenization
                tokens = text.split()
        
        # Remove stopwords with error handling
        try:
            nltk.data.find('corpora/stopwords')
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
            except:
                # Continue without stopword removal
                pass
        
        return ' '.join(tokens)
    # *** END OF USER'S REQUESTED REPLACEMENT CODE ***

    def find_text_column(self, df):
        """Automatically find the text column in the dataset"""
        # Try exact match first
        if 'Tweet' in df.columns:
            print("Using column 'Tweet' for text analysis")
            return 'Tweet'
        
        possible_columns = ['text', 'tweet', 'content', 'message', 'body', 'description']
        
        for col in possible_columns:
            if col in df.columns:
                print(f"Using column '{col}' for text analysis")
                return col
        
        # If no common names found, look for string columns
        string_columns = df.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            print(f"Using first string column '{string_columns[0]}' for text analysis")
            return string_columns[0]
        
        raise ValueError("No suitable text column found in the dataset")
    
    def clean_dataset(self, df, text_column=None):
        """Clean the entire dataset"""
        print("Cleaning text data...")
        df_clean = df.copy()
        
        # Find text column if not specified
        if text_column is None:
            text_column = self.find_text_column(df)
        elif text_column not in df.columns:
            print(f"Column '{text_column}' not found. Searching for alternative...")
            text_column = self.find_text_column(df)
        
        print(f"Using text column: '{text_column}'")
        
        # Clean text column
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        
        # Process date column if it exists
        if 'Date Created' in df_clean.columns:
            try:
                df_clean['date'] = pd.to_datetime(df_clean['Date Created'])
                print("Date column processed successfully")
            except Exception as e:
                print(f"Could not process date column: {e}")
        
        # Remove empty tweets after cleaning
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
        final_count = len(df_clean)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Cleaned dataset shape: {df_clean.shape}")
        print(f"Removed {initial_count - final_count} empty tweets after cleaning")
        
        return df_clean

def load_and_clean_data(file_path, text_column=None):
    """Load data and apply cleaning"""
    # Load the dataset
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Dataset loaded with shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Clean the data
    df_clean = cleaner.clean_dataset(df, text_column)
    
    return df_clean

def clean_tweet_data(df, text_column=None):
    """
    Clean tweet data - wrapper function for Streamlit app compatibility
    This function matches what the Streamlit app expects to import
    """
    cleaner = DataCleaner()
    return cleaner.clean_dataset(df, text_column)

def clean_tweet_data_simple(df, text_column=None):
    """
    Simple data cleaning without NLTK dependency
    """
    df_clean = df.copy()
    
    # Find text column if not specified
    if text_column is None:
        text_columns = ['Tweet', 'text', 'tweet', 'content']
        for col in text_columns:
            if col in df_clean.columns:
                text_column = col
                break
        if text_column is None:
            # Use first string column
            string_cols = df_clean.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                text_column = string_cols[0]
            else:
                raise ValueError("No text column found")
    
    # Simple text cleaning without NLTK
    def simple_clean(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    df_clean['cleaned_text'] = df_clean[text_column].apply(simple_clean)
    
    return df_clean

if __name__ == "__main__":
    # Test the data cleaner
    sample_text = "Wow! What an amazing goal by Messi! #FIFAWorldCup2022 https://example.com"
    cleaner = DataCleaner()
    # Mocking NLTK available if not running in a full environment
    global NLTK_AVAILABLE
    if 'NLTK_AVAILABLE' not in globals():
        NLTK_AVAILABLE = False
    
    cleaned = cleaner.clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned:", cleaned)
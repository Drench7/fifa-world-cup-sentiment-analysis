import pandas as pd
import re
import nltk
import ssl

# Fix SSL certificate issues for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data with error handling
def download_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords', 
        'wordnet': 'corpora/wordnet'
    }
    
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Could not download {resource}: {e}")
                return False
    return True

# Download resources
download_success = download_nltk_resources()

# Import after downloading
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available, using basic text processing")
    NLTK_AVAILABLE = False

class DataCleaner:
    def __init__(self):
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stop_words = set()
            self.lemmatizer = None
            
        # Football-specific words to keep
        self.football_words = {'goal', 'fifa', 'world', 'cup', 'football', 'soccer', 
                              'match', 'player', 'team', 'win', 'game', 'tournament',
                              'worldcup', 'qatar', 'worldcup2022', 'world cup'}
        self.custom_stop_words = self.stop_words - self.football_words

    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s!?]', '', text)
        
        # Tokenize
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
        
        # Remove stop words
        tokens = [word for word in tokens if word not in self.custom_stop_words and len(word) > 2]
        
        # Lemmatization
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
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

if __name__ == "__main__":
    # Test the data cleaner
    sample_text = "Wow! What an amazing goal by Messi! #FIFAWorldCup2022 https://example.com"
    cleaner = DataCleaner()
    cleaned = cleaner.clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned:", cleaned)
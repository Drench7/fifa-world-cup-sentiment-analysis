# --- Standard Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import re
import logging
from typing import Dict, List
from pathlib import Path
from datetime import datetime

# --- Visualization Imports ---
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# Suppress matplotlib warnings/output which can interfere with Streamlit
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- NLTK Imports and Cloud Deployment Setup (CRITICAL for cloud envs) ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
try:
    from wordcloud import WordCloud
except ImportError:
    st.warning("WordCloud library not found. Please install it (`pip install wordcloud`) to enable word cloud visualizations.")
    WordCloud = None # Define WordCloud as None if import fails

# Cloud deployment setup
# Ensure NLTK data is available for cloud deployment
try:
    # Check for core tokenizers
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    # Setting quiet=True to avoid verbose output during deployment
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True) # VADER lexicon is essential
    nltk.download('wordnet', quiet=True)

# Configuration and Initialization
logging.basicConfig(level=logging.INFO)

# Initialize NLTK components once
analyzer = SentimentIntensityAnalyzer()
STOP_WORDS = set(stopwords.words('english'))

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="FIFA World Cup 2022 Sentiment Analysis",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .positive { color: #2E8B57; }
    .negative { color: #DC143C; }
    .neutral { color: #FFD700; }
</style>
""", unsafe_allow_html=True)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- HELPER FUNCTIONS ---

def find_text_column(df):
    """Find the appropriate text column in the DataFrame"""
    if df is None or df.empty:
        return None
    
    # Prioritize 'cleaned_text' first, then common tweet columns
    text_columns_to_try = ['cleaned_text', 'text', 'Tweet', 'tweet', 'content', 'message', 'review']
    for col in text_columns_to_try:
        if col in df.columns:
            return col
    
    # If no standard text column found, try to find any string column
    string_columns = df.select_dtypes(include=['object', 'string']).columns
    if len(string_columns) > 0:
        return string_columns[0]
    
    return None

def remove_duplicate_columns(df):
    """Remove duplicate columns from DataFrame, keeping the first occurrence."""
    if df is None or df.empty:
        return df
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        duplicate_columns = [col for col in df.columns if list(df.columns).count(col) > 1]
        st.sidebar.warning(f"Found duplicate columns: {duplicate_columns}. Removing duplicates...")
        # Keep only the first occurrence of each column name
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    return df

def safe_dataframe_display(df, columns_to_show=None):
    """Safely display DataFrame columns without causing KeyError and handle duplicates"""
    if df is None or df.empty:
        st.info("No data to display")
        return
    
    # Remove duplicate columns first
    df = remove_duplicate_columns(df)
    
    if columns_to_show:
        # Filter to only existing columns and handle internal duplicates in the requested list
        existing_columns = []
        for col in columns_to_show:
            if col in df.columns and col not in existing_columns:
                existing_columns.append(col)
        
        if existing_columns:
            st.dataframe(df[existing_columns].head(10), use_container_width=True)
        else:
            st.warning(f"None of the requested columns found. Available columns: {list(df.columns)}")
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)

# --- INTEGRATED ANALYSIS FUNCTIONS ---

def clean_tweet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans tweet data, standardizing the text column and performing NLTK-based preprocessing.
    """
    if df is None or df.empty:
        return df
    
    df_clean = df.copy()
    text_column = find_text_column(df_clean)
    
    if text_column:
        # Store the original text column name for reference
        original_text_col = text_column
        
        # If the text column is not named 'text', create a standardized 'text' column
        # but only if 'text' doesn't already exist to avoid duplicates
        if original_text_col != 'text' and 'text' not in df_clean.columns:
            df_clean['text'] = df_clean[original_text_col]
        
        def simple_clean_and_preprocess(text):
            if pd.isna(text) or text is None:
                return ""
            
            text = str(text).lower()
            # Basic cleanup: remove URLs, mentions, and hashtags symbols
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#', '', text)
            
            # Use NLTK word_tokenize to split text
            try:
                tokens = word_tokenize(text)
            except:
                  # Fallback if NLTK data is somehow missing despite the download block
                tokens = text.split() 
            
            # Remove non-alphanumeric (mostly punctuation/special chars) and stop words
            filtered_words = [w for w in tokens if w.isalnum() and w not in STOP_WORDS]
            
            return ' '.join(filtered_words)
            
        # Use the appropriate text column for cleaning
        if 'text' in df_clean.columns:
            df_clean['cleaned_text'] = df_clean['text'].apply(simple_clean_and_preprocess)
        else:
            df_clean['cleaned_text'] = df_clean[original_text_col].apply(simple_clean_and_preprocess)
            
    return df_clean

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs VADER sentiment analysis on the 'cleaned_text' column of the DataFrame.
    """
    if df is None or df.empty or 'cleaned_text' not in df.columns:
        st.error("Cannot perform sentiment analysis: 'cleaned_text' column is missing.")
        return df

    def get_vader_sentiment_label(text):
        if not text:
            return 'neutral', 0.0
        
        score = analyzer.polarity_scores(str(text))
        compound = score['compound']
        
        # VADER threshold definitions
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound

    # Apply the function to the 'cleaned_text' column
    results = df['cleaned_text'].apply(lambda x: get_vader_sentiment_label(x)).apply(pd.Series)
    results.columns = ['sentiment', 'sentiment_score']
    
    # Concatenate the new sentiment columns
    df_result = pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
    
    # Remove any duplicates created during concatenation (defensive)
    df_result = remove_duplicate_columns(df_result)
    
    return df_result


# --- VISUALIZATION FUNCTIONS (Using Streamlit fallbacks, modified slightly) ---

def create_sentiment_chart(df):
    """Creates a Plotly Pie chart for sentiment distribution."""
    if 'sentiment' not in df.columns: 
        return None
    try:
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C', 
                'neutral': '#FFD700'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    except Exception as e:
        st.error(f"Error creating sentiment chart: {e}")
        return None

def create_wordcloud(df, sentiment_type='all'):
    """Creates a Matplotlib-based Word Cloud."""
    if WordCloud is None:
        st.warning("WordCloud library not installed. Please run: pip install wordcloud")
        return None
        
    # Use 'cleaned_text' for the word cloud
    text_column = 'cleaned_text' 
    if text_column not in df.columns or 'sentiment' not in df.columns: 
        return None
    
    try:
        if sentiment_type != 'all':
            text_data = ' '.join(df[df['sentiment'] == sentiment_type][text_column].dropna())
        else:
            text_data = ' '.join(df[text_column].dropna())
        
        if not text_data:
            st.info(f"No text data found for '{sentiment_type}' sentiment.")
            return None

        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            collocations=False # Generally better for word clouds
        ).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {sentiment_type.capitalize()} Sentiment', fontsize=16)
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        return None
    
def create_time_series(df):
    """Creates a Plotly Line chart for sentiment trends over time."""
    date_column = None
    for col in ['date', 'created_at', 'Date Created', 'timestamp']:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None or 'sentiment' not in df.columns: 
        return None
    
    try:
        df_time = df.copy()
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_time[date_column]):
            # Use 'coerce' to handle potential parsing errors gracefully
            df_time[date_column] = pd.to_datetime(df_time[date_column], errors='coerce')
            # Only drop rows where date conversion failed if there are still plenty of rows
            if len(df_time) > len(df_time.dropna(subset=[date_column])):
                st.info(f"Some date values couldn't be parsed. Using {len(df_time.dropna(subset=[date_column]))} valid dates.")
            df_time = df_time.dropna(subset=[date_column])

        if df_time.empty:
            return None

        df_time['date_only'] = df_time[date_column].dt.date
        
        # Group by date and sentiment
        time_series = df_time.groupby(['date_only', 'sentiment']).size().reset_index()
        time_series.columns = ['date', 'sentiment', 'count']
        
        fig = px.line(
            time_series,
            x='date',
            y='count',
            color='sentiment',
            title='Sentiment Trends Over Time',
            labels={'count': 'Number of Tweets', 'date': 'Date'},
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C', 
                'neutral': '#FFD700'
            }
        )
        fig.update_layout(xaxis_title='Date', yaxis_title='Number of Tweets')
        return fig
    except Exception as e:
        st.error(f"Error creating time series: {e}")
        return None

# --- STREAMLIT APP LOGIC ---

# Title and description
st.markdown('<div class="main-header">⚽ FIFA World Cup 2022 Tweet Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown("Analyzing public sentiment from tweets during the FIFA World Cup 2022")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["🏠 Dashboard", "📊 Data Overview", "😊 Sentiment Analysis", "📈 Visualizations", "💡 Insights"]
)

# Load data function
@st.cache_data
def load_data():
    try:
        # Try processed data first
        df = pd.read_csv('data/processed/processed_tweets.csv')
        st.sidebar.success("✅ Loaded processed data")
        return df
    except FileNotFoundError:
        try:
            # Fall back to raw data
            df = pd.read_csv('data/raw/fifa_world_cup_2022_tweets.csv')
            st.sidebar.info("📊 Loaded raw data")
            return df
        except FileNotFoundError:
            st.error("""
            ❌ Data files not found. Using sample data for demonstration.
            """)
            # Create comprehensive sample data
            sample_tweets = [
                "What an amazing match! The World Cup is incredible! ⚽🎉",
                "Terrible refereeing decisions ruining the game 😠",
                "The atmosphere in the stadium is electric tonight!",
                "Disappointing performance from our team today...",
                "GOAL! What a brilliant strike! Beautiful football!",
                "VAR is killing the spirit of the game",
                "Incredible sportsmanship shown by the players 👏",
                "The organization of this World Cup has been poor",
                "What a tournament! Best World Cup ever!",
                "Controversial penalty decision changing the game",
                "The passion of the fans is unbelievable!",
                "Poor quality pitch affecting the game quality",
                "Historic victory for the underdogs! Amazing!",
                "Another boring 0-0 draw...",
                "The opening ceremony was spectacular!",
                "Too many commercial breaks ruining the flow",
                "Young talents shining in this World Cup 🌟",
                "Questionable team selection by the coach",
                "The world is united by football! Beautiful!",
                "Security concerns at the stadium today"
            ]
            sample_data = {
                'text': sample_tweets,
                'created_at': pd.date_range('2022-11-20', periods=len(sample_tweets), freq='H'),
                'user_name': [f'user_{i}' for i in range(len(sample_tweets))],
                'retweet_count': np.random.randint(0, 1000, len(sample_tweets)),
                'favorite_count': np.random.randint(0, 5000, len(sample_tweets))
            }
            df = pd.DataFrame(sample_data)
            st.sidebar.warning("📝 Using sample data for demonstration")
            return df

# Load and process data
df = load_data()

# Remove duplicate columns immediately after loading (initial cleanup)
df = remove_duplicate_columns(df)

# --- SIMPLIFIED COLUMN STANDARDIZATION ---
# Skip complex renaming since data already has good structure
st.sidebar.info("✅ Using optimized column structure")

# Process data (Cleaning and Sentiment Analysis)
if df is not None and ('cleaned_text' not in df.columns or 'sentiment' not in df.columns):
    with st.spinner("Processing data..."):
        try:
            # 1. Cleaning
            df_processed = clean_tweet_data(df)
            
            # 2. Sentiment Analysis
            df_processed = analyze_sentiment(df_processed)
            
            # Remove duplicates from processed data
            df_processed = remove_duplicate_columns(df_processed)
            
            # Verify the processing worked
            if 'sentiment' in df_processed.columns and 'cleaned_text' in df_processed.columns:
                df = df_processed
                st.sidebar.success("✅ Data cleaned and sentiment analyzed successfully using NLTK VADER.")
            else:
                st.sidebar.error("⚠️ Data cleaning or Sentiment analysis failed to generate required columns.")
                df = df_processed
        except Exception as e:
            st.error(f"Error during core data processing: {e}")
            st.info("Continuing with raw data, visualizations may fail.")

# Debug information
with st.sidebar.expander("🔧 Debug Info"):
    st.write("DataFrame shape:", df.shape if df is not None else "No data")
    if df is not None:
        st.write("Columns:", df.columns.tolist())
        st.write("Duplicate columns:", [col for col in df.columns if list(df.columns).count(col) > 1])
        st.write("Sentiment column exists:", 'sentiment' in df.columns)
        text_col = find_text_column(df)
        st.write("Text column found:", text_col)
        
        # Show available engagement metrics
        engagement_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                          ['retweet', 'like', 'favorite', 'share', 'engagement'])]
        st.write("Engagement columns:", engagement_cols)

# --- APP SECTIONS ---

if section == "🏠 Dashboard":
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    if df is not None and 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        total_tweets = len(df)
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
    else:
        total_tweets = len(df) if df is not None else 0
        positive_count = negative_count = neutral_count = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tweets", f"{total_tweets:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card positive">', unsafe_allow_html=True)
        st.metric("Positive Tweets", f"{positive_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card negative">', unsafe_allow_html=True)
        st.metric("Negative Tweets", f"{negative_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card neutral">', unsafe_allow_html=True)
        st.metric("Neutral Tweets", f"{neutral_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    if df is not None and 'sentiment' in df.columns:
        fig = create_sentiment_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not generate sentiment chart")
    else:
        st.warning("No sentiment data available for visualization. Check data processing.")
    
    # Recent tweets preview
    st.subheader("Recent Tweets Preview")
    safe_dataframe_display(df, ['text', 'Tweet', 'sentiment', 'sentiment_score'])

elif section == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    if df is None:
        st.error("No data available")
        st.stop()
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    # Find date column for date range
    date_column = None
    for col in ['date', 'created_at', 'Date Created']:
        if col in df.columns:
            date_column = col
            break
            
    with col1:
        st.info(f"**Total Records:** {len(df):,}")
        st.info(f"**Columns:** {len(df.columns)}")
    
    with col2:
        if date_column and pd.api.types.is_datetime64_any_dtype(df[date_column]):
            date_min = df[date_column].min()
            date_max = df[date_column].max()
            if hasattr(date_min, 'date') and hasattr(date_max, 'date'):
                st.info(f"**Date Range:** {date_min.date()} to {date_max.date()}")
            else:
                st.info(f"**Date Range:** {date_min} to {date_max}")
        else:
            st.info("**Date Range:** N/A (Date column not found or formatted)")
        st.info(f"**Missing Values:** {df.isnull().sum().sum()}")
        
    st.subheader("Data Sample (First 10 Rows)")
    safe_dataframe_display(df)
    
    st.subheader("Column Details")
    for col in df.columns:
        with st.expander(f"📁 {col} ({df[col].dtype})"):
            st.write(f"Unique values: {df[col].nunique()}")
            st.write(f"Missing values: {df[col].isnull().sum()}")
            if df[col].dtype in ['object', 'string']:
                st.write("Sample values:")
                sample_values = df[col].dropna().head(5).tolist()
                for i, val in enumerate(sample_values):
                    st.write(f"{i+1}. {str(val)[:100]}{'...' if len(str(val)) > 100 else ''}")
            elif pd.api.types.is_numeric_dtype(df[col]):
                st.write(f"Mean: {df[col].mean():.2f}")
                st.write(f"Min: {df[col].min():.2f}")
                st.write(f"Max: {df[col].max():.2f}")

elif section == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis Overview")
    
    if df is None or 'sentiment' not in df.columns:
        st.error("No sentiment data available. Please check data processing steps.")
        st.stop()
    
    # Sentiment statistics
    sentiment_counts = df['sentiment'].value_counts()
    total_tweets = len(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card positive">', unsafe_allow_html=True)
        st.metric(
            "Positive Sentiment", 
            f"{sentiment_counts.get('positive', 0):,}",
            f"{(sentiment_counts.get('positive', 0)/total_tweets*100):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card negative">', unsafe_allow_html=True)
        st.metric(
            "Negative Sentiment", 
            f"{sentiment_counts.get('negative', 0):,}",
            f"{(sentiment_counts.get('negative', 0)/total_tweets*100):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card neutral">', unsafe_allow_html=True)
        st.metric(
            "Neutral Sentiment", 
            f"{sentiment_counts.get('neutral', 0):,}",
            f"{(sentiment_counts.get('neutral', 0)/total_tweets*100):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sentiment exploration
    st.subheader("Explore Tweets by Sentiment")
    
    sentiment_filter = st.selectbox(
        "Select sentiment to explore:",
        ["all", "positive", "negative", "neutral"]
    )
    
    if sentiment_filter != "all":
        filtered_df = df[df['sentiment'] == sentiment_filter]
    else:
        filtered_df = df
    
    st.write(f"Showing {len(filtered_df):,} tweets")
    
    # Use available text columns
    display_columns = []
    text_cols = ['text', 'Tweet', 'cleaned_text']
    for col in text_cols:
        if col in filtered_df.columns:
            display_columns.append(col)
            break
    
    display_columns.extend(['sentiment', 'sentiment_score'])
    
    # Add user and date if available
    if 'user_name' in filtered_df.columns:
        display_columns.append('user_name')
    if 'date' in filtered_df.columns:
        display_columns.append('date')
    elif 'Date Created' in filtered_df.columns:
        display_columns.append('Date Created')
    
    safe_dataframe_display(filtered_df, display_columns)

elif section == "📈 Visualizations":
    st.header("📈 Visualizations")
    
    if df is None:
        st.error("No data available for visualization")
        st.stop()
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose visualization:",
        ["Sentiment Distribution", "Word Cloud", "Time Series", "All Visualizations"]
    )
    
    if viz_type == "Sentiment Distribution" or viz_type == "All Visualizations":
        st.subheader("Sentiment Distribution")
        if 'sentiment' in df.columns:
            fig = create_sentiment_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not generate sentiment distribution chart")
        else:
            st.warning("No sentiment data available for this visualization")
    
    if viz_type == "Word Cloud" or viz_type == "All Visualizations":
        st.subheader("Word Clouds by Sentiment (Based on Cleaned Text)")
        
        if 'cleaned_text' in df.columns and 'sentiment' in df.columns:
            if WordCloud is None:
                st.warning("WordCloud library is not installed. Please run: `pip install wordcloud`")
            else:
                wc_sentiment = st.selectbox(
                    "Select sentiment for word cloud:",
                    ["all", "positive", "negative", "neutral"],
                    key='wc_select'
                )
                
                fig = create_wordcloud(df, wc_sentiment)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("Word cloud cannot be generated with current data.")
        else:
            st.warning("No cleaned text or sentiment data available for word cloud generation.")
    
    if viz_type == "Time Series" or viz_type == "All Visualizations":
        st.subheader("Sentiment Trends Over Time")
        
        date_column = None
        for col in ['date', 'created_at', 'Date Created']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column and 'sentiment' in df.columns:
            time_fig = create_time_series(df)
            if time_fig:
                st.plotly_chart(time_fig, use_container_width=True)
            else:
                st.info("Could not generate time series chart. Check if date column contains valid dates.")
        else:
            st.warning("No appropriate date column found or sentiment data is missing.")

elif section == "💡 Insights":
    st.header("💡 Key Insights")
    
    if df is None:
        st.error("No data available for insights")
        st.stop()
    
    # Calculate insights
    total_tweets = len(df)
    
    if 'sentiment' in df.columns:
        positive_pct = (df['sentiment'] == 'positive').sum() / total_tweets * 100
        negative_pct = (df['sentiment'] == 'negative').sum() / total_tweets * 100
        neutral_pct = (df['sentiment'] == 'neutral').sum() / total_tweets * 100
        sentiment_counts = df['sentiment'].value_counts()
    else:
        positive_pct = negative_pct = neutral_pct = 0
        sentiment_counts = pd.Series(dtype='int64')

    # Calculate engagement metrics from available columns
    retweet_col = None
    favorite_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'retweet' in col_lower or 'rt' in col_lower:
            retweet_col = col
        elif 'favorite' in col_lower or 'like' in col_lower or 'fav' in col_lower:
            favorite_col = col
    
    avg_retweets = df[retweet_col].mean() if retweet_col and retweet_col in df.columns else 0
    avg_favorites = df[favorite_col].mean() if favorite_col and favorite_col in df.columns else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Overview")
        st.metric("Overall Positive Sentiment", f"{positive_pct:.1f}%")
        st.metric("Overall Negative Sentiment", f"{negative_pct:.1f}%")
        st.metric("Overall Neutral Sentiment", f"{neutral_pct:.1f}%")
        
        # Sentiment summary
        if positive_pct > negative_pct and positive_pct > neutral_pct:
            st.success("🎉 Overall public sentiment leaning **POSITIVE** towards the World Cup.")
        elif negative_pct > positive_pct and negative_pct > neutral_pct:
            st.error("⚠️ Overall public sentiment leaning **NEGATIVE**. Investigate key negative topics (check Negative Word Cloud).")
        else:
            st.info("⚖️ Sentiment is largely **NEUTRAL** or balanced, indicating balanced discussion or lack of strong opinion.")
    
    with col2:
        st.subheader("📊 Engagement Metrics")
        
        if retweet_col and retweet_col in df.columns:
            st.metric("Average Retweets", f"{avg_retweets:.1f}")
        else:
            st.info("Retweet data not available")
        
        if favorite_col and favorite_col in df.columns:
            st.metric("Average Favorites/Likes", f"{avg_favorites:.1f}")
        else:
            st.info("Favorite/Like data not available")
        
        text_column = find_text_column(df)
        if text_column and total_tweets > 0:
            word_count = df[text_column].astype(str).str.split().str.len().sum()
            st.metric("Total Words Analyzed", f"{word_count:,}")
    
    st.subheader("💡 Strategic Recommendations")
    
    with st.expander("Recommendations based on Sentiment Balance"):
        st.markdown("""
        **Actionable Steps:**
        
        * **For Positive Peaks (Check Time Series):** Identify the **events/matches** that drove the most positive engagement and replicate successful communication strategies around those topics.
        * **For Negative Spikes:** Immediately analyze the **negative word cloud** during those time periods to pinpoint specific issues (e.g., "referee," "VAR," "organization") and formulate targeted responses.
        * **Content Strategy:** Use the **positive word cloud** to guide your team's content creation, focusing on terms and themes that resonate best with the audience.
        * **Neutral Engagement:** For neutral sentiment, focus on providing more information and context to help users form stronger opinions.
        """)
    
    # Export options
    st.subheader("📤 Export Results")
    if st.button("Generate Analysis Report"):
        with st.spinner("Generating report..."):
            overall_sentiment = 'POSITIVE' if positive_pct > negative_pct and positive_pct > neutral_pct else 'NEGATIVE' if negative_pct > positive_pct and negative_pct > neutral_pct else 'NEUTRAL'
            
            report = f"""
FIFA WORLD CUP 2022 SENTIMENT ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Total Tweets Analyzed: {total_tweets:,}
- Overall Sentiment: {overall_sentiment}
- Positive Sentiment: {positive_pct:.1f}% ({sentiment_counts.get('positive', 0):,} tweets)
- Negative Sentiment: {negative_pct:.1f}% ({sentiment_counts.get('negative', 0):,} tweets)
- Neutral Sentiment: {neutral_pct:.1f}% ({sentiment_counts.get('neutral', 0):,} tweets)
- Average Retweets: {avg_retweets:.1f}
- Average Favorites/Likes: {avg_favorites:.1f}

KEY TAKEAWAYS:
- The analysis provides valuable insights into public perception using NLTK VADER.
- Focus on amplifying the positive themes identified in the word clouds.
- Proactively address the issues highlighted by negative sentiment spikes.
- Use time series analysis to correlate sentiment with specific World Cup events.

RECOMMENDATIONS:
1. Leverage positive sentiment peaks for marketing and engagement
2. Address negative sentiment drivers through targeted communication
3. Monitor neutral sentiment for opportunities to provide more information
4. Use word cloud insights to guide content strategy
            """
            
            st.download_button(
                label="Download Full Summary Report (TXT)",
                data=report,
                file_name=f"fifa_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**FIFA World Cup 2022 Sentiment Analysis**

This app analyzes public sentiment from social media using the **NLTK VADER** lexicon. It provides real-time insights into fan engagement and perception during the tournament.
""")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data (Clear Cache)"):
    st.cache_data.clear()
    st.rerun()
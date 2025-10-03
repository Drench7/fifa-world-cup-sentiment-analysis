import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="FIFA World Cup 2022 Sentiment Analysis",
    page_icon="⚽",
    layout="wide"
)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your custom modules with better error handling
def import_custom_modules():
    """Import custom modules with fallback functions"""
    try:
        # Try to import the simple version first (no NLTK dependency)
        from src.data_cleaning import clean_tweet_data_simple
        st.success("✅ Successfully imported clean_tweet_data_simple")
        return clean_tweet_data_simple
    except ImportError as e:
        st.warning(f"❌ Could not import clean_tweet_data_simple: {e}")
        
        try:
            # Fall back to regular version (with NLTK dependency)
            from src.data_cleaning import clean_tweet_data
            st.success("✅ Successfully imported clean_tweet_data")
            return clean_tweet_data
        except ImportError as e:
            st.warning(f"❌ Could not import clean_tweet_data: {e}")
            
            # Define ultimate fallback function (basic cleaning logic)
            def clean_tweet_data_fallback(df):
                """Fallback data cleaning function"""
                try:
                    df_clean = df.copy()
                    # Find text column
                    text_columns_to_try = ['text', 'Tweet', 'tweet', 'content', 'message']
                    text_column = None
                    
                    for col in text_columns_to_try:
                        if col in df_clean.columns:
                            text_column = col
                            break
                    
                    if text_column:
                        # Simple cleaning without any external dependencies
                        def simple_clean(text):
                            if pd.isna(text):
                                return ""
                            text = str(text).lower()
                            text = re.sub(r'http\S+', '', text)
                            text = re.sub(r'@\w+', '', text)
                            text = re.sub(r'#', '', text)
                            text = re.sub(r'[^A-Za-z\s]', '', text)
                            text = ' '.join(text.split())
                            return text
                        
                        df_clean['cleaned_text'] = df_clean[text_column].apply(simple_clean)
                    return df_clean
                except Exception as e:
                    st.error(f"Error in data cleaning: {e}")
                    return df
            
            return clean_tweet_data_fallback

def import_sentiment_analysis():
    """Import sentiment analysis with fallback"""
    try:
        from src.sentiment_analysis import analyze_sentiment
        st.success("✅ Successfully imported analyze_sentiment")
        return analyze_sentiment
    except ImportError as e:
        st.warning(f"❌ Could not import analyze_sentiment: {e}")
        
        # Fallback sentiment analysis
        def analyze_sentiment_fallback(df):
            try:
                from textblob import TextBlob
                df_sentiment = df.copy()
                
                # Find text column
                text_column = None
                for col in ['cleaned_text', 'text', 'Tweet', 'tweet']:
                    if col in df_sentiment.columns:
                        text_column = col
                        break
                
                if text_column:
                    def get_sentiment(text):
                        try:
                            analysis = TextBlob(str(text))
                            polarity = analysis.sentiment.polarity
                            if polarity > 0.1:
                                return 'positive'
                            elif polarity < -0.1:
                                return 'negative'
                            else:
                                return 'neutral'
                        except:
                            return 'neutral'
                    
                    df_sentiment['sentiment'] = df_sentiment[text_column].apply(get_sentiment)
                    df_sentiment['sentiment_score'] = df_sentiment[text_column].apply(
                        lambda x: TextBlob(str(x)).sentiment.polarity
                    )
                return df_sentiment
            except Exception as e:
                st.error(f"Error in sentiment analysis: {e}")
                return df
        
        return analyze_sentiment_fallback

def import_visualization():
    """Import visualization functions with fallback"""
    try:
        from src.visualization import create_sentiment_chart, create_wordcloud, create_time_series
        st.success("✅ Successfully imported visualization functions")
        return create_sentiment_chart, create_wordcloud, create_time_series
    except ImportError as e:
        st.warning(f"❌ Could not import visualization functions: {e}")
        
        # Fallback visualization functions
        def create_sentiment_chart_fallback(df):
            try:
                if 'sentiment' in df.columns:
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
            return None # Added explicit return
        
        def create_wordcloud_fallback(df, sentiment_type='all'):
            try:
                from wordcloud import WordCloud
                # Find text column
                text_column = None
                for col in ['cleaned_text', 'text', 'Tweet', 'tweet']:
                    if col in df.columns:
                        text_column = col
                        break
                
                if text_column and 'sentiment' in df.columns:
                    if sentiment_type != 'all':
                        text_data = ' '.join(df[df['sentiment'] == sentiment_type][text_column].dropna())
                    else:
                        text_data = ' '.join(df[text_column].dropna())
                    
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='viridis'
                    ).generate(text_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Word Cloud - {sentiment_type.capitalize()} Sentiment', fontsize=16)
                    return fig
            except Exception as e:
                st.error(f"Error creating word cloud: {e}")
                return None
            return None # Added explicit return
        
        def create_time_series_fallback(df):
            try:
                # Find date column
                date_column = None
                for col in ['date', 'created_at', 'Date Created', 'timestamp']:
                    if col in df.columns:
                        date_column = col
                        break
                
                if date_column and 'sentiment' in df.columns:
                    df_time = df.copy()
                    if not pd.api.types.is_datetime64_any_dtype(df_time[date_column]):
                        df_time[date_column] = pd.to_datetime(df_time[date_column])
                    
                    df_time['date_only'] = df_time[date_column].dt.date
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
            return None # Added explicit return
        
        return create_sentiment_chart_fallback, create_wordcloud_fallback, create_time_series_fallback

# Import all functions
clean_tweet_data = import_custom_modules()
analyze_sentiment = import_sentiment_analysis()
create_sentiment_chart, create_wordcloud, create_time_series = import_visualization()

# NLTK data handling for cloud deployment
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('movie_reviews', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    st.warning("NLTK data download failed, but app will continue...")

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
    }
    .positive { color: #2E8B57; }
    .negative { color: #DC143C; }
    .neutral { color: #FFD700; }
</style>
""", unsafe_allow_html=True)

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
            ❌ Data files not found. Please make sure you have:
            - `data/raw/fifa_world_cup_2022_tweets.csv` or
            - `data/processed/processed_tweets.csv`
            
            Using sample data for demonstration.
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

# Process data if needed
if 'cleaned_text' not in df.columns or 'sentiment' not in df.columns:
    with st.spinner("Processing data..."):
        df_processed = clean_tweet_data(df)
        df_processed = analyze_sentiment(df_processed)
        df = df_processed

# Helper function to find text column
def find_text_column(df):
    """Find the appropriate text column in the DataFrame"""
    text_columns_to_try = ['cleaned_text', 'text', 'Tweet', 'tweet', 'content']
    for col in text_columns_to_try:
        if col in df.columns:
            return col
    return None

if section == "🏠 Dashboard":
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tweets", len(df))
    
    with col2:
        positive_count = len(df[df['sentiment'] == 'positive'])
        st.metric("Positive Tweets", positive_count)
    
    with col3:
        negative_count = len(df[df['sentiment'] == 'negative'])
        st.metric("Negative Tweets", negative_count)
    
    with col4:
        neutral_count = len(df[df['sentiment'] == 'neutral'])
        st.metric("Neutral Tweets", neutral_count)
    
    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    fig = create_sentiment_chart(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent tweets preview
    st.subheader("Recent Tweets Preview")
    
    # Find the correct text column
    text_column = find_text_column(df)
    
    if text_column:
        # Show available columns for context
        available_columns = [text_column, 'sentiment']
        if 'sentiment_score' in df.columns:
            available_columns.append('sentiment_score')
        
        st.dataframe(df[available_columns].head(10), use_container_width=True)
    else:
        st.warning("No text column found. Showing all columns:")
        st.dataframe(df.head(10), use_container_width=True)

elif section == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Total Records:** {len(df)}")
        st.info(f"**Columns:** {len(df.columns)}")
    
    with col2:
        # Find date column for date range
        date_column = None
        for col in ['date', 'created_at', 'Date Created']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column:
            st.info(f"**Date Range:** {df[date_column].min()} to {df[date_column].max()}")
        else:
            st.info("**Date Range:** N/A")
        
        st.info(f"**Missing Values:** {df.isnull().sum().sum()}")
    
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Column Details")
    for col in df.columns:
        with st.expander(f"📁 {col} ({df[col].dtype})"):
            st.write(f"Unique values: {df[col].nunique()}")
            st.write(f"Missing values: {df[col].isnull().sum()}")
            if df[col].dtype in ['object', 'string']:
                st.write("Sample values:")
                st.write(df[col].value_counts().head())

elif section == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    
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
    
    # Find text column for display
    text_column = find_text_column(filtered_df)
    
    st.write(f"Showing {len(filtered_df)} tweets")
    
    if text_column:
        display_columns = [text_column, 'sentiment']
        if 'sentiment_score' in filtered_df.columns:
            display_columns.append('sentiment_score')
        
        st.dataframe(filtered_df[display_columns].head(20), use_container_width=True)
    else:
        st.dataframe(filtered_df.head(20), use_container_width=True)

elif section == "📈 Visualizations":
    st.header("📈 Visualizations")
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose visualization:",
        ["Sentiment Distribution", "Word Cloud", "Time Series", "All Visualizations"]
    )
    
    if viz_type == "Sentiment Distribution" or viz_type == "All Visualizations":
        st.subheader("Sentiment Distribution")
        fig = create_sentiment_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    if viz_type == "Word Cloud" or viz_type == "All Visualizations":
        st.subheader("Word Clouds by Sentiment")
        
        wc_sentiment = st.selectbox(
            "Select sentiment for word cloud:",
            ["all", "positive", "negative", "neutral"]
        )
        
        fig = create_wordcloud(df, wc_sentiment)
        if fig:
            st.pyplot(fig)
        else:
            st.info("Word cloud cannot be generated with current data.")
    
    if viz_type == "Time Series" or viz_type == "All Visualizations":
        st.subheader("Sentiment Trends Over Time")
        # Find date column
        date_column = None
        for col in ['date', 'created_at', 'Date Created']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column:
            time_fig = create_time_series(df)
            if time_fig:
                st.plotly_chart(time_fig, use_container_width=True)
            else:
                st.info("Time series visualization requires date information.")
        else:
            st.warning("No date column found for time series analysis.")

elif section == "💡 Insights":
    st.header("💡 Key Insights")
    
    # Calculate insights
    total_tweets = len(df)
    positive_pct = (df['sentiment'] == 'positive').sum() / total_tweets * 100
    negative_pct = (df['sentiment'] == 'negative').sum() / total_tweets * 100
    neutral_pct = (df['sentiment'] == 'neutral').sum() / total_tweets * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Overview")
        st.metric("Overall Positive Sentiment", f"{positive_pct:.1f}%")
        st.metric("Overall Negative Sentiment", f"{negative_pct:.1f}%")
        st.metric("Overall Neutral Sentiment", f"{neutral_pct:.1f}%")
        
        # Sentiment summary
        if positive_pct > negative_pct:
            st.success("🎉 Overall sentiment is **POSITIVE**")
        elif negative_pct > positive_pct:
            st.error("⚠️ Overall sentiment is **NEGATIVE**")
        else:
            st.info("⚖️ Overall sentiment is **NEUTRAL**")
    
    with col2:
        st.subheader("📊 Engagement Metrics")
        if 'retweet_count' in df.columns:
            avg_retweets = df['retweet_count'].mean()
            st.metric("Average Retweets", f"{avg_retweets:.1f}")
        
        if 'favorite_count' in df.columns:
            avg_favorites = df['favorite_count'].mean()
            st.metric("Average Favorites", f"{avg_favorites:.1f}")
        
        # Most common words (simplified)
        text_column = find_text_column(df)
        if text_column:
            word_count = df[text_column].str.split().str.len().sum()
            st.metric("Total Words Analyzed", f"{word_count:,}")
    
    st.subheader("🔍 Detailed Analysis")
    
    # Sentiment by time (if available)
    date_column = None
    for col in ['date', 'created_at', 'Date Created']:
        if col in df.columns:
            date_column = col
            break
    
    if date_column:
        with st.expander("📅 Temporal Analysis"):
            st.write("Sentiment patterns over time can reveal key moments during the tournament.")
            # Add more temporal insights here
    
    # Recommendations
    with st.expander("💡 Recommendations"):
        st.markdown("""
        **Based on the sentiment analysis:**
        
        ✅ **For Positive Sentiment:**
        - Leverage positive moments for marketing campaigns
        - Highlight successful events and achievements
        - Engage with positive community content
        
        ⚠️ **For Negative Sentiment:**
        - Monitor spikes for crisis management
        - Address common concerns proactively
        - Improve communication on contentious issues
        
        📊 **General Recommendations:**
        - Continue monitoring sentiment in real-time
        - Compare with other tournaments for benchmarking
        - Use insights for future event planning
        """)
    
    # Export options
    st.subheader("📤 Export Results")
    if st.button("Generate Analysis Report"):
        with st.spinner("Generating report..."):
            # Create a simple report
            report = f"""
            FIFA WORLD CUP 2022 SENTIMENT ANALYSIS REPORT
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            SUMMARY:
            - Total Tweets Analyzed: {total_tweets:,}
            - Positive Sentiment: {positive_pct:.1f}%
            - Negative Sentiment: {negative_pct:.1f}%
            - Neutral Sentiment: {neutral_pct:.1f}%
            
            KEY FINDINGS:
            - Overall sentiment is {'positive' if positive_pct > negative_pct else 'negative' if negative_pct > positive_pct else 'neutral'}
            - The analysis provides valuable insights into public perception
            - Useful for strategic planning and community engagement
            
            RECOMMENDATIONS:
            - Focus on amplifying positive aspects
            - Address concerns raised in negative feedback
            - Use insights for future tournament planning
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"fifa_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**FIFA World Cup 2022 Sentiment Analysis**

Analyzing public sentiment from social media to understand fan engagement and perception during the tournament.
""")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

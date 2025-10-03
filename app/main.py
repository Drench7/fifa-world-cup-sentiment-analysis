import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_cleaning import clean_tweet_data
from src.sentiment_analysis import analyze_sentiment
from src.visualization import create_sentiment_chart, create_wordcloud, create_time_series

# App configuration
st.set_page_config(
    page_title="FIFA World Cup 2022 Sentiment Analysis",
    page_icon="⚽",
    layout="wide"
)

# Title and description
st.title("⚽ FIFA World Cup 2022 Tweet Sentiment Analysis")
st.markdown("Analyzing public sentiment from tweets during the FIFA World Cup 2022")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Data Overview", "Sentiment Analysis", "Visualizations", "Insights"]
)

# Load data
@st.cache_data
def load_data():
    try:
        # Try processed data first
        df = pd.read_csv('data/processed/processed_tweets.csv')
    except:
        # Fall back to raw data and process
        df = pd.read_csv('data/raw/fifa_world_cup_2022_tweets.csv')
        df = clean_tweet_data(df)
        df = analyze_sentiment(df)
    return df

df = load_data()

if section == "Data Overview":
    st.header("📊 Data Overview")
    
    st.subheader("Dataset Info")
    st.write(f"Total tweets: {len(df)}")
    st.write(f"Columns: {list(df.columns)}")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    st.subheader("Data Description")
    st.write(df.describe())

elif section == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive Tweets", sentiment_counts.get('positive', 0))
        with col2:
            st.metric("Negative Tweets", sentiment_counts.get('negative', 0))
        with col3:
            st.metric("Neutral Tweets", sentiment_counts.get('neutral', 0))
        
        st.subheader("Sentiment Distribution")
        st.bar_chart(sentiment_counts)
        
        # Show sample tweets by sentiment
        sentiment_filter = st.selectbox("Filter by sentiment:", ["all", "positive", "negative", "neutral"])
        
        if sentiment_filter != "all":
            filtered_df = df[df['sentiment'] == sentiment_filter]
        else:
            filtered_df = df
            
        st.dataframe(filtered_df[['text', 'sentiment']].head(10))

elif section == "Visualizations":
    st.header("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        fig = create_sentiment_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Over Time")
        if 'created_at' in df.columns:
            time_fig = create_time_series(df)
            st.plotly_chart(time_fig, use_container_width=True)
    
    st.subheader("Word Cloud")
    if st.button("Generate Word Cloud"):
        wc_fig = create_wordcloud(df)
        st.pyplot(wc_fig)

elif section == "Insights":
    st.header("💡 Key Insights")
    
    st.subheader("Overall Sentiment")
    if 'sentiment' in df.columns:
        positive_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
        negative_pct = (df['sentiment'] == 'negative').sum() / len(df) * 100
        
        st.write(f"✅ **Positive sentiment**: {positive_pct:.1f}%")
        st.write(f"❌ **Negative sentiment**: {negative_pct:.1f}%")
        st.write(f"⚪ **Neutral sentiment**: {100 - positive_pct - negative_pct:.1f}%")
    
    st.subheader("Top Keywords")
    # Add your keyword extraction logic here
    
    st.subheader("Recommendations")
    st.info("""
    Based on the sentiment analysis:
    - Monitor negative sentiment spikes for crisis management
    - Leverage positive sentiment for marketing campaigns
    - Identify key topics driving engagement
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit • FIFA World Cup 2022 Analysis")
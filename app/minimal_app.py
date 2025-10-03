import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="FIFA WC Sentiment - Minimal",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FIFA World Cup 2022 Sentiment Analysis")
st.success("🚀 App deployed successfully!")

# Simple sample data
sample_tweets = [
    "What an amazing match! The World Cup is incredible! ⚽🎉",
    "Terrible refereeing decisions ruining the game 😠", 
    "The atmosphere in the stadium is electric tonight!",
    "Disappointing performance from our team today...",
    "GOAL! What a brilliant strike! Beautiful football!"
]

sample_data = {
    'Tweet': sample_tweets,
    'Sentiment': ['positive', 'negative', 'positive', 'negative', 'positive'],
    'Likes': [100, 50, 200, 75, 150]
}

df = pd.DataFrame(sample_data)

st.subheader("Sample Tweets Analysis")
st.dataframe(df)

# Simple metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Tweets", len(df))
with col2:
    st.metric("Positive", len(df[df['Sentiment'] == 'positive']))
with col3:
    st.metric("Negative", len(df[df['Sentiment'] == 'negative']))

st.info("This is a minimal version for testing deployment. Full features coming soon!")
"""
Utility functions for Football Sentiment Analysis Project
Author: Tafadzwa Marisa
Education: BCA Big Data Analytics, LPU India
Nationality: Zimbabwean
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_project_report(df, team_sentiments):
    """Generate comprehensive project report"""
    
    # Basic metrics
    total_tweets = len(df)
    sentiment_distribution = df['sentiment'].value_counts().to_dict()
    avg_polarity = df['polarity'].mean()
    avg_subjectivity = df['subjectivity'].mean()
    
    # Team analysis
    most_discussed = max(team_sentiments.items(), key=lambda x: x[1]['total_tweets'])[0] if team_sentiments else 'N/A'
    most_positive = max(team_sentiments.items(), key=lambda x: x[1]['average_polarity'])[0] if team_sentiments else 'N/A'
    
    report = {
        'project_title': 'FIFA World Cup 2022 Twitter Sentiment Analysis',
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_tweets_analyzed': total_tweets,
        'sentiment_distribution': sentiment_distribution,
        'average_polarity': avg_polarity,
        'average_subjectivity': avg_subjectivity,
        'most_discussed_team': most_discussed,
        'most_positive_team': most_positive,
        'data_sources': ['Kaggle - FIFA World Cup 2022 Tweets Dataset'],
        'technologies_used': [
            'Python', 'Pandas', 'NumPy', 'TextBlob', 
            'Matplotlib', 'Seaborn', 'Scikit-learn', 'NLTK'
        ]
    }
    
    return report

def calculate_engagement_metrics(df):
    """Calculate social media engagement metrics"""
    if 'Number of Likes' in df.columns:
        engagement = {
            'total_likes': int(df['Number of Likes'].sum()),
            'avg_likes_per_tweet': float(df['Number of Likes'].mean()),
            'max_likes': int(df['Number of Likes'].max()),
            'likes_by_sentiment': df.groupby('sentiment')['Number of Likes'].mean().to_dict()
        }
        return engagement
    return {'message': 'No engagement data available'}

def export_insights(df, team_sentiments, output_path='../outputs/'):
    """Export key insights for presentation"""
    os.makedirs(output_path, exist_ok=True)
    
    # Key insights
    insights = []
    
    # Sentiment insights
    dominant_sentiment = df['sentiment'].value_counts().idxmax()
    dominant_percentage = (df['sentiment'].value_counts().max() / len(df)) * 100
    insights.append(f"Dominant sentiment was {dominant_sentiment} ({dominant_percentage:.1f}% of tweets)")
    
    # Team insights
    if team_sentiments:
        most_discussed = max(team_sentiments.items(), key=lambda x: x[1]['total_tweets'])
        insights.append(f"Most discussed team: {most_discussed[0]} ({most_discussed[1]['total_tweets']} mentions)")
        
        most_positive = max(team_sentiments.items(), key=lambda x: x[1]['average_polarity'])
        insights.append(f"Most positive team: {most_positive[0]} (polarity: {most_positive[1]['average_polarity']:.3f})")
    
    # Engagement insights
    if 'Number of Likes' in df.columns:
        avg_likes = df['Number of Likes'].mean()
        insights.append(f"Average engagement: {avg_likes:.1f} likes per tweet")
    
    # Write to file
    with open(f'{output_path}key_insights.txt', 'w', encoding='utf-8') as f:
        f.write("KEY INSIGHTS - FIFA World Cup 2022 Twitter Analysis\n")
        f.write("=" * 50 + "\n\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
    
    print(f"✅ Insights exported to: {output_path}key_insights.txt")
    return insights

def save_analysis_summary(df, team_sentiments, file_path='../outputs/analysis_summary.txt'):
    """Save comprehensive analysis summary"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("FIFA WORLD CUP 2022 TWITTER SENTIMENT ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Tweets Analyzed: {len(df):,}\n")
        f.write(f"Analysis Period: {df['date'].min().date()} to {df['date'].max().date()}\n")
        f.write(f"Dominant Sentiment: {df['sentiment'].value_counts().idxmax()}\n\n")
        
        f.write("SENTIMENT DISTRIBUTION\n")
        f.write("-" * 25 + "\n")
        for sentiment, count in df['sentiment'].value_counts().items():
            percentage = (count / len(df)) * 100
            f.write(f"{sentiment.capitalize()}: {count} tweets ({percentage:.1f}%)\n")
        
        f.write("\nTOP TEAMS ANALYSIS\n")
        f.write("-" * 20 + "\n")
        if team_sentiments:
            sorted_teams = sorted(team_sentiments.items(), key=lambda x: x[1]['total_tweets'], reverse=True)[:5]
            for team, data in sorted_teams:
                f.write(f"{team}: {data['total_tweets']} tweets, {data['positive_percentage']:.1f}% positive\n")
        
        f.write("\nTECHNICAL DETAILS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Polarity: {df['polarity'].mean():.3f}\n")
        f.write(f"Average Subjectivity: {df['subjectivity'].mean():.3f}\n")
        f.write(f"Average Text Length: {df['text_length'].mean():.1f} characters\n")
        
        if 'Number of Likes' in df.columns:
            f.write(f"Total Likes: {df['Number of Likes'].sum():,}\n")
            f.write(f"Average Likes per Tweet: {df['Number of Likes'].mean():.1f}\n")
    
    print(f"✅ Analysis summary saved to: {file_path}")
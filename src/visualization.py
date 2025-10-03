import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np

class SentimentVisualizer:
    def __init__(self):
        plt.style.use('default')
        self.colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    
    def plot_sentiment_distribution(self, df, title="Sentiment Distribution"):
        """Plot sentiment distribution pie chart"""
        sentiment_counts = df['sentiment'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=[self.colors.get(s, '#3498db') for s in sentiment_counts.index])
        ax1.set_title(f'{title} - Pie Chart')
        
        # Bar chart
        bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=[self.colors.get(s, '#3498db') for s in sentiment_counts.index])
        ax2.set_title(f'{title} - Bar Chart')
        ax2.set_ylabel('Number of Tweets')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_over_time(self, df):
        """Plot sentiment trends over time using Date Created"""
        if 'date' in df.columns:
            try:
                # Group by date and sentiment
                daily_sentiments = df.groupby([df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                
                plt.figure(figsize=(14, 8))
                for sentiment in daily_sentiments.columns:
                    plt.plot(daily_sentiments.index, daily_sentiments[sentiment], 
                            label=sentiment, color=self.colors.get(sentiment, '#3498db'), 
                            linewidth=2.5, marker='o', markersize=4)
                
                plt.title('Sentiment Trends Over Time - FIFA World Cup 2022')
                plt.xlabel('Date')
                plt.ylabel('Number of Tweets')
                plt.legend(title='Sentiment')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                # Also plot moving average for smoother trends
                plt.figure(figsize=(14, 8))
                for sentiment in daily_sentiments.columns:
                    # 3-day moving average
                    moving_avg = daily_sentiments[sentiment].rolling(window=3, min_periods=1).mean()
                    plt.plot(daily_sentiments.index, moving_avg, 
                            label=f'{sentiment} (3-day avg)', color=self.colors.get(sentiment, '#3498db'), 
                            linewidth=2.5, linestyle='--')
                
                plt.title('Sentiment Trends (3-day Moving Average) - FIFA World Cup 2022')
                plt.xlabel('Date')
                plt.ylabel('Number of Tweets')
                plt.legend(title='Sentiment')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Could not plot sentiment over time: {e}")
        else:
            print("No date column available for time series analysis")
    
    def plot_likes_by_sentiment(self, df):
        """Plot number of likes by sentiment"""
        if 'Number of Likes' in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Box plot
            plt.subplot(1, 2, 1)
            sns.boxplot(data=df, x='sentiment', y='Number of Likes', 
                       palette=[self.colors.get(s, '#3498db') for s in df['sentiment'].unique()])
            plt.title('Likes Distribution by Sentiment')
            plt.xticks(rotation=45)
            
            # Bar plot (average likes)
            plt.subplot(1, 2, 2)
            avg_likes = df.groupby('sentiment')['Number of Likes'].mean()
            bars = plt.bar(avg_likes.index, avg_likes.values, 
                          color=[self.colors.get(s, '#3498db') for s in avg_likes.index])
            plt.title('Average Likes by Sentiment')
            plt.ylabel('Average Number of Likes')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print("\nLikes by Sentiment:")
            for sentiment in df['sentiment'].unique():
                likes_data = df[df['sentiment'] == sentiment]['Number of Likes']
                print(f"{sentiment}: Mean={likes_data.mean():.1f}, Median={likes_data.median():.1f}, Max={likes_data.max()}")
        else:
            print("'Number of Likes' column not available")
    
    def plot_source_analysis(self, df):
        """Analyze tweet sources by sentiment"""
        if 'Source of Tweet' in df.columns:
            # Top sources
            top_sources = df['Source of Tweet'].value_counts().head(10)
            
            plt.figure(figsize=(14, 8))
            
            # Top sources overall
            plt.subplot(2, 1, 1)
            bars = plt.barh(range(len(top_sources)), top_sources.values)
            plt.yticks(range(len(top_sources)), top_sources.index)
            plt.title('Top 10 Tweet Sources')
            plt.xlabel('Number of Tweets')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{int(width)}', ha='left', va='center')
            
            # Sentiment by top 5 sources
            plt.subplot(2, 1, 2)
            top_5_sources = top_sources.head(5).index
            source_sentiment = df[df['Source of Tweet'].isin(top_5_sources)].groupby(
                ['Source of Tweet', 'sentiment']).size().unstack(fill_value=0)
            
            source_sentiment.plot(kind='bar', stacked=True, 
                                color=[self.colors.get(s, '#3498db') for s in source_sentiment.columns],
                                ax=plt.gca())
            plt.title('Sentiment Distribution by Top 5 Sources')
            plt.xlabel('Tweet Source')
            plt.ylabel('Number of Tweets')
            plt.legend(title='Sentiment')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
        else:
            print("'Source of Tweet' column not available")
    
    def plot_wordcloud(self, df, sentiment=None, max_words=100):
        """Generate word cloud for specific sentiment"""
        if sentiment:
            text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
            title = f'Word Cloud - {sentiment.capitalize()} Sentiment'
        else:
            text = ' '.join(df['cleaned_text'])
            title = 'Word Cloud - All Tweets'
        
        if text.strip():  # Check if text is not empty
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=max_words,
                                colormap='viridis').generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(title)
            plt.axis('off')
            plt.show()
        else:
            print(f"No text available for {sentiment if sentiment else 'all'} sentiment word cloud")
    
    def plot_polarity_distribution(self, df):
        """Plot distribution of polarity scores"""
        plt.figure(figsize=(12, 6))
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in df['sentiment'].values:
                data = df[df['sentiment'] == sentiment]['polarity']
                plt.hist(data, alpha=0.6, label=sentiment, 
                        color=self.colors.get(sentiment), bins=30)
        
        plt.title('Distribution of Sentiment Polarity Scores')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_team_sentiment_comparison(self, team_sentiments):
        """Compare sentiments across different teams"""
        if not team_sentiments:
            print("No team sentiment data available")
            return
        
        teams = list(team_sentiments.keys())
        sentiments = ['positive', 'neutral', 'negative']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Stacked bar chart
        bottom = np.zeros(len(teams))
        for sentiment in sentiments:
            values = [team_sentiments[team]['sentiment_distribution'].get(sentiment, 0) 
                     for team in teams]
            ax1.bar(teams, values, label=sentiment, bottom=bottom,
                   color=self.colors.get(sentiment))
            bottom += values
        
        ax1.set_title('Team Sentiment Distribution - FIFA World Cup 2022')
        ax1.set_ylabel('Percentage (%)')
        ax1.legend(title='Sentiment')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average polarity comparison
        polarities = [team_sentiments[team]['average_polarity'] for team in teams]
        colors = ['green' if p > 0.05 else 'red' if p < -0.05 else 'orange' for p in polarities]
        
        bars = ax2.bar(teams, polarities, color=colors, alpha=0.7)
        ax2.set_title('Average Sentiment Polarity by Team')
        ax2.set_ylabel('Average Polarity')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.show()

    def create_interactive_plot(self, df):
        """Create interactive plot using Plotly"""
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        
        fig = px.pie(sentiment_counts, values='count', names='sentiment',
                    title='Sentiment Distribution - FIFA World Cup 2022 Tweets',
                    color='sentiment',
                    color_discrete_map=self.colors)
        fig.show()

def save_visualizations(df, team_sentiments=None):
    """Save all visualizations"""
    visualizer = SentimentVisualizer()
    
    # Create and save plots
    visualizer.plot_sentiment_distribution(df)
    visualizer.plot_polarity_distribution(df)
    visualizer.plot_sentiment_over_time(df)
    visualizer.plot_likes_by_sentiment(df)
    visualizer.plot_source_analysis(df)
    visualizer.plot_wordcloud(df, 'positive')
    visualizer.plot_wordcloud(df, 'negative')
    
    if team_sentiments:
        visualizer.plot_team_sentiment_comparison(team_sentiments)

# Wrapper functions for Streamlit app compatibility
def create_sentiment_chart(df):
    """
    Create sentiment distribution chart - wrapper function for Streamlit app
    """
    visualizer = SentimentVisualizer()
    
    # Use Plotly for interactive charts in Streamlit
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        
        fig = px.pie(sentiment_counts, values='count', names='sentiment',
                    title='Sentiment Distribution - FIFA World Cup 2022 Tweets',
                    color='sentiment',
                    color_discrete_map=visualizer.colors)
        return fig
    return None

def create_wordcloud(df, sentiment_type='all'):
    """
    Create word cloud for tweets - wrapper function for Streamlit app
    """
    visualizer = SentimentVisualizer()
    
    try:
        if 'cleaned_text' in df.columns:
            if sentiment_type != 'all':
                text_data = ' '.join(df[df['sentiment'] == sentiment_type]['cleaned_text'].dropna())
            else:
                text_data = ' '.join(df['cleaned_text'].dropna())
            
            if text_data.strip():
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
        print(f"Error creating word cloud: {e}")
    return None

def create_time_series(df):
    """
    Create time series of sentiments - wrapper function for Streamlit app
    """
    visualizer = SentimentVisualizer()
    
    try:
        if 'date' in df.columns and 'sentiment' in df.columns:
            # Use Plotly for interactive time series
            df_time = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_time['date']):
                df_time['date'] = pd.to_datetime(df_time['date'])
            
            df_time['date_only'] = df_time['date'].dt.date
            time_series = df_time.groupby(['date_only', 'sentiment']).size().reset_index()
            time_series.columns = ['date', 'sentiment', 'count']
            
            fig = px.line(
                time_series,
                x='date',
                y='count',
                color='sentiment',
                title='Sentiment Trends Over Time - FIFA World Cup 2022',
                labels={'count': 'Number of Tweets', 'date': 'Date'},
                color_discrete_map=visualizer.colors
            )
            fig.update_layout(xaxis_title='Date', yaxis_title='Number of Tweets')
            return fig
    except Exception as e:
        print(f"Error creating time series: {e}")
    return None
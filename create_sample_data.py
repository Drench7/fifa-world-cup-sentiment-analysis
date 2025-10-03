import pandas as pd
import os

# Create directories
os.makedirs('data/sample', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Create sample data
sample_data = {
    'Tweet': [
        'Amazing goal by Messi! What a fantastic World Cup match! ⚽🎉',
        'Terrible refereeing decisions in the game today 😠',
        'The stadium atmosphere is electric tonight!',
        'Disappointing performance from our team...',
        'GOAL! Beautiful football! World Cup is incredible!',
        'VAR is ruining the spirit of the game',
        'Incredible sportsmanship shown by players 👏',
        'Poor organization of this World Cup',
        'Best tournament ever! Love the World Cup!',
        'Controversial penalty decision',
        'What a stunning free kick! Amazing technique!',
        'The referee should be investigated for this decision',
        'Fantastic crowd support in the stadium tonight',
        'Our team needs to improve their defense',
        'Historic victory for the underdogs!',
        'Another boring 0-0 draw in the tournament',
        'The opening ceremony was spectacular!',
        'Too many fouls ruining the beautiful game',
        'Young talents shining in this World Cup 🌟',
        'Questionable team selection by the coach'
    ],
    'Date Created': pd.date_range('2022-11-20', periods=20, freq='h'),  # Changed 'H' to 'h'
    'User Name': [f'user_{i}' for i in range(20)],
    'Number of Likes': [100, 50, 200, 30, 150, 40, 180, 25, 220, 35, 120, 45, 190, 28, 210, 32, 170, 38, 230, 42],
    'Source of Tweet': ['Twitter for iPhone'] * 10 + ['Twitter for Android'] * 10
}

df_sample = pd.DataFrame(sample_data)
df_sample.to_csv('data/sample/sample_tweets.csv', index=False)
print("✅ Sample data created at data/sample/sample_tweets.csv")

# Also create a minimal version in raw folder for fallback
df_sample.to_csv('data/raw/fifa_world_cup_2022_tweets.csv', index=False)
print("✅ Fallback data created at data/raw/fifa_world_cup_2022_tweets.csv")
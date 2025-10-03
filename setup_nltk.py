import nltk
import os

print('Setting up NLTK for cloud deployment...')

# Create NLTK data directory
nltk_data_dir = './nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)

# Add to NLTK path
nltk.data.path.append(nltk_data_dir)

# Download essential NLTK data
packages = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet']

for package in packages:
    try:
        print(f'Downloading {package}...')
        nltk.download(package, download_dir=nltk_data_dir, quiet=True)
        print(f'✅ {package}')
    except Exception as e:
        print(f'⚠️ Failed to download {package}: {e}')

print('NLTK setup completed!')

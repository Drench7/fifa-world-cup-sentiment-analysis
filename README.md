# ⚽ FIFA World Cup 2022 Tweet Sentiment Analysis

A comprehensive analysis of public sentiment from tweets during the FIFA World Cup 2022, deployed as an interactive Streamlit web application.

## 🚀 Live Demo
[Add your Streamlit Cloud URL here]

## 📁 Project Structure
fifa-world-cup-sentiment-analysis/
│
├── .gitignore
├── README.md
├── requirements.txt
├── setup.sh (optional, for Streamlit deployment)
│
├── data/
│   ├── raw/
│   │   └── fifa_world_cup_2022_tweets.csv
│   └── processed/
│       └── processed_tweets.csv
│
├── notebooks/
│   ├── complete_analysis.ipynb
│   ├── project_presentation.ipynb
│   ├── project_report.ipynb
│   └── sentiment_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── sentiment_analysis.py
│   ├── utils.py
│   └── visualization.py
│
├── app/ (new folder for Streamlit)
│   ├── __init__.py
│   ├── main.py (your Streamlit app)
│   └── components/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── charts.py
│       └── insights.py
│
└── assets/ (optional)
    ├── images/
    └── styles/

    # ⚽ FIFA World Cup 2022 Sentiment Analysis

A Streamlit app that analyzes public sentiment from World Cup tweets using NLTK VADER.

## Features
- Real-time sentiment analysis
- Interactive visualizations
- Word clouds by sentiment
- Time series trends
- Exportable reports

## Live Demo
[Add your Streamlit Cloud URL here after deployment]

## Local Development
```bash
pip install -r requirements.txt
streamlit run app/main.py
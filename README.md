# SWP
# News Search and Analysis Platform

A Flask web application that aggregates news articles from multiple sources and provides advanced search capabilities with text analysis features.

## Features

- News aggregation from NewsAPI and OpenAlex
- Advanced search with multiple similarity algorithms
- Sentiment analysis and language detection
- Article filtering by category, language, and sentiment
- Search quality metrics (precision, recall, F1-score)
- Word cloud and histogram generation

## Installation

### Requirements
- Python 3.8+
- NewsAPI key

### Setup

1. Clone the repository
```bash
git clone <repository-url>
cd SWP
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

4. Configure API key in `app.py`
```python
API_KEY = 'your_newsapi_key_here'
```

5. Run the application
```bash
python app.py
```

6. Open browser to `http://localhost:5000`

## Usage

### Main Page
Browse articles with filtering options:
- Category selection (technology, sports, health, etc.)
- Language filtering
- Sentiment filtering
- Language distribution visualization

### Search Page
Advanced search functionality:
- Text search with spell correction
- Multiple similarity scoring algorithms
- Search quality metrics
- Word cloud and frequency analysis

## Technical Details

### Search Algorithms
- TF-IDF vectorization
- Cosine similarity
- Levenshtein distance
- Jaccard similarity with n-grams

### Text Processing
- Automatic spell correction
- Text lemmatization
- Sentiment analysis using VADER
- Language detection

### Database
SQLite database storing:
- Article title and description
- URL and publication date
- Processed article data

## Dependencies

Core libraries:
- Flask 2.2.5
- Flask-SQLAlchemy 3.0.3
- scikit-learn 1.2.2
- nltk 3.8.1
- requests 2.31.0

See `requirements.txt` for complete list.

## API Endpoints

- `/` - Main dashboard (GET/POST)
- `/forms` - Search interface (GET)
- `/search` - Execute search (POST)

## License

MIT License

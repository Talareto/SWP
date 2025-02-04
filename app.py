from flask import Flask, render_template, request
import requests
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import Levenshtein
from nltk.util import ngrams
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import process
from langdetect import detect, DetectorFactory




nltk.download('wordnet')
nltk.download('vader_lexicon')
app = Flask(__name__)
API_KEY = '3434439a803240cdae66cf6ba24812f9'




app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///articles.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)





class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    url = db.Column(db.String(300), nullable=False, unique=True)
    published_at = db.Column(db.String(100), nullable=False)

with app.app_context():
    db.create_all()



def get_articles(category):
    url = f'https://newsapi.org/v2/everything?q={category}&sortBy=publishedAt&apiKey={API_KEY}'
    response = requests.get(url)
    return response.json().get('articles', []) if response.status_code == 200 else []



def get_openalex_articles(query):
    url = f'https://api.openalex.org/works?filter=title.search:{query}&per-page=10' 
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get('results', [])
        articles = []
        for item in data:
            articles.append({
                'title': item['title'],
                'description': item.get('abstract', 'No abstract available'),
                'url': f"https://doi.org/{item['doi']}" if 'doi' in item else item['id'],
                'publishedAt': item.get('publication_date', 'N/A')
            })
        return articles
    return []



def save_articles(articles):
    for article in articles:
        if not article.get('title'):
            continue
        existing_article = Article.query.filter_by(url=article['url']).first()
        if not existing_article:
            new_article = Article(
                title=article['title'],
                description=article.get('description', ''),
                url=article['url'],
                published_at=article.get('publishedAt', 'N/A'),
            )
            db.session.add(new_article)
    db.session.commit()



def correct_spelling(query, article_titles):
    words = query.split()
    corrected_words = []
    for word in words:
        # Znajdź najlepsze dopasowanie z artykułami
        best_match = process.extractOne(word, article_titles)
        if best_match and best_match[1] >= 70:  # Możesz dostosować próg
            corrected_words.append(best_match[0])
        else:
            corrected_words.append(word)  # Jeśli nie znaleziono dobrego dopasowania
    return " ".join(corrected_words)



def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])



def levenshtein_similarity(str1, str2):
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1 - (Levenshtein.distance(str1, str2) / max_len)



def jaccard_similarity(str1, str2, k=3):
    set1 = set(ngrams(str1, k))
    set2 = set(ngrams(str2, k))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)



def analyze_sentiment(text):  
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']

    # Klasyfikacja
    if sentiment_score > 0.2:
        return "Pozytywny"
    elif sentiment_score < -0.2:
        return "Negatywny"
    else:
        return "Neutralny"
    

    DetectorFactory.seed = 0 
def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return "Nie rozpoznano"


@app.route('/', methods=['GET', 'POST'])
def index():
    category = request.form.get('category', 'technology')
    newsapi_articles = get_articles(category)
    save_articles(newsapi_articles)
    return render_template('index.html', articles=newsapi_articles, selected_category=category)

@app.route('/forms', methods=['GET'])
def show_form():
    return render_template('forms.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form.get('searchWord', '').strip()
    if not search_query:
        return render_template('forms.html', articles=[], detected_language="Nie wprowadzono tekstu")

    articles = Article.query.all()
    article_titles = [article.title.lower() for article in articles]

    # Poprawa literówek i lematyzacja
    search_query = correct_spelling(search_query.lower(), article_titles)
    search_query = lemmatize_text(search_query)

    documents = [lemmatize_text(article.title + " " + (article.description or "")) for article in articles]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(documents)
    query_vec = tfidf_vectorizer.transform([search_query])

    similarity_scores = []
    for idx, article in enumerate(articles):
        lev_score = levenshtein_similarity(search_query, documents[idx])
        jaccard_score = jaccard_similarity(search_query, documents[idx])
        tfidf_score = query_vec.dot(X[idx].T).toarray()[0][0]
        final_score = (0.5 * tfidf_score) + (0.3 * lev_score) + (0.2 * jaccard_score)
        similarity_scores.append((idx, final_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_articles = []

    for idx, _ in similarity_scores[:5]:
        article = articles[idx]
        # Rozpoznawanie języka dla title + description
        full_text = article.title + " " + (article.description or "")
        article_language = detect_language(full_text)

        top_articles.append({
            'title': article.title,
            'description': article.description or 'No description available',
            'url': article.url,
            'publishedAt': article.published_at,
            'sentiment': analyze_sentiment(full_text),
            'language': article_language  # Dodano język artykułu
        })

    return render_template('forms.html', articles=top_articles)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import requests
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import Levenshtein
from nltk.util import ngrams
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import process
from langdetect import detect, DetectorFactory
from wordcloud import WordCloud
import io
import base64
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import re


nltk.download('stopwords')
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


sia = SentimentIntensityAnalyzer()
def analyze_sentiment(text):  
    
    sentiment_score = sia.polarity_scores(text)['compound']

    # Klasyfikacja
    if sentiment_score > 0.2:
        return "pozytywny"  
    elif sentiment_score < -0.2:
        return "negatywny"
    else:
        return "neutralny"
    

    DetectorFactory.seed = 0 
def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return "Nie rozpoznano"




def generate_wordcloud(text):
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white", 
        colormap="viridis"
    ).generate(text)

    # Zapisujemy chmurę słów do pamięci jako obrazek
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()



def generate_word_histogram(text):
    words = re.findall(r'\b\w+\b', text.lower())  # Pobieramy słowa (ignorujemy interpunkcję)
    stopwords = set(nltk.corpus.stopwords.words('english'))  # Możesz dodać inne języki
    filtered_words = [word for word in words if word not in stopwords]  # Usuwamy stopwords

    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(10)  # Pobieramy 10 najczęściej występujących słów

    words, counts = zip(*common_words) if common_words else ([], [])
    
    # Tworzenie wykresu
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color="blue")
    plt.xlabel("Słowa")
    plt.ylabel("Liczba wystąpień")
    plt.xticks(rotation=45)
    plt.title("Najczęściej występujące słowa")
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()


def generate_language_bar_chart(articles):
    
    language_counts = {}

    for article in articles:
        lang = article.get('language', 'unknown')
        language_counts[lang] = language_counts.get(lang, 0) + 1

    labels = list(language_counts.keys())
    values = list(language_counts.values())

    # Tworzenie wykresu słupkowego
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel("Język artykułów")
    plt.ylabel("Liczba artykułów")
    plt.title("Podział artykułów według języka")
    plt.xticks(rotation=45)

    # Konwersja na obrazek w base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()



def cosine_similarity_score(vec1, vec2):
    return cosine_similarity(vec1, vec2)[0][0]




@app.route('/', methods=['GET', 'POST'])
def index():
    category = request.form.get('category', 'technology')
    selected_language = request.form.get('language', 'all')
    selected_sentiment = request.form.get('sentiment', 'all').lower().strip()

    # Lista dostępnych kategorii
    categories = ['technology', 'tesla', 'wallstreet', 'sport', 'health', 'entertainment']

    # Pobieranie artykułów – jeśli "Wszystkie kategorie", pobieramy z każdej
    if category == 'all':
        newsapi_articles = []
        for cat in categories:
            newsapi_articles.extend(get_articles(cat))
    else:
        newsapi_articles = get_articles(category)

    processed_newsapi_articles = [
        {
            'title': article.get('title', 'No title'),
            'description': article.get('description', 'No description available'),
            'url': article.get('url', '#'),
            'publishedAt': article.get('publishedAt', 'N/A'),
            'language': detect_language((article.get('title') or '') + " " + (article.get('description') or '')),
            'sentiment': analyze_sentiment((article.get('title') or '') + " " + (article.get('description') or ''))
        }
        for article in newsapi_articles
    ]

    # save_articles(newsapi_articles)

    # Pobieranie artykułów z bazy – jeśli "Wszystkie kategorie", pobieramy wszystkie
    if category == 'all':
        all_articles = Article.query.all()
    else:
        all_articles = Article.query.filter(Article.title.ilike(f"%{category}%")).all()

    processed_db_articles = [
        {
            'title': article.title,
            'description': article.description or 'No description available',
            'url': article.url,
            'publishedAt': article.published_at,
            'language': detect_language(article.title + " " + (article.description or "")),
            'sentiment': analyze_sentiment(article.title + " " + (article.description or ""))
        }
        for article in all_articles
    ]

    combined_articles = processed_newsapi_articles + processed_db_articles

    # Konwersja daty publikacji na obiekt datetime
    for article in combined_articles:
        try:
            article["publishedAt"] = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            article["publishedAt"] = None  # Jeśli format jest błędny, przypisz None

    # Filtrowanie po języku
    if selected_language and selected_language != "all":
        combined_articles = [article for article in combined_articles if article['language'] == selected_language]

    # Filtrowanie po sentymencie
    if selected_sentiment and selected_sentiment != "all":
        print(f"Filtrujemy po sentymencie: {selected_sentiment}")  # Testowanie
        combined_articles = [article for article in combined_articles if article['sentiment'].lower() == selected_sentiment]

    # Sortowanie według daty publikacji (najnowsze pierwsze)
    combined_articles.sort(key=lambda x: x["publishedAt"] if x["publishedAt"] else datetime.min, reverse=True)

    # Generowanie histogramu języków artykułów
    language_bar_chart = generate_language_bar_chart(combined_articles)

    return render_template('index.html', articles=combined_articles, selected_category=category, language_bar_chart=language_bar_chart)




@app.route('/forms', methods=['GET'])
def show_form():
    return render_template('forms.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form.get('searchWord', '').strip()
    if not search_query:
        return render_template('forms.html', articles=[], wordcloud_image=None, query_metrics=None)

    articles = Article.query.all()
    article_titles = [article.title.lower() for article in articles]

    #  Poprawa literówek i lematyzacja
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
        cos_score = cosine_similarity_score(query_vec, X[idx])

        # Nowy sposób liczenia finalnego wyniku
        final_score = (0.3 * tfidf_score) + (0.25 * lev_score) + (0.2 * jaccard_score) + (0.25 * cos_score)
        similarity_scores.append((idx, final_score, lev_score, jaccard_score, tfidf_score, cos_score))

    # Sortowanie wyników według końcowego wyniku
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_articles = []

    #  Przygotowanie tekstu do chmury słów
    wordcloud_text = ""

    # Znajdowanie artykułów, które częściowo pasują do zapytania
    relevant_articles = [
        article for article in articles
        if jaccard_similarity(search_query, article.title.lower()) > 0.2
        or jaccard_similarity(search_query, (article.description or "").lower()) > 0.2
    ]

    # Jeśli nie znaleziono żadnych idealnych dopasowań, spróbujmy z TF-IDF
    if not relevant_articles:
        relevant_articles = [
            article for article in articles 
            if tfidf_vectorizer.transform([article.title]).dot(query_vec.T).toarray()[0][0] > 0.1
        ]

    relevant_count = len(relevant_articles)  # Liczba rzeczywiście trafnych artykułów
    retrieved_count = len(similarity_scores[:15])  # Liczba zwróconych wyników

    retrieved_relevant_count = sum(
        1 for idx, _, _, _, _, _ in similarity_scores[:15] 
        if any(articles[idx].title.lower() == a.title.lower() for a in relevant_articles)
    )

    
    precision = retrieved_relevant_count / retrieved_count if retrieved_count > 0 else 0
    recall = retrieved_relevant_count / relevant_count if relevant_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    #  Dodano printy do debugowania
    print(f" search_query: {search_query}")
    print(f"etrieved_count: {retrieved_count}")
    print(f"retrieved_relevant_count: {retrieved_relevant_count}")
    print(f"relevant_count: {relevant_count}")
    print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")

    #  Zapisujemy wyniki do tabeli
    query_metrics = {
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-score": round(f1, 3)
    }

    #  Tworzenie listy artykułów do wyświetlenia
    for idx, final_score, lev_score, jaccard_score, tfidf_score, cos_score in similarity_scores[:15]:
        article = articles[idx]
        wordcloud_text += " " + article.title + " " + (article.description or "")

        top_articles.append({
            'title': article.title,
            'description': article.description or 'No description available',
            'url': article.url,
            'publishedAt': article.published_at,
            'sentiment': analyze_sentiment(article.title + " " + (article.description or "")),
            'levenshtein': round(lev_score, 3),
            'jaccard': round(jaccard_score, 3),
            'tfidf': round(tfidf_score, 3),
            'cosine': round(cos_score, 3),
            'final_score': round(final_score, 3),
            'language': detect_language(article.title + " " + (article.description or ""))  
        })

    #  Generujemy chmurę słów
    wordcloud_image = generate_wordcloud(wordcloud_text)

    
    histogram_text = " ".join([article['title'] + " " + (article['description'] or "") for article in top_articles])

    #  Generujemy histogram
    word_histogram_image = generate_word_histogram(histogram_text)

    return render_template('forms.html', 
                        articles=top_articles, 
                        wordcloud_image=wordcloud_image, 
                        word_histogram_image=word_histogram_image, 
                        query_metrics=query_metrics)


if __name__ == '__main__':
    app.run(debug=True)

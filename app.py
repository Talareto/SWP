from flask import Flask, render_template, request
import requests
from flask_sqlalchemy import SQLAlchemy
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import math
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 


app = Flask(__name__)


API_KEY = '3434439a803240cdae66cf6ba24812f9'

# Konfiguracja bazy danych SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///articles.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model artykułu
class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    url = db.Column(db.String(300), nullable=False, unique=True)
    published_at = db.Column(db.String(100), nullable=False)


# Tworzenie bazy danych
with app.app_context():
    db.create_all()

# Funkcja do pobierania artykułów z NewsAPI
def get_articles(category):
    if category == 'technology':
        url = f'https://newsapi.org/v2/everything?q=technology&from=2024-12-27&sortBy=publishedAt&apiKey={API_KEY}'
    elif category == 'tesla':
        url = f'https://newsapi.org/v2/everything?q=tesla&from=2024-12-27&sortBy=publishedAt&apiKey={API_KEY}'
    elif category == 'wallstreet':
        url = f'https://newsapi.org/v2/everything?domains=wsj.com&apiKey={API_KEY}'
    elif category == 'sport':
         url = f'https://newsapi.org/v2/everything?q=sport&from=2024-12-27&sortBy=publishedAt&apiKey={API_KEY}'
    elif category == 'health':
         url = f'https://newsapi.org/v2/everything?q=health&from=2024-12-27&sortBy=publishedAt&apiKey={API_KEY}'
    elif category == 'science':
         url = f'https://newsapi.org/v2/everything?q=science&from=2024-12-27&sortBy=publishedAt&apiKey={API_KEY}'  
    elif category == 'entertainment':
         url = f'https://newsapi.org/v2/everything?q=entertainment&from=2024-12-27&sortBy=publishedAt&apiKey={API_KEY}'   
    else:
        return []

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('articles', [])
    else:
        return []

# Funkcja do pobierania artykułów z OpenAlex API
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

# Funkcja do zapisywania artykułów w bazie danych
def save_articles(articles):
    for article in articles:
        if not article.get('title'):
            print(f"Pomijam artykuł bez tytułu: {article}")
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





# Funkcja do wczytywania lematów z pliku
def wczytaj_lemy(nazwa_pliku):
    mapa_lemmatyzacji = {}
    with open(nazwa_pliku, 'r') as plik:
        for linia in plik:
            oryginalny, lema = linia.strip().split()
            mapa_lemmatyzacji[oryginalny] = lema
    return mapa_lemmatyzacji

# Funkcja do lematyzacji tokenów z użyciem pliku lematyzacji
def lematyzuj_tokeny(tokeny, lemy):
    return [lemy.get(token, token) for token in tokeny]

# Funkcja obliczająca TF-IDF
def podziel_na_tokeny(tekst):
    return re.findall(r'\b\w+\b', tekst.lower())

def oblicz_czestotliwosc_tokenow(tokeny):
    licznik_tokenow = Counter(tokeny)
    suma_tokenow = len(tokeny)
    return {slowo: count / suma_tokenow for slowo, count in licznik_tokenow.items()}

def oblicz_idf(wszystkie_dokumenty):
    liczba_dok = len(wszystkie_dokumenty)
    wystapienia_termow = defaultdict(lambda: 0)
    for doc in wszystkie_dokumenty:
        unikalne_term = set(doc)
        for term in unikalne_term:
            wystapienia_termow[term] += 1
    return {term: math.log(liczba_dok / count) for term, count in wystapienia_termow.items()}

def oblicz_tfidf(tf, idf):
    return {slowo: tf_wartosc * idf.get(slowo, 0) for slowo, tf_wartosc in tf.items()}






def train_naive_bayes_classifier(articles):
    # Przygotowanie dokumentów (tytuł + opis) oraz kategorii
    documents = [article.title + " " + (article.description or "") for article in articles]
    
    # Przypisanie kategorii ręcznie lub na podstawie jakiejś logiki
    categories = ['technology' for _ in articles]  # Przykład: wszystkie artykuły mają kategorię 'technology'
    
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(documents)
    
    X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.2, random_state=42)
    
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    
    y_pred = nb_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return nb_classifier, tfidf_vectorizer, accuracy, report






@app.route('/index', methods=['GET', 'POST'])
def index():
    category = request.form.get('category', 'technology')
    newsapi_articles = get_articles(category)
    openalex_articles = get_openalex_articles(category)

    # Połącz wszystkie artykuły
    all_articles = newsapi_articles + openalex_articles

    # Zapisz w bazie danych
    save_articles(all_articles)
    return render_template('index.html', articles=all_articles, selected_category=category)

@app.route('/forms', methods=['GET', 'POST'])
def show_data():
    articles = Article.query.all()
    return render_template('forms.html', articles=articles)

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['searchWord']

    
    articles = get_articles('technology')  # Możesz zmienić kategorię

    
    article_tokens = [podziel_na_tokeny((article['title'] or "") + " " + (article['description'] or "")) for article in articles]
    query_tokens = podziel_na_tokeny(search_query)
    lemy = wczytaj_lemy('diffs.txt')  # Załaduj lematy z pliku

    # Lematyzowanie tokenów artykułów i zapytania
    lematyzowane_artykuly = [lematyzuj_tokeny(tokens, lemy) for tokens in article_tokens]
    lematyzowane_zapytanie = lematyzuj_tokeny(query_tokens, lemy)

    # Obliczanie częstotliwości tokenów dla artykułów i zapytania
    tf_articles = [oblicz_czestotliwosc_tokenow(tokens) for tokens in lematyzowane_artykuly]
    tf_query = oblicz_czestotliwosc_tokenow(lematyzowane_zapytanie)

    
    idf = oblicz_idf(lematyzowane_artykuly)
    tfidf_articles = [oblicz_tfidf(tf, idf) for tf in tf_articles]
    tfidf_query = oblicz_tfidf(tf_query, idf)

    # Dopasowanie artykułów do zapytania za pomocą TF-IDF
    similarity_scores = []
    for i, tfidf_article in enumerate(tfidf_articles):
        # Obliczanie podobieństwa między artykułem a zapytaniem (iloczyn skalarny)
        similarity_score = sum([tfidf_query.get(word, 0) * tfidf_article.get(word, 0) for word in tfidf_query])
        similarity_scores.append((i, similarity_score))

    # Sortowanie artykułów według podobieństwa (od najwyższego)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Pobierz 5 najbardziej podobnych artykułów
    top_articles = [articles[i] for i, _ in similarity_scores[:5]]

    return render_template('search_results.html', articles=top_articles)




@app.route('/searchPlus', methods=['GET', 'POST'])
def search_plus():
    if request.method == 'POST':
        search_query = request.form['searchWord']  # Słowo kluczowe wprowadzone przez użytkownika
        
        # Pobieranie artykułów z bazy danych
        articles = Article.query.all()
        
        # Przygotowanie dokumentów (tytuł + opis)
        documents = [article.title + " " + (article.description or "") for article in articles]
        titles = [article.title for article in articles]
        urls = [article.url for article in articles]
        
        # Przekształcanie dokumentów na wektory TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        X = tfidf_vectorizer.fit_transform(documents)

        # Przekształcanie zapytania użytkownika na wektor TF-IDF
        query_vec = tfidf_vectorizer.transform([search_query])

        # Używamy Nearest Neighbors, aby znaleźć najbardziej podobne artykuły
        nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        nn.fit(X)  # Dopasowanie modelu na wszystkich artykułach

        distances, indices = nn.kneighbors(query_vec) 

        # Zbieramy artykuły, które są najbardziej podobne do zapytania
        similar_articles = []
        for idx in indices[0]:
            similar_articles.append({
                'title': titles[idx],
                'url': urls[idx],
                'description': documents[idx]
            })

        # Zakładając, że przewidujesz kategorię artykułu (np. jako zmienną)
        predicted_category = "Science"  # Przykład, można to dynamicznie ustawić w zależności od klasyfikatora
        accuracy = 0.85  # Przykład, można ustawić dokładność modelu
        classification_report = "Report..."  # Zawartość raportu

        return render_template('searchPlus.html', 
                               articles=similar_articles,
                               predicted_category=predicted_category,
                               accuracy=accuracy,
                               classification_report=classification_report)

    return render_template('searchPlus.html', articles=[])



@app.route('/bayes_classifier', methods=['GET', 'POST'])
def bayes_classifier():
    if request.method == 'POST':
        search_query = request.form['searchWord']  # Słowo kluczowe wpisane przez użytkownika
        
        # Pobieranie artykułów z bazy danych
        articles = Article.query.all()

        # Sprawdzenie, czy model Bayesa jest wytrenowany, jeśli nie, trenujemy
        if not hasattr(bayes_classifier, 'nb_classifier'):
            bayes_classifier.nb_classifier, bayes_classifier.tfidf_vectorizer, bayes_classifier.accuracy, bayes_classifier.classification_report = train_naive_bayes_classifier(articles)
        
        # Przekształcanie dokumentów na wektory TF-IDF
        tfidf_vectorizer = bayes_classifier.tfidf_vectorizer
        X = tfidf_vectorizer.transform([article.title + " " + (article.description or "") for article in articles])

        # Przekształcanie zapytania użytkownika na wektor TF-IDF
        query_vec = tfidf_vectorizer.transform([search_query])

        # Używamy Nearest Neighbors, aby znaleźć najbardziej podobne artykuły
        nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        nn.fit(X)  # Dopasowanie modelu na wszystkich artykułach

        distances, indices = nn.kneighbors(query_vec)  # Szukamy najbliższych sąsiadów

        # Przewidywanie kategorii dla każdego znalezionego artykułu
        similar_articles = []
        for idx in indices[0]:
            article_text = articles[idx].title + " " + (articles[idx].description or "")
            article_vec = tfidf_vectorizer.transform([article_text])
            predicted_category = bayes_classifier.nb_classifier.predict(article_vec)[0]

            similar_articles.append({
                'title': articles[idx].title,
                'url': articles[idx].url,
                'description': articles[idx].description,
                'category': predicted_category  # Dodajemy kategorię
            })
        
        return render_template('bayes_classifier.html', 
                               articles=similar_articles,
                               accuracy=bayes_classifier.accuracy,
                               classification_report=bayes_classifier.classification_report)

    return render_template('bayes_classifier.html', articles=[])



if __name__ == '__main__':
    app.run(debug=True)
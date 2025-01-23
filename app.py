from flask import Flask, render_template, request
import requests
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Twój klucz API do NewsAPI
API_KEY = '3434439a803240cdae66cf6ba24812f9'

# Funkcja do pobierania artykułów z NewsAPI
def get_articles(category):
    if category == 'technology':
        url = f'https://newsapi.org/v2/everything?q=technology&from=2024-12-25&sortBy=publishedAt&apiKey={API_KEY}'
    elif category == 'tesla':
        url = f'https://newsapi.org/v2/everything?q=tesla&from=2024-12-25&sortBy=publishedAt&apiKey={API_KEY}'
    elif category == 'wallstreet':
        url = f'https://newsapi.org/v2/everything?domains=wsj.com&apiKey={API_KEY}'
    else:
        return []

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('articles', [])
    else:
        return []

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

# Funkcja do zapisywania artykułów w bazie danych
def save_articles(articles):
    for article in articles:
        # Sprawdź, czy artykuł już istnieje w bazie danych
        existing_article = Article.query.filter_by(url=article['url']).first()
        if not existing_article:
            new_article = Article(
                title=article['title'],
                description=article.get('description'),
                url=article['url'],
                published_at=article['publishedAt']
            )
            db.session.add(new_article)
    db.session.commit()

@app.route('/', methods=['GET', 'POST'])
def index():
    category = request.form.get('category', 'technology')  # Domyślnie "technology"
    articles = get_articles(category)
    save_articles(articles)  # Zapisz artykuły w bazie danych
    return render_template('index.html', articles=articles, selected_category=category)

@app.route('/forms', methods=['GET', 'POST'])
def showData():
    articles = Article.query.all()  # Pobierz wszystkie artykuły z bazy danych
    return render_template('forms.html', articles=articles)

@app.route('/search', methods=['GET', 'POST'])
def show_search():
    search_query = request.form.get('searchWord', '')  # Pobierz zapytanie wyszukiwania
    if search_query:
        articles = Article.query.filter(Article.title.contains(search_query)).all()
    else:
        articles = []
    return render_template('forms.html', articles=articles, search_query=search_query)

if __name__ == '__main__':
    app.run(debug=True)
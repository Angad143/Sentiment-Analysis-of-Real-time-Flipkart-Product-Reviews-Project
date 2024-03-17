from flask import Flask, request, render_template
from joblib import load
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained random forest model and TF-IDF vectorizer
random_forest_model = load('Data/Random_forest_model.pkl')
tfidf_vectorizer = load('Data/tfidf_vectorizer.pkl')

# Preprocessing functions
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r'\W+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_words)

# Predict sentiment function
def predict_sentiment(input_text):
    preprocess_text = clean_text(input_text)
    preprocessed_text = lemmatize_text(preprocess_text)
    features = tfidf_vectorizer.transform([preprocessed_text])
    prediction = random_forest_model.predict(features)[0]
    return "Positive" if prediction == 1 else "Negative"

# Define a route to handle both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        if not text.strip():
            return render_template('index.html', sentiment='', emoji='')
        sentiment = predict_sentiment(text)
        emoji = '😊' if sentiment == 'Positive' else '😔'
        return render_template('index.html', sentiment=sentiment, emoji=emoji)
    return render_template('index.html', sentiment='', emoji='')

if __name__ == '__main__':
    app.run(debug=True)

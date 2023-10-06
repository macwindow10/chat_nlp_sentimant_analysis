from flask import *
from werkzeug.utils import secure_filename
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
from spacy import displacy
from spacy import displacy
from gensim import models

matplotlib.use('Agg')

nltk.downloader.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# load w2v from pre-built Google data
w2v = models.word2vec.Word2Vec()
# download bin.gz from: https://code.google.com/archive/p/word2vec/
w2v = models.KeyedVectors.load_word2vec_format(
    "D:\\pythonprojects\\GoogleNews-vectors-negative300\\GoogleNews-vectors-negative300.bin",
    binary=True)
w2v_vocab = set(w2v.index_to_key)
print("Loaded {} words in vocabulary".format(len(w2v_vocab)))

analyzer = SentimentIntensityAnalyzer()


def remove_stop_words(word_tokens, stop_words):
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', message="")


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        f.save('uploads/' + secure_filename(f.filename))
        return render_template('index.html', message="File uploaded successfully")
    return render_template('index.html', message="Error occurred")


@app.route("/process")
def process():
    student_id = request.form.get("student_id")
    return jsonify(status="success")


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()

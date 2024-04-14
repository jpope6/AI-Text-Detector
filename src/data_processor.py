import string
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# First time you run you need to download uncomment this.
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download("stopwords")

STOPWORDS = stopwords.words('english')

class DataProcessor:
    def __init__(self, data) -> None:
        self.data = data

        self.process_corpus = np.vectorize(self.process_text)
        self.processed_corpus = self.process_corpus(self.data)

        # Tf-Idf Vectorizer
        self.vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, use_idf = True)
        self.matrix = self.vectorizer.fit_transform(self.processed_corpus)

    def process_text(self, text):
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)

        # Lowercasing
        tokens = [token.lower() for token in tokens]
        
        # Join the tokens back into a string
        processed_text = " ".join(tokens)

        return processed_text

    def get_input_matrix(self, input_text):
        processed_input = self.process_text(input_text)

        print("\n", processed_input, "\n")
        matrix_input = self.vectorizer.transform([processed_input])
        return matrix_input

    # Might get rid of this if we use punctuation as
    # a feature in our model
    def remove_punctuation(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub('[''""…]', '', text)
        text = re.sub('\n', '', text)

        return text

    # Split sentence into array of words
    # Useful for passing into our machine learning model
    def tokenize(self, text):
        tokens = text.split()
        return self.remove_stopwords(tokens)

    # Stopwords are the words which are most common
    # like I, am, there, where etc. They usually don’t
    # help in certain NLP tasks and are best removed to
    # save computation and time.
    def remove_stopwords(self, tokens):
            return [token for token in tokens if token not in STOPWORDS]

import string
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix

# First time you run you need to download uncomment this.
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("stopwords")

STOPWORDS = stopwords.words("english")


class DataProcessor:
    def __init__(self, data, n_components=100):
        """
        Initializes the DataProcessor with the given dataset and the number of components for SVD.

        Args:
        data (DataFrame): Input data containing text and additional features.
        n_components (int): Number of components to keep during dimensionality reduction.
        """
        # Extracts the text data from the dataframe.
        self.data = data["text"].values

        # Vectorizes the function that processes text for parallel processing.
        self.process_corpus = np.vectorize(self.process_text)
        # Applies text processing to all text data.
        self.processed_corpus = self.process_corpus(self.data)

        # Initializes and fits the Tf-Idf Vectorizer to transform the text data into a TF-IDF matrix.
        self.vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, use_idf=True)
        self.matrix = self.vectorizer.fit_transform(self.processed_corpus)

        # Extracts specified features from the data, assumed to be pre-calculated.
        other_features = data[
            [
                "char_count",
                "word_count",
                "capital_char_count",
                "capital_word_count",
                "punctuation_count",
                "quoted_word_count",
                "sent_count",
                "unique_word_count",
                "stopword_count",
                "avg_word_length",
                "avg_sent_length",
                "unique_vs_words",
                "stopwords_vs_words",
            ]
        ]

        # Converts other features into a sparse matrix format to ensure compatibility.
        other_features_sparse = csr_matrix(other_features.values)

        # Stacks the TF-IDF matrix and other feature matrices horizontally (column-wise).
        combined_features = hstack((self.matrix, other_features_sparse))

        # Initializes and applies Truncated SVD to reduce dimensionality of the combined feature set.
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.combined_features_reduced = self.svd.fit_transform(combined_features)

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
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub("[" '""…]', "", text)
        text = re.sub("\n", "", text)

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

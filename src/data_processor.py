import nltk
import string
import re
from nltk.corpus import stopwords

# First time you run you need to download uncomment this.
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download("stopwords")

STOPWORDS = stopwords.words('english')

class DataProcessor:
    def __init__(self, data) -> None:
        self.data = data
        self.tokenized_data = []

    def process_data(self):
        for text in self.data:
            text = text.lower()
            text = self.remove_punctuation(text)
            self.tokenized_data.append(self.tokenize(text))

        self.remove_stopwords()

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
        return text.split()

    # Stopwords are the words which are most common
    # like I, am, there, where etc. They usually don’t
    # help in certain NLP tasks and are best removed to
    # save computation and time.
    def remove_stopwords(self):
        return [token for token in self.tokenized_data if token not in STOPWORDS]

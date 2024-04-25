import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def count_chars(text):
    return len(text)

def count_words(text):
    return len(text.split())

def count_capital_chars(text):
    return sum(1 for char in text if char.isupper())

def count_capital_words(text):
    return sum(1 for word in text.split() if word and word[0].isupper())

def count_punctuations(text):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return sum(text.count(char) for char in punctuations)

def count_words_in_quotes(text):
    matches = re.findall(r"\"[^\"]*\"|'[^']*'", text)
    return sum(count_words(match[1:-1]) for match in matches)

def count_sent(text):
    return len(nltk.sent_tokenize(text))

def count_unique_words(text):
    return len(set(text.split()))

def count_htags(text):
    return len(re.findall(r'#[\w]+', text))

def count_mentions(text):
    return len(re.findall(r'@[\w]+', text))

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    return sum(1 for w in word_tokens if w.lower() in stop_words)

def add_features(data):
    data['char_count'] = data['text'].apply(count_chars)
    data['word_count'] = data['text'].apply(count_words)
    data['capital_char_count'] = data['text'].apply(count_capital_chars)
    data['capital_word_count'] = data['text'].apply(count_capital_words)
    data['punctuation_count'] = data['text'].apply(count_punctuations)
    data['quoted_word_count'] = data['text'].apply(count_words_in_quotes)
    data['sent_count'] = data['text'].apply(count_sent)
    data['unique_word_count'] = data['text'].apply(count_unique_words)
    data['stopword_count'] = data['text'].apply(count_stopwords)
    data['avg_word_length'] = data['char_count'] / data['word_count']
    data['avg_sent_length'] = data['word_count'] / data['sent_count']
    data['unique_vs_words'] = data['unique_word_count'] / data['word_count']
    data['stopwords_vs_words'] = data['stopword_count'] / data['word_count']
    return data

# Setup environment and file paths
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'data/Training_Essay_Data.csv')

# Read the CSV file
data = pd.read_csv(data_path)

print("Setting up Model...")
data_with_features = add_features(data)

# Save to CSV
output_path = os.path.join(current_dir, 'data/Updated_Training_Essay_Data.csv')
data_with_features.to_csv(output_path, index=False)
print(f"Data saved to '{output_path}'.")

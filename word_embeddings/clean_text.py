import re

def clean_and_tokenize(text):
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.lower()
    tokens = cleaned_text.split(' ')
    with open("./stopwords-en.txt", "r") as f:
        stop_words = f.read()
    stop_words = stop_words.replace('\n', ' ').split(' ')
    return [token for token in tokens if token not in stop_words][:-1]
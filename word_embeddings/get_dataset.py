import pandas as pd
from clean_text import clean_and_tokenize
from target_context_pairs import target_context_tuples
from extract_tokens import extract_tokens


def get_dataset(tokens, window_size):

    unique_words = set(tokens)
    word_id = {word:i for (i, word) in enumerate(unique_words)}
    id_word = {i:word for (i, word) in enumerate(unique_words)}

    target_context_pairs = target_context_tuples(tokens, window_size)

    df = pd.DataFrame(target_context_pairs, columns=['target', 'context'])
    
    return df, word_id, id_word
    
def get_unique_words(tokens):
    return set(tokens)
import pandas as pd
from clean_text import clean_and_tokenize
from target_context_pairs import target_context_tuples
from extract_tokens import extract_tokens
from get_dataset import get_dataset, get_unique_words
from w2v_dataset import W2V_Dataset
from train import train

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader


window_size = 2
embed_size = 100
lr = 1e-2
epochs = 300


with open("./text.txt", "r") as text_file:
    text_file = text_file.read()

tokens = extract_tokens(text_file)

df, word_id, id_word = get_dataset(tokens, window_size)

unique_words = get_unique_words(tokens)
vocab_size = len(unique_words)

torch_indices = [word_id[token] for token in unique_words]
encodings = F.one_hot(torch.tensor(torch_indices), num_classes = vocab_size).float()

df["target_one_hot"] = df["target"].apply(lambda x : encodings[word_id[x]])
df["context_one_hot"] = df["context"].apply(lambda x : encodings[word_id[x]])

dataset = W2V_Dataset(df)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = train(vocab_size, embed_size, lr, epochs, dataloader)
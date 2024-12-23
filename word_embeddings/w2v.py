import re
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
    data = text_file.read()

def clean_and_tokenize(text):
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.lower()
    tokens = cleaned_text.split(' ')
    with open("./stopwords-en.txt", "r") as f:
        stop_words = f.read()
    stop_words = stop_words.replace('\n', ' ').split(' ')
    return [token for token in tokens if token not in stop_words][:-1]
tokens = clean_and_tokenize(data)

unique_words = set(tokens)
word_id = {word:i for (i,word) in enumerate(unique_words)}
id_word = {i:word for (i,word) in enumerate(unique_words)}

window_size = 2

def target_context_tuples(tokens, window_size):
    context = []
    for i, token in enumerate(tokens):
        context_words = [t for t in merge(tokens, i, window_size) if t != token]
        for c in context_words:
            context.append((token, c))
    return context


def merge(tokens, i, window_size):
    left_id = i - window_size if i >= window_size else i - 1 if i != 0 else i
    right_id = i + window_size + 1 if i + window_size <= len(tokens) else len(tokens)
    return tokens[left_id:right_id]

target_context_pairs = target_context_tuples(tokens, 2)

import pandas as pd
df = pd.DataFrame(target_context_pairs, columns=["target","context"])

import torch.nn.functional as F
import torch

vocab_size = len(unique_words)
token_indexes = [word_id[token] for token in unique_words]
encodings = F.one_hot(torch.tensor(token_indexes), num_classes=vocab_size).float()

df["target_ohe"] = df["target"].apply(lambda x : encodings[word_id[x]])
df["context_ohe"] = df["context"].apply(lambda x : encodings[word_id[x]])

from torch.utils.data import Dataset

class W2VDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context = df["context_ohe"][idx]
        target = df["target_ohe"][idx]
        return context, target

dataset = W2VDataset(df)

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.linear1 = torch.nn.Linear(vocab_size, embed_size)
        self.linear2 = torch.nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_SIZE = 10
model = Word2Vec(vocab_size, EMBED_SIZE)
model.to(device)
LR = 1e-2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

epochs = 300

loss_values = []
for epoch in range(epochs):
    running_loss = 0.0
    # model.train() # no need since model is in train mode by default
    for batch, (context, target) in enumerate(dataloader):
        context = context.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(context)
        loss = loss_fn(pred, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss/len(dataloader)
    if (epoch+1)%10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")

    loss_values.append(epoch_loss)
    



import random
import numpy as np
import time
import string
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from rnn import CharRNN
from names_dataset import NamesDataset
from data_prep import lineToTensor


def train(rnn, training_data, n_epochs = 10, batch_size = 64, report_every = 50, lr = 0.2, criterion = nn.NLLLoss()):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.to(device)
    criterion.to(device)
    
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
    
    start = time.time()
    print(f"Training on dataset with n = {len(training_data)} and {n_epochs} epochs")
    
    for iter in range(1, n_epochs+1):
        
        # clearing the gradients
        rnn.zero_grad()
        
        # we cannot use dataloaders as names are of different length
        # creating mini-batches
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // batch_size)
        
        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = training_data[i]
                label_tensor, text_tensor = label_tensor.to(device), text_tensor.to(device)
                output = rnn(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss
                
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
            
            current_loss += batch_loss.item() / len(batch)
            
        all_losses.append(current_loss / len(batches))
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epochs:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0
        
    return all_losses



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
allowed_characters = string.ascii_letters + " .,;'"
n_letters = len(allowed_characters)

n_hidden = 128
all_data = NamesDataset("./data/names")
train_set, test_set = torch.utils.data.random_split(all_data, [.85, .15])
generator = torch.Generator(device=device).manual_seed(42)
rnn = CharRNN(n_letters, n_hidden, len(all_data.labels_uniq)).to(device)
#all_data = NamesDataset("./data/names")
#train_set, test_set = torch.utils.data.random_split(all_data, [.85, .15])

start = time.time()
all_losses = train(rnn, train_set, n_epochs=20, lr=0.15, report_every=5)

plt.figure()
plt.plot(all_losses)
plt.show()
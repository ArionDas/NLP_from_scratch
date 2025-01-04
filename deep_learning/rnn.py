import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from names_dataset import NamesDataset
from data_prep import lineToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

allowed_characters = string.ascii_letters + " .,;'"
n_letters = len(allowed_characters)

all_data = NamesDataset("./data/names")
train_set, test_set = torch.utils.data.random_split(all_data, [.85, .15])
generator = torch.Generator(device=device).manual_seed(42)

#print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")

class CharRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        
        return output
    

n_hidden = 128
rnn = CharRNN(n_letters, n_hidden, len(all_data.labels_uniq)).to(device)
#print(rnn)

def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

# input = lineToTensor('Arion').to(device)
# output = rnn(input)
# print(output)
# print(label_from_output(output, all_data.labels_uniq))

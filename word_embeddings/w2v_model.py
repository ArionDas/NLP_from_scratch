import torch
import torch.nn as nn

class W2V_Model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(W2V_Model, self).__init__()
        self.linear1 = nn.Linear(vocab_size, embed_size)
        self.linear2 = nn.Linear(embed_size, vocab_size, bias=False)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    ## Notice how I'm not using a softmax layer here. 
    ## This is because the loss function I'm using, nn.CrossEntropyLoss, already applies a softmax function.
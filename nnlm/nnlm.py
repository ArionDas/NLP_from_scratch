import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []
    
    for sen in sentences:
        
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]
        
        input_batch.append(input)
        target_batch.append(target)
        
    return input_batch, target_batch


class nnlm(nn.Module):
    
    def __init__(self):
        super(nnlm, self).__init__()
        self.C = nn.Embedding(num_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, num_class, bias=False)
        self.W = nn.Linear(n_step * m, num_class, bias=False)
        self.b = nn.Parameter(torch.ones(num_class))
        
    
    def forward(self, x):
        x = self.C(x)
        x = x.view(-1, n_step * m)
        tanh = torch.tanh(self.d + self.H(x))
        output = self.b + self.W(x) + self.U(tanh)
        return output
    
        
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    n_step = 2
    n_hidden = 2
    m = 2
    
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    num_class = len(word_dict)
    
    model = nnlm().to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch).to(device)
    target_batch = torch.LongTensor(target_batch).to(device)
    print(f"Input batch is on device: {input_batch.device}")
    print(f"Target batch is on device: {target_batch.device}")
    
    for epoch in range(5000):
        
        optimizer.zero_grad()
        output = model(input_batch)
        
        loss = criterion(output, target_batch)
        
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            
        loss.backward()
        optimizer.step()
        
    
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze().cpu()])
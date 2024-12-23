import torch
from torch import nn
from w2v_model import W2V_Model


def train(vocab_size, embed_size, lr, epochs, dataloader):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = W2V_Model(vocab_size, embed_size).to(device)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_values = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch, (context, target) in enumerate(dataloader):
            
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            pred = model(context)
            
            loss = loss_fn(pred, target)
            
            running_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            
        epoch_loss = running_loss / len(dataloader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")

        loss_values.append(epoch_loss)
        
    return model
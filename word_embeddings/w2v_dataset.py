from torch.utils.data import Dataset

class W2V_Dataset(Dataset):
    
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitiem__(self, idx):
        context = self.df["context_one_hot"][idx]
        target = self.df["target_one_hot"][idx]
        return context, target
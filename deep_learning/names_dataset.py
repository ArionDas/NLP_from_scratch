from io import open
import os
import glob
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data_prep import lineToTensor


class NamesDataset(Dataset) : 
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()
        
        self.data = []
        self.data_tensors = []
        self.labels = []
        self.label_tensors = []
        
        text_files = glob.glob(os.path.join(data_dir, "*.txt"))
        for filename in text_files : 
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            
            for name in lines: 
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)
                
            self.labels_uniq = list(labels_set)
            for idx in range(len(self.labels)):
                temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
                self.label_tensors.append(temp_tensor)
                
        
    def __len__(self):
        return len(self.data)
            
            
    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.label_tensors[idx]
                
        return label_tensor, data_tensor, data_label, data_item
        
        
# all_data = NamesDataset("./data/names")
# print(f"Length of dataset: {len(all_data)}")
# print(f"Sample data: {all_data[0]}")

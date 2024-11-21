import json
import torch
import pandas as pd
from torch.utils.data import Dataset


class SeriesData(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        self.data = pd.read_csv(csv_file)
        with open('../label.json') as f:
            label_str = f.read()
            label = json.loads(label_str)
        self.target = label["1"]
        self.train = train
        self.transform = transform

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        ticket = self.data.columns[idx]
        s = self.data.iloc[:, idx]
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0) # unsqz. for adding 1 channel dimension for cnn
        if self.transform:
            s = self.transform(s)
        if self.train:
            label = 1 if ticket in self.target else 0
            label = torch.tensor(label, dtype=torch.long)
            return s, label
        else:
            return s
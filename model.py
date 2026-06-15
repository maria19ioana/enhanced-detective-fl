import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class FraudDetectionMLP(nn.Module):
    def __init__(self, input_dim=30):
        super(FraudDetectionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class FraudDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # pre-procesare
        scaler = StandardScaler()
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
        df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
        
        self.X = torch.tensor(df.drop('Class', axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df['Class'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_data(csv_path, batch_size=32):
    dataset = FraudDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
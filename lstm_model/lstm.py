import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader
import pandas as pd
import numpy as np
import os

class TactileDataset(Dataset):
    def __init__(self, path, mode="train", max_seq_length=30):
        self.data_path = path+mode+'/'
        os.chdir(self.data_path)
        self.datapoints = os.listdir('.')
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, i):
        df = pd.read_csv('./' + self.datapoints[i])
        row, col = df.shape
        final_angle = df.iloc[-1:]['true_angle']

        df_tensor = torch.Tensor(df.values).float()
        
        # if row >= self.max_seq_length:
        #     df = df[:self.max_seq_length]
        # else:
        #     padding = np.zeros((self.max_seq_length-row, col))
        #     padding[:, -1] = final_angle
        #     df = df.append

        return df_tensor[:,0:-1], df_tensor[:,-1]
        
        

class RegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_LSTM_layers):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_LSTM_layers
        
        self.lstm = nn.LSTM(
            input_size = self.num_features,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.num_layers
        )
        
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        #out is size (batch, seq, feat*layers) for batch_first=True
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #Fold the bacth and seq dimensions together, so each sequence will be basically like a batch element
        out = self.linear(out.view(-1, out.size(2)))
        return out

def main():
    pass

if __name__=="__main__":
    main()
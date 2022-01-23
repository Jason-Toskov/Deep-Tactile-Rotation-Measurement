import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

class TactileDataset(Dataset):
    def __init__(self, path, mode="train"):   #, max_seq_length=30):
        self.data_path = path+mode+'/'
        os.chdir(self.data_path)
        self.datapoints = os.listdir('.')
        #self.max_seq_length = max_seq_length
    
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

        return df_tensor[:,0:-1], df_tensor[:,-1] - df_tensor[0,-1]
        
        

class RegressionLSTM(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_LSTM_layers):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_LSTM_layers
        self.device = device
        
        self.lstm = nn.LSTM(
            input_size = self.num_features,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.num_layers
        )
        
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        
        #out is size (batch, seq, feat*layers) for batch_first=True
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #Fold the bacth and seq dimensions together, so each sequence will be basically like a batch element
        out = self.linear(out.view(-1, out.size(2)))
        return out
    
def train(device, loader, model, loss_func, optim, l1loss):
    model.train()
    loss_count = 0
    abs_error_count = 0
    
    for i, (features, label) in enumerate(loader):
        out = model(features.to(device))
        loss = loss_func(torch.squeeze(out), torch.squeeze(label.to(device)))
        l1error = l1loss(torch.squeeze(out), torch.squeeze(label.to(device)))
        
        loss_count += loss.item()
        abs_error_count += l1error.item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    loss_count /= i+1
    abs_error_count /= i+1
    
    return loss_count, abs_error_count

def main():
    lr = 1e-4
    num_epochs = 1200
    
    #Set device to GPU_indx if GPU is avaliable
    GPU_indx = 0
    device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
    
    path = './Dataset_usbc_box/'
    
    data = TactileDataset(path, mode='train')
    loader = DataLoader(data, batch_size = 1, shuffle=True)
    
    features, label = next(iter(loader))
    
    
    model = RegressionLSTM(device, 142, 200, 1)
    model = model.to(device)
    # out = model(features.to(device))
    
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    l1loss = nn.L1Loss()
    
    loss_array = []
    error_array = []
    best_model = copy.deepcopy(model)
    lowest_error = 1e5
    for i in range(num_epochs):
        loss_current, abs_error_current = train(device, loader, model, loss_func, optim, l1loss)
        loss_array.append(loss_current)
        error_array.append(abs_error_current)
        print("Epoch: %i, Absolute error: %f, Loss: %f" % (i, abs_error_current, loss_current) )
        
        if abs_error_current < lowest_error:
            # print("new best! old best: %f, new best: %f"%(lowest_error, abs_error_current))
            best_model_dict = copy.deepcopy(model.state_dict())
            lowest_error = abs_error_current
            print("new best!")
    
    print("Lowest error was: %f"%(lowest_error))
    best_model = RegressionLSTM(device, 142, 200, 1)
    best_model.load_state_dict(best_model_dict)
    best_model = best_model.to(device)
        
    # breakpoint()
    #plot that shows labels/out of some sequences
    fig, axs = plt.subplots(2, 2)
    loader = DataLoader(data, batch_size = 1, shuffle=True)
    # breakpoint()
    
    for ax in axs.flat:
        features, label = next(iter(loader))
        while(max(label.squeeze())-min(label.squeeze()) < 5):
            features, label = next(iter(loader))
        out = best_model(features.to(device))
        out = out.squeeze()
        label = label.squeeze()
        x_range = [*range(len(out))]
        ax.plot(x_range, label.detach().to('cpu'), label = 'Ground truth')
        ax.plot(x_range, out.detach().to('cpu'), label = 'Prediction')
    
    os.chdir('..')
    plt.savefig('../examples.png')
    plt.show()
    

    plt.title("Loss (MSE)")
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.plot(np.array([*range(num_epochs)])+1, loss_array)
    plt.grid(True)
    plt.savefig('../loss.png')
    plt.show()
    
    plt.title("Absolute Error")
    plt.xlabel('Epoch number')
    plt.ylabel('Error')
    plt.plot(np.array([*range(num_epochs)])+1, error_array)
    plt.grid(True)
    plt.savefig('../error.png')
    plt.show()
    
    # breakpoint()
    

if __name__=="__main__":
    main()
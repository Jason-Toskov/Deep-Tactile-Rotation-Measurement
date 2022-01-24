import torch
from torch import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import time
import wandb

wandb.init(project="SRP")

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
        # row, col = df.shape
        # final_angle = df.iloc[-1:]['true_angle']

        df_tensor = torch.Tensor(df.values).float()
        
        # if row >= self.max_seq_length:
        #     df = df[:self.max_seq_length]
        # else:
        #     padding = np.zeros((self.max_seq_length-row, col))
        #     padding[:, -1] = final_angle
        #     df = df.append

        return df_tensor[:,0:-2], (df_tensor[:,-2] - df_tensor[0,-2])/90
        
        

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
            num_layers = self.num_layers,
            dropout = 0.2
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

def test(device, loader, model, loss_func, optim, l1loss):
    model.eval()

    loss_count = 0
    abs_error_count = 0

    with torch.no_grad():
        for i, (features, label) in enumerate(loader):
            out = model(features.to(device))
            loss = loss_func(torch.squeeze(out), torch.squeeze(label.to(device)))
            l1error = l1loss(torch.squeeze(out), torch.squeeze(label.to(device)))
            
            loss_count += loss.item()
            abs_error_count += l1error.item()
            
            # optim.zero_grad()
            # loss.backward()
            # optim.step()
            
        loss_count /= i+1
        abs_error_count /= i+1
    
    return loss_count, abs_error_count


def main():
    lr = 1e-4
    num_epochs = 1200
    train_frac = 0.8

    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "train_fraction": train_frac
    }
    
    #Set device to GPU_indx if GPU is avaliable
    GPU_indx = 0
    device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
    
    path = './Data/'
    
    data = TactileDataset(path, mode='train')
    train_data_length = round(len(data)*train_frac)
    test_data_length = len(data) - train_data_length
    train_data, test_data = random_split(data, [train_data_length, test_data_length], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size = 1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size = 1, shuffle=True)
    
    # features, label = next(iter(train_data))
    
    
    model = RegressionLSTM(device, 142, 200, 2)
    model = model.to(device)
    # out = model(features.to(device))
    
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    l1loss = nn.L1Loss()
    
    loss_array_train = []
    error_array_train = []
    loss_array_test= []
    error_array_test = []
    best_model = copy.deepcopy(model)
    lowest_error = 1e5
    old_time = time.time()
    for i in range(num_epochs):
        loss_train, abs_error_train = train(device, train_loader, model, loss_func, optim, l1loss)
        loss_array_train.append(loss_train)
        error_array_train.append(abs_error_train*90)

        loss_test, abs_error_test = test(device, test_loader, model, loss_func, optim, l1loss)
        loss_array_test.append(loss_test)
        error_array_test.append(abs_error_test*90)

        wandb.log({
            "loss/train":loss_train,
            "loss/test":loss_test,
            "abs_error/train":abs_error_train*90,
            "abs_error/test":abs_error_test*90,
        })

        new_time = time.time()
        print("Epoch: %i, Absolute error: %f, Loss: %f, Time taken: %f" % (i, abs_error_train*90, loss_train, new_time-old_time) )
        old_time = copy.deepcopy(new_time)
        if abs_error_train < lowest_error:
            # print("new best! old best: %f, new best: %f"%(lowest_error, abs_error_current))
            best_model_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), '../../best_model.pt')
            lowest_error = abs_error_train
            print("new best!")
    
    print("Lowest error was: %f"%(lowest_error))
    best_model = RegressionLSTM(device, 142, 200, 2)
    best_model.load_state_dict(best_model_dict)
    best_model = best_model.to(device)

        
    # breakpoint()
    #plot that shows labels/out of some sequences
    fig, axs = plt.subplots(2, 2)
    train_loader = DataLoader(train_data, batch_size = 1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size = 1, shuffle=True)
    # breakpoint()
    
    for ax in axs.flat:
        features, label = next(iter(train_loader))
        while(max(label.squeeze())-min(label.squeeze()) < 5):
            features, label = next(iter(train_loader))
        out = best_model(features.to(device))
        out = out.squeeze()
        label = label.squeeze()
        x_range = [*range(len(out))]
        ax.plot(x_range, label.detach().to('cpu')*90, label = 'Ground truth')
        ax.plot(x_range, out.detach().to('cpu')*90, label = 'Prediction')
    
    fig.suptitle("Train examples")
    # os.chdir('..')
    plt.savefig('../../examples_train.png')
    plt.show()

    fig, axs = plt.subplots(2, 2)
    for ax in axs.flat:
        features, label = next(iter(test_loader))
        while(max(label.squeeze())-min(label.squeeze()) < 5):
            features, label = next(iter(test_loader))
        out = best_model(features.to(device))
        out = out.squeeze()
        label = label.squeeze()
        x_range = [*range(len(out))]
        ax.plot(x_range, label.detach().to('cpu')*90, label = 'Ground truth')
        ax.plot(x_range, out.detach().to('cpu')*90, label = 'Prediction')
    
    fig.suptitle("Test examples")
    # os.chdir('..')
    plt.savefig('../../examples_test.png')
    plt.show()
    

    os.chdir('..')

    plt.title("Loss (MSE)")
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.plot(np.array([*range(num_epochs)])+1, loss_array_train, label="Train")
    plt.plot(np.array([*range(num_epochs)])+1, loss_array_test, label="Test")
    plt.grid(True)
    plt.legend()
    plt.savefig('../loss.png')
    plt.show()
    
    plt.title("Absolute Error")
    plt.xlabel('Epoch number')
    plt.ylabel('Error')
    plt.plot(np.array([*range(num_epochs)])+1, error_array_train, label="Train")
    plt.plot(np.array([*range(num_epochs)])+1, error_array_test, label="Test")
    plt.grid(True)
    plt.legend()
    plt.savefig('../error.png')
    plt.show()
    
    # breakpoint()
    

if __name__=="__main__":
    main()
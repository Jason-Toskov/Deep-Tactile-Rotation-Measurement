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
import argparse
import sys
import yaml
from pathlib import Path
from arg_set import parse_arguments
from lstm_papilarray import RegressionLSTM, TactileDataset

args = parse_arguments()

# Check if sweeping or using default config
if len(sys.argv) == 1:
    config_file = args.config
    with open(config_file, 'r') as f:
        cfg_input = yaml.safe_load(f) 
elif len(sys.argv) > 1:
    del args.config
    cfg_input = args
else:
    raise ValueError("Weird arg error")

run = wandb.init(project="SRP", config=cfg_input)
config = wandb.config
print(config)

#Set device to GPU_indx if GPU is avaliable
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')


#Load best model
best_model = RegressionLSTM(device, config["num_features"], config["hidden_size"], config["num_layers"], config["dropout"])
best_model.load_state_dict(torch.load(config["model_path"]))
best_model = best_model.to(device)

data = TactileDataset(config["data_path"], label_scale = config["label_scale"])
train_data_length = round(len(data)*config["train_frac"])
test_data_length = len(data) - train_data_length
train_data, test_data = random_split(data, [train_data_length, test_data_length], generator=torch.Generator().manual_seed(42))


#Refresh loaders
train_loader = DataLoader(train_data, batch_size = 1, shuffle=True)
test_loader = DataLoader(test_data, batch_size = 1, shuffle=True)
    
#plot that shows labels/out of some sequences
fig, axs = plt.subplots(10, 10)
for ax in axs.flat:
    features, label = next(iter(train_loader))
    count = 0
    while(max(label.squeeze())-min(label.squeeze()) < 5/config["label_scale"]) and count < len(train_data):
        features, label = next(iter(train_loader))
        count += 1

    out = best_model(features.to(device))
    out = out.squeeze()
    label = label.squeeze()
    x_range = [*range(len(out))]
    ax.plot(x_range, label.detach().to('cpu')*config["label_scale"], label = 'Ground truth')
    ax.plot(x_range, out.detach().to('cpu')*config["label_scale"], label = 'Prediction')


fig.suptitle("Train examples")
wandb.log({'Examples/Train': fig})
plt.savefig("./results_1/" + 'examples_train.png')
# plt.show()

fig, axs = plt.subplots(10, 10)
for ax in axs.flat:
    features, label = next(iter(test_loader))
    count = 0
    while(max(label.squeeze())-min(label.squeeze()) < 5/config["label_scale"]) and count < len(test_data):
        features, label = next(iter(test_loader))
        count += 1
    out = best_model(features.to(device))
    out = out.squeeze()
    label = label.squeeze()
    x_range = [*range(len(out))]
    ax.plot(x_range, label.detach().to('cpu')*config["label_scale"], label = 'Ground truth')
    ax.plot(x_range, out.detach().to('cpu')*config["label_scale"], label = 'Prediction')

fig.suptitle("Test examples")
wandb.log({'Examples/Test': fig})
plt.savefig("./results_1/" + 'examples_train.png')
# plt.show()

print("Training complete!")


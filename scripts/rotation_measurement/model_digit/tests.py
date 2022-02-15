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
import glob
import platform
from pathlib import Path
import torchvision.transforms.functional as F
from PIL import Image
from pytouch import PyTouchZoo, sensors
from pytouch.models.touch_detect import TouchDetectModel
from torchvision import models

from arg_set import parse_arguments


class DigitDataset(Dataset):
    def __init__(self, path, mode=None, seq_length=None, label_scale = 1):
        self.data_path = path if mode is None else path+mode+'/'
        self.initial_path = os.getcwd()
        os.chdir(self.data_path)
        self.datapoints = os.listdir('.')
        os.chdir(self.initial_path)
        self.seq_length = seq_length
        self.label_scale = label_scale
        self.platform = platform.system()
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, i):
        path_to_data = self.data_path+self.datapoints[i]+'/'
        images_left = glob.glob(path_to_data+'*0.jpeg')
        images_right = glob.glob(path_to_data+'*1.jpeg')
        
        split_el = '\\' if self.platform == 'Windows' else '/'
        
        images_left.sort(key=lambda x: x.split(split_el)[-1].split('_')[0])
        images_right.sort(key=lambda x: x.split(split_el)[-1].split('_')[0])
        
        seq_length = len(images_left)
        
        df = pd.read_csv(path_to_data+'ground_truth.csv')
        
        data_left = []
        data_right = []
        for im_left_pth, im_right_pth in zip(images_left, images_right):
            im_left = Image.open(im_left_pth)
            im_right = Image.open(im_right_pth)
            
            data_left.append(F.to_tensor(im_left))
            data_right.append(F.to_tensor(im_right))
            
        tensor_left = torch.stack(data_left)
        tensor_right = torch.stack(data_right)
        
        df_tensor = torch.Tensor(df.values).float()
        # breakpoint()
        
        return tensor_left, tensor_right, (df_tensor[:,0] - df_tensor[0, 0])/self.label_scale


dset = DigitDataset('./digit_data_unpacked/')

loader = DataLoader(dset, batch_size = 1, shuffle=True)

min_length = 10000
for t_L, t_R, lab in loader:
    if t_L.size()[1] < min_length:
        min_length = t_L.size()[1]
        print(min_length)

print(min_length)
        
import numpy as np
import torch
import glob
import pickle
from cv2 import transform
import torch
# from torch import 
import random
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
from torchvision import models, transforms
from tqdm import tqdm
from enum import Enum
from arg_set import parse_arguments


# dirs = []
# for dir in glob.glob('./rotation-detection-IROS-2021/RobotData/Results/GroundTruth/*.npy'):
#     dirs.append(dir)
    
# print(len(dirs))
# arr = np.load(dirs[5], allow_pickle=True)
# print(arr)
# print('\n\n')
# print(arr[()]['rotationOnset'])

# print('\n\n')
# temp = [v for v in arr[()].values()][1:]
# print(torch.Tensor(temp))

# print(dirs[5])

class SampleType(Enum):
    FRONT = 0
    CENTER = 1
    RANDOM = 2

class DigitDataset(Dataset):
    def __init__(self, path, mode=None, seq_length=None, label_scale = 1, angle_difference=False, sample_type = SampleType.RANDOM, diff_from_start=False):
        self.data_path = path if mode is None else path+mode+'/'
        # self.initial_path = os.getcwd()
        # os.chdir(self.data_path)
        # self.datapoints = os.listdir('.')
        # os.chdir(self.initial_path)
        self.dirs = [dir for dir in glob.glob(self.data_path+"*_*/")]
        self.gt_dir = self.data_path + "Results/GroundTruth/"
        
        # breakpoint()
        self.seq_length = seq_length
        self.label_scale = label_scale
        self.platform = platform.system()
        self.angle_difference = angle_difference
        self.sample_type = sample_type
        self.diff_from_start = diff_from_start
        
        self.trans = transforms.Compose([
            transforms.Resize([64,64]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def trans(self):
        trans_list = [
            transforms.Resize([64,64]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        return transforms.Compose(trans_list)
    
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, i):
        path_to_data = self.dirs[i]
        
        split_el = '\\' if self.platform == 'Windows' else '/'
        datapoint_name = path_to_data.split(split_el)[-2]
        
        datapoint_gt_dir = self.gt_dir + datapoint_name + '.npy'
        
        images = glob.glob(path_to_data + "gelSightMerge/*.jpg")
        images.sort(key = lambda x: x.split('/')[-1].split('.')[0])
        
        seq_length = len(images)
        gt_dict = np.load(datapoint_gt_dir, allow_pickle=True)[()]
        
        init_angle = gt_dict['rotationOnset']
        try:
            gt_tensor = torch.Tensor([v for v in gt_dict.values() ][1:])
        except TypeError:
            print(datapoint_name)
            breakpoint()
            gt_tensor = None
        
        data_tensor = torch.stack([self.trans(Image.open(x)) for x in images])
        
        # breakpoint()
    
        return data_tensor, gt_tensor
    

    
sample_type = SampleType.RANDOM
seq_length = 15
angle_difference = False
diff_from_start = True

# Parse args
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

run = wandb.init(project="LSTM_digit_box", 
                    entity="deep-tactile-rotatation-estimation", 
                    config=cfg_input,
                    notes="Not relative angle",
                    mode="disabled"
)

config = wandb.config
print(config)

if config['sample'] == 'random':
    sample_type = SampleType.RANDOM
elif config['sample'] == 'center':
    sample_type = SampleType.CENTER
elif config['sample'] == 'front':
    sample_type = SampleType.FRONT

#Set device to GPU_indx if GPU is avaliable
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
print(device)
    
data = DigitDataset(config["data_path"], label_scale = config["label_scale"], sample_type=sample_type, seq_length=seq_length, angle_difference=angle_difference, diff_from_start=diff_from_start)
    
count = 0
for data_tensor, gt_tensor in tqdm(data):
    if gt_tensor is None:
        count += 1

print(count)
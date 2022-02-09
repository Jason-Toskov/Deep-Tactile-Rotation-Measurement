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

class SampleType(Enum):
    FRONT = 0
    CENTER = 1
    RANDOM = 2
    
class GelSightDataset(Dataset):
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
            transforms.Resize([256,256]),
            transforms.CenterCrop([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # def trans(self):
    #     trans_list = [
    #         transforms.Resize([64,64]),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ]
        
    #     return transforms.Compose(trans_list)
    
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
            gt_tensor = torch.Tensor([v if v is not None else 0 for v in gt_dict.values() ][1:])
            gt_tensor += init_angle
        except TypeError:
            breakpoint()
        
        data_tensor = torch.stack([self.trans(Image.open(x)) for x in images])
        
        # breakpoint()
    
        return data_tensor - data_tensor[0], gt_tensor/self.label_scale
    
    def collate_fn(self, batch):

        # 0 is the data, 1 is GT
        min_length = min(map(lambda x : x[0].shape[0], batch))
        # breakpoint()

        # https://www.geeksforgeeks.org/python-k-middle-elements/
        if self.seq_length is None:
            K = min_length
            # print("HERE", K)
        elif self.seq_length > min_length:
            print(self.seq_length,' ',min_length)
            K = min_length
            # ValueError('Seq length is too long!')
        else:
            K = self.seq_length
        
        torch_array = torch.zeros((len(batch), K, batch[0][0].shape[1], batch[0][0].shape[2], batch[0][0].shape[3]))
        if self.angle_difference:
            gt_array = torch.zeros((len(batch)))  
        else:
            gt_array = torch.zeros((len(batch), K))        
        # print(torch_array.shape)
        # input()   

        for (index, (data, gt)) in enumerate(batch):

            # computing strt, and end index 
            strt_idx = 0 #(len(data) // 2) - (K // 2)
            end_idx = (len(data) // 2) + (K // 2)

            if self.sample_type == SampleType.CENTER:
                strt_idx = (len(data) // 2) - (K // 2)
                # print(K, strt_idx, len(data))
            elif self.sample_type == SampleType.FRONT:
                strt_idx = 0
            elif self.sample_type == SampleType.RANDOM:
                max_start_value = len(data) - K -1
                # print()
                if max_start_value == 0:
                    max_start_value = 1
                strt_idx = random.randrange(0, max_start_value, 1)
                # print(strt_idx)
            else:
                ValueError('Invalid sample type')
            # print(data.shape, gt.shape)
            # print(data_cropped.shape, gt_cropped.shape)
            # print("cropped size", data[strt_idx: end_idx + 1, :].shape)
            # slicing extracting middle elements
            data_cropped = data[strt_idx: strt_idx + K, :, :, :]
            gt_cropped = gt[strt_idx: strt_idx + K]
            # print(data_cropped.shape, gt_cropped.shape)
            # input()            
            
            if self.diff_from_start:
                torch_array[index, :, :, :, :] = data_cropped - data_cropped[0]
            else:
                torch_array[index, :, :, :, :] = data_cropped
                
            if self.angle_difference:
                gt_array[index] = torch.tensor(gt_cropped - gt_cropped[0])[-1]
            else:
                # gt_array[index, :] = torch.tensor(gt_cropped - gt_cropped[0])
                try:
                    gt_array[index, :] = torch.tensor(gt_cropped)
                except:
                    data_cropped = data[0: 0 + K, :, :, :]
                    gt_cropped = gt[0: 0 + K]
                    torch_array[index, :, :, :, :] = data_cropped

                    # breakpoint()
            # breakpoint()
            # print((gt_cropped - gt_cropped[0]))
            # print(gt_cropped)
            # input()

        return torch_array.view(len(batch)*K, batch[0][0].shape[1], batch[0][0].shape[2], batch[0][0].shape[3]), gt_array.view(len(batch)*K)

class RegressionLSTM(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        pytouch_zoo = PyTouchZoo()
        touch_detect_model_dict = pytouch_zoo.load_model_from_zoo(  # noqa: F841
            "touchdetect_resnet", sensors.DigitSensor
        )
        # self.touch_model = TouchDetectModel(state_dict = touch_detect_model_dict)
        self.res = models.resnext50_32x4d()
        # self.res.fc = nn.Linear(self.res.fc.in_features, 2)
        # self.res.load_state_dict(touch_detect_model_dict)
        self.res.fc = nn.Linear(self.res.fc.in_features, self.num_features)
        
        
        self.lstm = nn.LSTM(
            input_size = self.num_features,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.num_layers,
            dropout = dropout
        )
        
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        # print('step 1')
        
        x = torch.squeeze(x, 0)
        #Out is size (seq, feat)
        res_out = self.res(x)
        # print('step 2')
        
        #Add in a batch dimension
        res_out = torch.unsqueeze(res_out, 0)    
        # print('step 3')    
        #out is size (batch, seq, feat*layers) for batch_first=True
        out, (hn, cn) = self.lstm(res_out, (h0, c0))
        # print('step 4')
        #Fold the bacth and seq dimensions together, so each sequence will be basically like a batch element
        out = self.linear(out.view(-1, out.size(2)))
        # print('step 5')
        return out
    
class RegressionCNN(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        pytouch_zoo = PyTouchZoo()
        touch_detect_model_dict = pytouch_zoo.load_model_from_zoo(  # noqa: F841
            "touchdetect_resnet", sensors.DigitSensor
        )
        # self.touch_model = TouchDetectModel(state_dict = touch_detect_model_dict)
        self.res = models.resnext50_32x4d()
        # self.res.fc = nn.Linear(self.res.fc.in_features, 2)
        # self.res.load_state_dict(touch_detect_model_dict)
        self.res.fc = nn.Linear(self.res.fc.in_features, self.num_features)
        
        
        # self.lstm = nn.LSTM(
        #     input_size = self.num_features,
        #     hidden_size = self.hidden_size,
        #     batch_first = True,
        #     num_layers = self.num_layers,
        #     dropout = dropout
        # )
        
        self.linear = nn.Linear(in_features=self.num_features, out_features=1)
        
    def forward(self, x):
        # batch_size = x.shape[0]
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        # print('step 1')
        
        x = torch.squeeze(x, 1)
        #Out is size (batch, feat)
        res_out = self.res(x)
        # print('step 2')
        
        #Add in a batch dimension
        # res_out = torch.unsqueeze(res_out, 0)    
        # print('step 3')    
        #out is size (batch, seq, feat*layers) for batch_first=True
        # out, (hn, cn) = self.lstm(res_out, (h0, c0))
        # print('step 4')
        #Fold the bacth and seq dimensions together, so each sequence will be basically like a batch element
        out = self.linear(torch.flatten(res_out, 1))
        # print('step 5')
        return out
    
def train(device, loader, model, loss_func, optim, l1loss):
    model.train()
    loss_count = 0
    abs_error_count = 0
    
    for i, (features, label) in tqdm(enumerate(loader)):
        # print(features1.size())
        out = model(features.to(device))
        loss = loss_func(out.squeeze(), label.to(device).squeeze())
        l1error = l1loss(out.squeeze(), label.to(device).squeeze())
        
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
        for i, (features, label) in tqdm(enumerate(loader)):
            out = model(features.to(device))
            loss = loss_func(out.squeeze(), label.to(device).squeeze())
            l1error = l1loss(out.squeeze(), label.to(device).squeeze())
            
            loss_count += loss.item()
            abs_error_count += l1error.item()
            
        loss_count /= i+1
        abs_error_count /= i+1
    
    return loss_count, abs_error_count


def main():
    sample_type = SampleType.RANDOM
    seq_length = 10
    angle_difference = False
    diff_from_start = False
    
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
    
    run = wandb.init(project="Gelsight_models", 
                     entity="deep-tactile-rotatation-estimation", 
                     config=cfg_input,
                     notes="diff image, larger seq size",
                    #  mode="disabled"
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
    
    # Create dataset/dataloaders
    data = GelSightDataset(config["data_path"], label_scale = config["label_scale"], sample_type=sample_type, seq_length=seq_length, angle_difference=angle_difference, diff_from_start=diff_from_start)
    train_data_length = round(len(data)*config["train_frac"])
    test_data_length = len(data) - train_data_length
    train_data, test_data = random_split(data, [train_data_length, test_data_length], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size = config["train_batch_size"], shuffle=True, collate_fn=data.collate_fn)
    test_loader = DataLoader(test_data, batch_size = config["test_batch_size"], shuffle=True, collate_fn=data.collate_fn)
    print('Data loaders created')
    
    # pytouch_zoo = PyTouchZoo()
    # available_models = pytouch_zoo.list_models()
    # print(available_models)
    
    # # load DIGIT sensor touch detect model from pytouch zoo
    # # This is a state dict for the TouchDetectModel
    # # Code should go into lstm initialization
    # # will want to collapse the seq batch dimension into the batch dimension (somehow?)
    # touch_detect_model_dict = pytouch_zoo.load_model_from_zoo(  # noqa: F841
    #     "touchdetect_resnet", sensors.DigitSensor
    # )
    
    # touch_model = TouchDetectModel(state_dict = touch_detect_model_dict)

    # x,y,z = next(iter(train_loader))
    
    # res_output = touch_model(x.squeeze())
    # breakpoint()
    
    # Create model
    model = RegressionCNN(device, config["num_features"], config["hidden_size"], config["num_layers"], config["dropout"])
    print('Model created')
    
    # breakpoint()
    if config["resume_from_checkpoint"]:
        model.load_state_dict(torch.load(config["model_path"]))
    model = model.to(device)
    
    # Define loss and optimization functions
    loss_func = nn.MSELoss()
    l1loss = nn.L1Loss()
    
    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    
    if config["test_only"]:
        loss_train, abs_error_train = test(device, train_loader, model, loss_func, optim, l1loss)
        loss_test, abs_error_test = test(device, test_loader, model, loss_func, optim, l1loss)
        wandb.log({
                "Loss/train":loss_train,
                "Loss/test":loss_test,
                "abs_error/train":abs_error_train*config["label_scale"],
                "abs_error/test":abs_error_test*config["label_scale"],
            })
        print('Train error: %f, Train loss: %f, Test error: %f, Test loss: %f'%(abs_error_train*config["label_scale"], loss_train, abs_error_test*config["label_scale"], loss_test))
    else:
        # Run training
        best_model = copy.deepcopy(model)
        lowest_error = 1e5
        old_time = time.time()
        print('Training began')
        for i in range(config["num_epochs"]):
            print(str(i))
            loss_train, abs_error_train = train(device, train_loader, model, loss_func, optim, l1loss)
            loss_test, abs_error_test = test(device, test_loader, model, loss_func, optim, l1loss)

            wandb.log({
                "Loss/train":loss_train,
                "Loss/test":loss_test,
                "abs_error/train":abs_error_train*config["label_scale"],
                "abs_error/test":abs_error_test*config["label_scale"],
            })

            new_time = time.time()
            print("Epoch: %i, Test error: %f, Test loss: %f, Time taken: %.2f sec/epoch" % (i, abs_error_test*config["label_scale"], loss_test, new_time-old_time) )
            old_time = copy.deepcopy(new_time)
            if abs_error_test < lowest_error:
                best_model_dict = copy.deepcopy(model.state_dict())
                torch.save(best_model_dict, config["model_path"])
                lowest_error = abs_error_test
                print("new best!")
        
        print("Lowest error was: %f"%(lowest_error*config["label_scale"]))
        
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(config["model_path"]) 
        run.log_artifact(artifact)
        # run.join()  
    
    if seq_length != 1:
        #Load best model
        best_model = RegressionCNN(device, config["num_features"], config["hidden_size"], config["num_layers"], config["dropout"])
        best_model.load_state_dict(torch.load(config["model_path"]))
        best_model = best_model.to(device)
        
        
        #Refresh loaders
        train_loader = DataLoader(train_data, batch_size = 1, shuffle=True, collate_fn=data.collate_fn)
        test_loader = DataLoader(test_data, batch_size = 1, shuffle=True, collate_fn=data.collate_fn)
            
        #plot that shows labels/out of some sequences
        fig, axs = plt.subplots(5, 5)
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
        wandb.log({'Examples_Train': fig})
        # plt.savefig(plot_path + 'examples_train.png')
        # plt.show()

        fig, axs = plt.subplots(5, 5)
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
        wandb.log({'Examples_Test': fig})
        # plt.savefig(plot_path + 'examples_train.png')
        # plt.show()
    
    print("Training complete!")
    

if __name__=="__main__":
    main()
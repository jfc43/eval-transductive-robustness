import numpy as np
import os
import torch
import random
from utils.data_util import *
from collections import defaultdict
from torchvision import transforms

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def rand(maxi=10**9):
    return np.random.randint(maxi)

def get_transformed_data(xs, ys):
    dataset = MyDataset(xs, ys, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False) 
    
    new_xs = []
    new_ys = []
    for batch_x, batch_y in dataloader:
        new_xs.extend(batch_x.numpy())
        new_ys.extend(batch_y.numpy())
    
    return np.array(new_xs), np.array(new_ys)

def select_samples(xs, ys):
    new_xs = []
    new_ys = []
    for x, y in zip(xs, ys):
        if y<10:
            new_xs.append(x)
            new_ys.append(y)
    
    return np.array(new_xs), np.array(new_ys)
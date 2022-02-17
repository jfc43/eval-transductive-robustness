import os
import pickle
import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
version = sys.version_info

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
import skimage.data
import scipy.io as sio
import cv2

def image_brightness_normalisation(image):
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    return image

def preprocess_data(X):
    
    for i in range(len(X)):
        X[i,:,:,:] = image_brightness_normalisation(X[i,:,:,:])
   
    return X


def get_gtsrb_data(path):
    loaded = np.load(os.path.join(path, 'train.npz'))
    train_images = loaded['images']
    train_images = preprocess_data(train_images)
    train_labels = loaded['labels']
    
    return (train_images, train_labels)


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
    
class MyDatasetNoLabel(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)
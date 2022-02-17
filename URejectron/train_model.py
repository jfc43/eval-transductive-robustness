import argparse
import numpy as np
import os
import torch
from utils.data_util import *
from utils.lib import *
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.optim as optim
from model import ResNet18


def train_model(model, train_dataloader, nepoch):
    model.train()
    loss_class = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(nepoch):
        for i, (s_img, s_label) in enumerate(train_dataloader):
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            class_output = model(s_img)
            loss = loss_class(class_output, s_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1


def test(model, dataloader):
    model.eval()
    n_correct, n_total = 0, 0
    for img, label in iter(dataloader):
        batch_size = len(label)
        img, label = img.cuda(), label.cuda()
        with torch.no_grad():
            class_output = model(img)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    acc = n_correct.double() / n_total
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain model')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--data-dir', default='./data/', type=str, help='dir to dataset')
    parser.add_argument('--base-dir', default='./checkpoints/classifier/', type=str, help='dir to save model')

    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    (PX_all, Py_all) = get_gtsrb_data(args.data_dir)
    PX_all, Py_all = get_transformed_data(PX_all, Py_all)
    PX_all, Py_all = select_samples(PX_all, Py_all)

    num_classes = 10
    batch_size = 128
    nepoch = 10
    n_donate = 3000*num_classes

    PX, PX_donate, Py, Py_donate = train_test_split(PX_all, Py_all, test_size=n_donate, shuffle=True, stratify=Py_all, random_state=rand())
    PX_train, PX_test, Py_train, Py_test = train_test_split(PX, Py, test_size=0.1, shuffle=True, random_state=rand())

    model = ResNet18(num_classes=num_classes).cuda()

    train_dataset = MyDataset(PX_train, Py_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    base_dir = args.base_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    train_model(model, train_dataloader, nepoch)
    torch.save(model, os.path.join(base_dir, 'checkpoint.pth'))

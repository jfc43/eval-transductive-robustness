import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision import datasets

from utils.lib import *
from models.dann_model import DANNModel


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

def train_model(model, train_dataloader, test_dataloader, nepoch, save_epoch, save_dir):
    loss_class = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(nepoch):
        model.train()
        for i, (s_img, s_label) in enumerate(train_dataloader):
            s_img = s_img.cuda()
            s_label = s_label.cuda()

            class_output = model(s_img)
            loss = loss_class(class_output, s_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1

        if (epoch+1) % save_epoch == 0:
            acc = test(model, test_dataloader)
            print('Epoch: {:d}, Test Acc: {:.2f}%'.format(epoch+1, acc*100))
            torch.save(model, os.path.join(save_dir, 'checkpoint_{:d}.pth'.format(epoch+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain domain-invariant check model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base-dir', default='./checkpoints/nat_model/', type=str, help='dir to save model')

    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)

    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    
    save_dir = args.base_dir
    batch_size = 128
    
    img_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform, download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform, download=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model Setup
    model = DANNModel(mean=[0.1307], std=[0.3015]).cuda()
    train_model(model, train_dataloader, test_dataloader, nepoch=100, save_epoch=20, save_dir=save_dir)

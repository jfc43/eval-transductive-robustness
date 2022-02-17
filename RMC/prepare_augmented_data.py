import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision import datasets

from utils.lib import *
from utils.pgd_attack import *
from models.resnet import ResNet


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
    parser = argparse.ArgumentParser(description='Generate augmented training dataset and extract features')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-type', default='nat_model', 
                    choices=['nat_model', 'adv_model'], type=str, help='model type')
    parser.add_argument('--save-dir', default='./generate_data/', type=str, help='dir to save data')
    parser.add_argument('--model-dir', default='./checkpoints/', type=str, help='dir to saved model')

    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    model_type = args.model_type
    save_dir = os.path.join(args.save_dir, model_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(args.model_dir, model_type, "checkpoint.pth")
    batch_size = 128

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
    std = [x/255.0 for x in [63.0, 62.1, 66.7]]

    train_dataset = datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dataset = datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model Setup
    model = torch.load(model_path).cuda()
    model.eval()

    attacker = LinfPGDAttack(model, eps=8/255.0, nb_iter=40,
            eps_iter=1/255.0, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=True)

    augment_data = []
    augment_label = []

    for batch_x, batch_y in train_dataloader:
        augment_data.extend(batch_x.numpy())
        augment_label.extend(batch_y.numpy())

    correct = 0.0
    count = 0.0

    for j in range(4):
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            adv_batch_x = attacker.perturb(batch_x, batch_y)
            augment_data.extend(adv_batch_x.cpu().numpy())
            augment_label.extend(batch_y.cpu().numpy())

            with torch.no_grad():
                outputs = model(adv_batch_x)
            
            preds = torch.argmax(outputs, axis=1)

            correct += torch.sum(preds==batch_y)
            count += batch_x.shape[0]

    print("Adv acc: {:.2f}%".format((correct/count)*100))
    augment_data = np.array(augment_data)
    augment_label = np.array(augment_label)

    np.save(os.path.join(save_dir, "augment_data.npy"), augment_data)
    np.save(os.path.join(save_dir, "augment_label.npy"), augment_label)

    augment_data = torch.Tensor(augment_data) 
    augment_label = torch.Tensor(augment_label).long()
    augment_dataset = TensorDataset(augment_data, augment_label)
    augment_dataloader = DataLoader(augment_dataset, batch_size=batch_size, shuffle=False) 

    augment_features = []
    for batch_x, batch_y in augment_dataloader:
        batch_x = batch_x.cuda()
        with torch.no_grad():
            feature = model.get_feature(batch_x)
        augment_features.extend(feature.cpu().numpy())

    augment_features = np.array(augment_features)
    np.save(os.path.join(save_dir, "augment_feature.npy"), augment_features)

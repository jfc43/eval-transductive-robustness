import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision import datasets
import utils.torch
import utils.numpy
import models
import json
from autoattack import AutoAttack


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

def gen_adv_examples(model, test_dataloader, epsilon):

    attacker = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)

    test_adv_imgs = []
    test_adv_labels = []

    for i, (img, label) in enumerate(test_dataloader):
        img = img.cuda()
        label = label.cuda()
        adv_img = attacker.run_standard_evaluation(img, label)
        test_adv_imgs.extend(adv_img.cpu().numpy())
        test_adv_labels.extend(label.cpu().numpy())

    return np.array(test_adv_imgs), np.array(test_adv_labels)

def get_args():
    parser = argparse.ArgumentParser(description='eval robust model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config-file', type=str, required=True, help='config file')
    # args parse
    return parser.parse_args()

def main():

    args = get_args()
    
    # Set random seed
    utils.torch.set_seed(args.seed)
    
    
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    dataset = config['dataset']
    model_arch = config['model_arch']
    epsilon = config['epsilon']
    batch_size = config['batch_size']
    model_dir = config['model_dir']

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ])

    logger.info(args)

    if dataset == 'cifar10':
        N_class = 10
        resolution = (3, 32, 32)
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        transform_test = transforms.ToTensor()

        train_dataset = datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform_train)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        epsilon /= 255.

    elif dataset == 'mnist':
        N_class = 10
        resolution = (1, 28, 28)
        train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.MNIST(root='./datasets/mnist', train=False, transform=transforms.ToTensor(), download=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
  
    # Model Setup
    if model_arch == "lenet":
        model = models.FixedLeNet(N_class, resolution)
    elif model_arch == "resnet20":
        model = models.ResNet(N_class, resolution, blocks=[3, 3, 3])
    else:
        raise ValueError

    checkpoint = torch.load(os.path.join(model_dir, "model_0", "classifier.pth.tar"))
    model.load_state_dict(checkpoint['model'])
    model.cuda()

    acc = test(model, test_dataloader)
    print(f"Accuracy: {acc:.2%}")

    test_adv_imgs, test_adv_labels = gen_adv_examples(model, test_dataloader, epsilon)
    test_adv_imgs = torch.Tensor(test_adv_imgs) 
    test_adv_labels = torch.Tensor(test_adv_labels).long()
    test_adv_dataset = TensorDataset(test_adv_imgs, test_adv_labels)
    test_adv_dataloader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=False) 
    defend_acc = test(model, test_adv_dataloader)
    print(f"Defend acc against auto-attack: {defend_acc:.2%}")

if __name__ == "__main__":
    main()

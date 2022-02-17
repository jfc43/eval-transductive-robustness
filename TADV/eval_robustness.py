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
from attacks.gmsa_attack import GMSAMINLinfPGDAttack, GMSAAVGLinfPGDAttack, LinfPGDAttack
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

def gen_adv_examples(model_ensemble, model_index, test_dataloader, attack_method, epsilon, nb_iter, eps_iter):
    if attack_method=="gmsa-avg":
        attacker = GMSAAVGLinfPGDAttack(model_ensemble[:model_index+1], eps=epsilon, nb_iter=nb_iter,
                eps_iter=eps_iter, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
    elif attack_method=="gmsa-min":
        attacker = GMSAMINLinfPGDAttack(model_ensemble[:model_index+1], eps=epsilon, nb_iter=nb_iter,
                eps_iter=eps_iter, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
    elif attack_method=="pgd":
        attacker = LinfPGDAttack(model_ensemble[0], eps=epsilon, nb_iter=nb_iter,
                eps_iter=eps_iter, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
    elif attack_method=="fpa":
        attacker = LinfPGDAttack(model_ensemble[model_index], eps=epsilon, nb_iter=nb_iter,
                eps_iter=eps_iter, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
    elif attack_method=="auto-attack":
        attacker = AutoAttack(model_ensemble[0], norm='Linf', eps=epsilon, version='standard', verbose=False)
    else:
        raise KeyError("not supported attack")
    
    test_adv_imgs = []
    test_adv_labels = []

    for i, (img, label) in enumerate(test_dataloader):
        img = img.cuda()
        label = label.cuda()
        if attack_method=="auto-attack":
            adv_img = attacker.run_standard_evaluation(img, label)
        else:
            adv_img = attacker.perturb(img, label)

        test_adv_imgs.extend(adv_img.cpu().numpy())
        test_adv_labels.extend(label.cpu().numpy())

    return np.array(test_adv_imgs), np.array(test_adv_labels)

def get_args():
    parser = argparse.ArgumentParser(description='eval robust model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config-file', type=str, required=True, help='config file')
    parser.add_argument('--attack-method', default='pgd', 
                    choices=['auto-attack', 'pgd', 'fpa', 'gmsa-avg', 'gmsa-min'], type=str, help='attack method')
    
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
    eps_iter = config['eps_iter']
    nb_iter = config['nb_iter']
    batch_size = config['batch_size']
    model_dir = config['model_dir']
    ensemble_size = config['ensemble_size']
    attack_method = args.attack_method

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
        eps_iter /= 255.

    elif dataset == 'mnist':
        N_class = 10
        resolution = (1, 28, 28)
        train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.MNIST(root='./datasets/mnist', train=False, transform=transforms.ToTensor(), download=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
  
    # Model Setup
    model_ensemble = []

    for k in range(ensemble_size+1):
        if model_arch == "lenet":
            model = models.FixedLeNet(N_class, resolution)
        elif model_arch == "resnet20":
            model = models.ResNet(N_class, resolution, blocks=[3, 3, 3])
        else:
            raise ValueError

        checkpoint = torch.load(os.path.join(model_dir, f"model_{k:d}", "classifier.pth.tar"))
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model_ensemble.append(model)
    
    worst_adv_imgs = None
    worst_adv_labels = None
    worst_defend_acc = 1.1

    for model_index in range(ensemble_size):
        test_adv_imgs, test_adv_labels = gen_adv_examples(model_ensemble, model_index, test_dataloader, attack_method, epsilon, nb_iter, eps_iter)
        
        test_adv_imgs = torch.Tensor(test_adv_imgs) 
        test_adv_labels = torch.Tensor(test_adv_labels).long()

        test_adv_dataset = TensorDataset(test_adv_imgs, test_adv_labels)
        test_adv_dataloader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=False) 

        defend_acc = test(model_ensemble[model_index+1], test_adv_dataloader)
        print(f"Round {model_index}, defend acc: {defend_acc:.2%}")

        if defend_acc < worst_defend_acc:
            worst_defend_acc = defend_acc
            worst_adv_imgs = test_adv_imgs.numpy()
            worst_adv_labels = test_adv_labels.numpy()

        if attack_method=="pgd" or attack_method=="auto-attack":
            break

    test_adv_imgs = torch.Tensor(worst_adv_imgs) 
    test_adv_labels = torch.Tensor(worst_adv_labels).long()

    test_adv_dataset = TensorDataset(test_adv_imgs, test_adv_labels)
    test_adv_dataloader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=False) 

    if model_arch == "lenet":
        model = models.FixedLeNet(N_class, resolution)
    elif model_arch == "resnet20":
        model = models.ResNet(N_class, resolution, blocks=[3, 3, 3])
    else:
        raise ValueError

    checkpoint = torch.load(os.path.join(model_dir, f"model_{ensemble_size+1:d}", "classifier.pth.tar"))
    model.load_state_dict(checkpoint['model'])
    model.cuda()

    acc = test(model, test_dataloader)
    print(f"Accuracy: {acc:.2%}")
    defend_acc = test(model, test_adv_dataloader)
    print(f"Defend acc against {attack_method} attack: {defend_acc:.2%}")

if __name__ == "__main__":
    main()

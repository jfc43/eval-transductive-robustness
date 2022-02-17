import argparse
import os
import numpy as np
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision import datasets

from utils.lib import *
from utils.pgd_attack import *
from utils.rmc import *
from models.resnet import ResNet
from autoattack import AutoAttack


def get_eval_data(test_dataloader):
    adv_sequence = []

    for i, (batch_x, batch_y) in enumerate(test_dataloader):

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        adv_sequence.append((batch_x, batch_y))
        if i+1==1000:
            break

    return adv_sequence

def generate_transfer_adv_sequence(model, test_dataloader, attack_method):
    adv_sequence = []

    if attack_method == "pgd":
        attacker = LinfPGDAttack(model, eps=8/255.0, nb_iter=40,
                eps_iter=1/255.0, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
    elif attack_method == "auto-attack":
        attacker = AutoAttack(model, norm='Linf', eps=8/255.0, version='standard', verbose=False)
    else:
        raise KeyError("not supported attack")
    
    for i, (batch_x, batch_y) in enumerate(test_dataloader):

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        if attack_method == "pgd":
            adv_batch_x = attacker.perturb(batch_x, batch_y)
        elif attack_method == "auto-attack":
            adv_batch_x = attacker.run_standard_evaluation(batch_x, batch_y)

        adv_sequence.append((adv_batch_x, batch_y))
        if i+1==1000:
            break

    return adv_sequence

def generate_rmc_gmsa_adv_sequence(rmc, test_dataloader, attack_method):
    robustness = AverageMeter()
    adv_sequence = []
    
    for i, (batch_x, batch_y) in enumerate(test_dataloader):

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        saved_model = copy.deepcopy(rmc.model)
        models = [saved_model]
        
        if attack_method == "gmsa-min":
            group_attacker = GMSAMINLinfPGDAttack(models, eps=8/255.0, nb_iter=40,
                        eps_iter=1/255.0, rand_init=True, clip_min=0., clip_max=1.,
                        targeted=False, num_classes=10, elementwise_best=True)
        elif attack_method == "gmsa-avg":
            group_attacker = GMSAAVGLinfPGDAttack(models, eps=8/255.0, nb_iter=40,
                        eps_iter=1/255.0, rand_init=True, clip_min=0., clip_max=1.,
                        targeted=False, num_classes=10, elementwise_best=True)
        elif attack_method == "fpa":
            group_attacker = LinfPGDAttack(saved_model, eps=8/255.0, nb_iter=40,
                        eps_iter=1/255.0, rand_init=True, clip_min=0., clip_max=1.,
                        targeted=False, num_classes=10, elementwise_best=True)

        worst_acc = 1.0
        best_adv_batch_x = batch_x
        for _ in range(10):
            adv_batch_x = group_attacker.perturb(batch_x, batch_y)
            preds = rmc.get_pred(adv_batch_x)
            acc = torch.mean((preds==batch_y).float())
            if acc < worst_acc:
                worst_acc = acc
                best_adv_batch_x = adv_batch_x

            if attack_method == "fpa":
                group_attacker.model = copy.deepcopy(rmc.model)
            else:
                group_attacker.models.append(copy.deepcopy(rmc.model))
                
            rmc.update_model(saved_model)
            
            if worst_acc == 0:
                break
        
        adv_sequence.append((best_adv_batch_x, batch_y))
        preds = rmc.get_pred(best_adv_batch_x)
        acc = torch.mean((preds==batch_y).float())

        robustness.update(acc, batch_x.shape[0])
        print("Example {:d}, Robustness: {:.2f}%".format(i+1, robustness.avg*100))
        rmc.calibrate()

        if (i+1)==1000:
            break

    return adv_sequence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval RMC')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base-model', default='nat', 
                    choices=['nat', 'adv'], 
                    type=str, help='base model')
    parser.add_argument('--attack-method', default='gmsa-min', 
                    choices=['gmsa-avg', 'gmsa-min', 'fpa', 'pgd', 'auto-attack', 'none'], 
                    type=str, help='attack method')

    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)

    base_model = args.base_model
    attack_method = args.attack_method
    batch_size = 128

    if base_model == "nat":
        generate_data_dir = "./generate_data/nat_model/"
        model_path = "./checkpoints/nat_model/checkpoint.pth"
    elif base_model == "adv":
        generate_data_dir = "./generate_data/adv_model/"
        model_path = "./checkpoints/adv_model/checkpoint.pth"
    
    # Model Setup
    augment_data = np.load(os.path.join(generate_data_dir, "augment_data.npy"))
    augment_label = np.load(os.path.join(generate_data_dir, "augment_label.npy"))
    augment_feature = np.load(os.path.join(generate_data_dir, "augment_feature.npy"))

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
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=2)

    if attack_method in ['gmsa-avg', 'gmsa-min', 'fpa']:
        rmc = RMC(model_path, augment_data, augment_label, augment_feature)
        adv_sequence = generate_rmc_gmsa_adv_sequence(rmc, test_dataloader, attack_method)
    elif attack_method in ['pgd', 'auto-attack']:
        model = torch.load(model_path).cuda()
        adv_sequence = generate_transfer_adv_sequence(model, test_dataloader, attack_method)
    else:
        adv_sequence = get_eval_data(test_dataloader)

    rmc = RMC(model_path, augment_data, augment_label, augment_feature)
    robustness = AverageMeter()
    for i, (adv_x, adv_y) in enumerate(adv_sequence):
        preds = rmc.get_pred(adv_x)
        acc = torch.mean((preds==adv_y).float())

        robustness.update(acc, adv_x.shape[0])
        if attack_method == "none":
            print(f"Example {i+1:d}, Defender Accuracy: {robustness.avg:.2%}")
        else:
            print(f"Example {i+1:d}, Defender Robustness: {robustness.avg:.2%}")
            
        rmc.calibrate()
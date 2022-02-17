import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision import datasets

from utils.lib import *
from utils.attacks import *
from models.dann_model import DANNModel
from autoattack import AutoAttack


def sample_batch(data_iter, source):
    try:
        img, label = data_iter.next()
    except StopIteration:
        return [], [], []

    # domain labels
    batch_size = len(label)
    if source:
        domain_label = torch.zeros(batch_size).long()
    else:
        domain_label = torch.ones(batch_size).long()

    return img.cuda(), label.cuda(), domain_label.cuda()

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

def train_dann(model, dataloader_source, dataloader_target, nepoch):

    model.train()
    loss_class = torch.nn.CrossEntropyLoss().cuda()
    loss_domain = torch.nn.CrossEntropyLoss().cuda()
    len_dataloader = min(len(dataloader_source), len(dataloader_target))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(nepoch):
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader:
            # progressive
            p = float(i + epoch * len_dataloader) / nepoch / len_dataloader
            alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.1

            # source
            s_img, s_label, domain_label = sample_batch(data_source_iter, True)
            if len(s_img) == 0:
                data_source_iter = iter(dataloader_source)
                s_img, s_label, domain_label = sample_batch(data_source_iter, True)

            class_output, domain_output = model(s_img, alpha=alpha, return_domain_output=True)
            loss_s_domain = loss_domain(domain_output, domain_label)
            loss_s_label = loss_class(class_output, s_label)

            # target
            t_img, _, domain_label = sample_batch(data_target_iter, False)
            if len(t_img) == 0:
                data_target_iter = iter(dataloader_target)
                t_img, _, domain_label = sample_batch(data_target_iter, False)
                
            _, domain_output = model(t_img, alpha=alpha, return_domain_output=True)
            loss_t_domain = loss_domain(domain_output, domain_label)

            # domain-invariant loss
            loss = loss_t_domain + loss_s_domain + loss_s_label
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1

def gen_adv_examples(test_dataloader, attacker, attack_method):
    
    test_adv_imgs = []
    test_adv_labels = []

    for i, (img, label) in enumerate(test_dataloader):
        img = img.cuda()
        label = label.cuda()

        if attack_method == "Auto-Attack":
            adv_img = attacker.run_standard_evaluation(img, label)
        else:
            adv_img = attacker.perturb(img, label)

        test_adv_imgs.extend(adv_img.cpu().numpy())
        test_adv_labels.extend(label.cpu().numpy())

    return np.array(test_adv_imgs), np.array(test_adv_labels)

def eval_transfer(model, test_dataloader, attack_method):
    if attack_method=="PGD":
        attacker = LinfPGDAttack(model, eps=0.3, nb_iter=200,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=True)
    elif attack_method=="Auto-Attack":
        attacker = AutoAttack(model, norm='Linf', eps=0.3, version='standard', verbose=False)

    test_adv_imgs, test_adv_labels = gen_adv_examples(test_dataloader, attacker, attack_method)
    return test_adv_imgs, test_adv_labels

def eval_gmsa(model, train_dataloader, test_dataloader, batch_size, attack_method, num_round=10):
    
    if attack_method !="FPA":
        dann_models = [model]

    worst_adv_imgs = None
    worst_adv_labels = None
    worst_defend_acc = 1.1

    for k in range(num_round):
        if attack_method =="FPA":
            attacker = LinfPGDAttack(model, eps=0.3, nb_iter=200,
                eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
            test_adv_imgs, test_adv_labels = gen_adv_examples(test_dataloader, attacker, attack_method)
        elif attack_method=="GMSA-AVG":
            attacker = GMSAAVG(dann_models, eps=0.3, nb_iter=200,
                eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
            test_adv_imgs, test_adv_labels = gen_adv_examples(test_dataloader, attacker, attack_method)
        elif attack_method=="GMSA-MIN":
            attacker = GMSAMIN(dann_models, eps=0.3, nb_iter=200,
                eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True)
            test_adv_imgs, test_adv_labels = gen_adv_examples(test_dataloader, attacker, attack_method)

        test_adv_imgs = torch.Tensor(test_adv_imgs) 
        test_adv_labels = torch.Tensor(test_adv_labels).long()

        test_adv_dataset = TensorDataset(test_adv_imgs, test_adv_labels)
        test_adv_dataloader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=False) 
        train_adv_dataloader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=True)

        model = DANNModel(mean=[0.1307], std=[0.3015]).cuda()
        train_dann(model, train_dataloader, train_adv_dataloader, nepoch=100)
        if attack_method !="FPA":
            dann_models.append(model)
        defend_acc = test(model, test_adv_dataloader)

        if defend_acc < worst_defend_acc:
            worst_defend_acc = defend_acc
            worst_adv_imgs = test_adv_imgs.numpy()
            worst_adv_labels = test_adv_labels.numpy()

        print("Round {}, defend acc: {:.2f}%".format(k, defend_acc*100))

    return worst_adv_imgs, worst_adv_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DANN and ATRM defense')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--attack-method', default='GMSA-MIN', 
                    choices=['None', 'Auto-Attack', 'PGD', 'FPA', 'GMSA-MIN', 'GMSA-AVG'], type=str, help='attack method')

    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    batch_size = 128
    num_round = 10
    attack_method = args.attack_method
    
    img_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform, download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform, download=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model Setup
    model = torch.load("./checkpoints/nat_model/checkpoint_100.pth").cuda()
        
    acc = test(model, test_dataloader)
    print('Test Acc of the pre-trained model: {:.2f}%'.format(acc*100))

    if attack_method in ["GMSA-MIN", "GMSA-AVG", "FPA"]:
        worst_adv_imgs, worst_adv_labels = eval_gmsa(model, train_dataloader, test_dataloader, batch_size, attack_method, num_round)
    elif attack_method in ["PGD", "Auto-Attack"]:
        worst_adv_imgs, worst_adv_labels = eval_transfer(model, test_dataloader, attack_method)
    elif attack_method == "None":
        train_test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        model = DANNModel(mean=[0.1307], std=[0.3015]).cuda()
        train_dann(model, train_dataloader, train_test_dataloader, nepoch=100)
        acc = test(model, test_dataloader)
        print(f'Test Acc of DANN: {acc:.2%}')

    if attack_method != "None":
        test_adv_imgs = torch.Tensor(worst_adv_imgs) 
        test_adv_labels = torch.Tensor(worst_adv_labels).long()

        test_adv_dataset = TensorDataset(test_adv_imgs, test_adv_labels)
        test_adv_dataloader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=False) 
        train_adv_dataloader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=True)

        model = DANNModel(mean=[0.1307], std=[0.3015]).cuda()
        train_dann(model, train_dataloader, train_adv_dataloader, nepoch=100)

        defend_acc = test(model, test_adv_dataloader)
        print(f"Robustness of DANN against {attack_method} attack: {defend_acc:.2%}")

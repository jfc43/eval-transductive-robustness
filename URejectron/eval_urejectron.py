import argparse
import numpy as np
import os
import torch
import random
from collections import defaultdict
from utils.data_util import *
from utils.corruption import *
from utils.lib import *
from utils.pgd_attack import LinfPGDAttack
from utils.cw_attack import CW

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.optim as optim
from model import ResNet18

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
    

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

def predict(model, X):
    model.eval()
    test_dataset = MyDatasetNoLabel(X)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
    preds = []
    for batch_x in iter(test_dataloader):
        batch_x = batch_x.cuda()
        with torch.no_grad():
            batch_output = model(batch_x)
            batch_preds = torch.argmax(batch_output, axis=1)
            
        preds.extend(batch_preds.cpu().numpy())
        
    return np.array(preds)

def predict_proba(model, X):
    model.eval()
    test_dataset = MyDatasetNoLabel(X)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
    probas = []
    for batch_x in iter(test_dataloader):
        batch_x = batch_x.cuda()
        with torch.no_grad():
            batch_probas = torch.softmax(model(batch_x), axis=1)
            
        probas.extend(batch_probas.cpu().numpy())
        
    return np.array(probas)
    
def build_discriminator(PX_train, QX_train, batch_size, num_epochs=10):
    dist = ResNet18(num_classes=2).cuda()
    print(f"Training distinguisher on {len(PX_train):,} P vs {len(QX_train):,} Q")
    dist_train_data = np.vstack([PX_train] + [QX_train])
    dist_train_label = np.array([0 for _ in PX_train]+[1 for _ in QX_train])
    
    dist_train_dataset = MyDataset(dist_train_data, dist_train_label)
    dist_train_dataloader = DataLoader(dist_train_dataset, batch_size=batch_size, shuffle=True) 
    
    train_model(dist, dist_train_dataloader, num_epochs)
    
    return dist

def evaluation(clf, dist, PX_train, PX_test, QX_test, Qy_test, plot_max_p_rej=1.1):

    errors = (predict(clf, QX_test)!=np.array(Qy_test))
    Pdist_hat = predict_proba(dist, PX_test)[:, 1]
    Qdist_hat = predict_proba(dist, QX_test)[:, 1]
    
    Q_len = len(Qdist_hat)//2

    thresholds = np.linspace(0.0, 1, 100)
    p_rejs = [np.mean(Pdist_hat>tau) for tau in thresholds]
    thresholds = [t for t, r in zip(thresholds, p_rejs) if r<plot_max_p_rej]

    p_rejs = [np.mean(Pdist_hat>tau) for tau in thresholds]
    rejs = [np.mean(Qdist_hat>tau) for tau in thresholds]
    
    z_rejs = [np.mean(Qdist_hat[:Q_len]>tau) for tau in thresholds]
    
    normalized_errs = []
    for tau in thresholds:
        select_errors = errors[Qdist_hat<=tau]
        if len(select_errors) > 0:
            normalized_errs.append(np.mean(select_errors))
        else:
            normalized_errs.append(0.0)
    
    return rejs, z_rejs, normalized_errs

def generate_corrupted_example(X, y):
    
    test_dataset = MyDataset(X, y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    adv_X = []
    adv_y = []
    
    for batch_X, batch_y in test_dataloader:
        
        corrupted_batch_X = gen_corruction_image(batch_X, "Brightness", 1)
        adv_X.extend(corrupted_batch_X.detach().numpy())
        adv_y.extend(batch_y.numpy())
    
    return np.array(adv_X), np.array(adv_y)


def generate_cw_adv_example(classifier, X, y):
    
    test_dataset = MyDataset(X, y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    attacker = CW(classifier, c=1.0, kappa=0, steps=100, lr=0.01)
    
    adv_X = []
    adv_y = []
    
    for batch_X, batch_y in test_dataloader:
        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()
        adv_batch_X = attacker.forward(batch_X, batch_y)
        adv_X.extend(adv_batch_X.cpu().detach().numpy())
        adv_y.extend(batch_y.cpu().numpy())
    
    return np.array(adv_X), np.array(adv_y)


def generate_pgd_adv_example(classifier, X, y):
    
    test_dataset = MyDataset(X, y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    attacker = LinfPGDAttack(classifier, eps=8/255.0, nb_iter=40,
            eps_iter=1/255.0, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=num_classes, elementwise_best=True)
    
    adv_X = []
    adv_y = []
    
    for batch_X, batch_y in test_dataloader:
        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()
        adv_batch_X = attacker.perturb(batch_X, batch_y)
        adv_X.extend(adv_batch_X.cpu().detach().numpy())
        adv_y.extend(batch_y.cpu().numpy())
    
    return np.array(adv_X), np.array(adv_y)

def test_accuracy(classifier, X, y):
    test_dataset = MyDataset(X, y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    return test(classifier, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate URejectron')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--attack-type', default='pgd_attack', 
                    choices=['pgd_attack', 'cw_attack', 'corruption'], type=str, help='attack type')
    parser.add_argument('--data-dir', default='./data/', type=str, help='dir to dataset')
    parser.add_argument('--checkpoint-path', default='./checkpoints/classifier/checkpoint.pth', type=str, help='path to save model')
    parser.add_argument('--save-dir', default='./saved_figures/', type=str, help='dir to save results')

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

    classifier = ResNet18(num_classes=num_classes).cuda()

    train_dataset = MyDataset(PX_train, Py_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    classifier = torch.load(args.checkpoint_path).cuda()

    test_dataset = MyDataset(PX_test, Py_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    train_acc = test(classifier, train_dataloader)
    test_acc = test(classifier, test_dataloader)
    print("Train accuracy: {:.2f}%, Test accuracy: {:.2f}%".format(train_acc*100, test_acc*100))

    attack_type = args.attack_type
    if attack_type == 'pgd_attack' or attack_type == 'corruption':
        EX, Ey = generate_pgd_adv_example(classifier, PX_donate, Py_donate)
        adv_test_acc = test_accuracy(classifier, EX, Ey)
        print("Adv Test accuracy: {:.2f}%".format(adv_test_acc*100))
    elif attack_type == "cw_attack":
        EX, Ey = generate_cw_adv_example(classifier, PX_donate, Py_donate)
        adv_test_acc = test_accuracy(classifier, EX, Ey)
        print("Adv Test accuracy: {:.2f}%".format(adv_test_acc*100))

    if attack_type == 'corruption':
        CX, Cy = generate_corrupted_example(PX_donate, Py_donate)
        c_test_acc = test_accuracy(classifier, CX, Cy)
        print("Corruption Test accuracy: {:.2f}%".format(c_test_acc*100))
        QEX = np.vstack([CX, EX])
        QEy = np.hstack([Cy, Ey])
    else:
        QEX = np.vstack([PX_donate, EX]) 
        QEy = np.hstack([Py_donate, Ey])

    dist = build_discriminator(PX_train, QEX, batch_size)
    rejs, z_rejs, normalized_errs = evaluation(classifier, dist, PX_train, PX_test, QEX, QEy)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(4, 3))
    plt.ylim((-0.03, 1.0))

    if attack_type == 'pgd_attack':
        plt.title("URejectron under PGD Attack")
    elif attack_type == "cw_attack":
        plt.title("URejectron under CW Attack")
    elif attack_type == 'corruption':
        plt.title("URejectron under Image Corruption")
        
    plt.plot(*zip(*[(r, e) for r, e in zip(rejs, normalized_errs) if not np.isnan(e)]), label=r'Err on $\tilde{x}$')
    plt.plot(rejs, z_rejs, label=r'Rej on $z$')
    plt.xlabel(r'Rejection rate on $\tilde{x}$')  
    plt.legend()
        
    plt.savefig(os.path.join(save_dir, 'urejectron_{}.pdf'.format(attack_type)), bbox_inches = "tight")
    print("The result is saved in {}".format(os.path.join(save_dir, 'urejectron_{}.pdf'.format(attack_type))))


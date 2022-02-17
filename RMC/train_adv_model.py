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


def adjust_learning_rate(optimizer, base_lr, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = base_lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def train_adv_epoch(model, train_dataloader, optimizer):
    loss_class = torch.nn.CrossEntropyLoss().cuda()
    attacker = LinfPGDAttack(model, eps=8/255.0, nb_iter=10,
            eps_iter=2/255.0, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=True)
    
    for i, (s_img, s_label) in enumerate(train_dataloader):
        s_img = s_img.cuda()
        s_label = s_label.cuda()

        s_adv_img = attacker.perturb(s_img, s_label)
        model.train()
        adv_class_output = model(s_adv_img)
        loss = loss_class(adv_class_output, s_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain model using adversarial training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base-dir', default='./checkpoints/adv_model/', type=str, help='dir to save model')

    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)

    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    
    save_dir = args.base_dir
    batch_size = 128
    nepoch = 100
    lr = 0.1
    
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
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model Setup
    model = ResNet(means=mean, sds=std).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0001)

    for epoch in range(nepoch):
        adjust_learning_rate(optimizer, lr, epoch)
        train_adv_epoch(model, train_dataloader, optimizer)
        if (epoch+1) % 10 == 0:
            acc = test(model, test_dataloader)
            print('Epoch: {:d}, Test Acc: {:.2f}%'.format(epoch+1, acc*100))

    torch.save(model, os.path.join(save_dir, 'checkpoint.pth'))
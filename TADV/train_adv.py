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
from attacks.pgd_attack import LinfPGDAttack


def test(model, attacker, test_dataloader, N_class, max_batches, logger):
    model.eval()

    clean_losses = None
    clean_accs = None

    for b, (inputs, targets) in enumerate(test_dataloader):
        
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        clean_losses = utils.numpy.concatenate(clean_losses, utils.torch.classification_loss(outputs, targets, reduction='none').detach().cpu().numpy())
        clean_accs = utils.numpy.concatenate(clean_accs, 1 - utils.torch.classification_error(outputs, targets, reduction='none').detach().cpu().numpy())

    logger.info("clean_loss: {:.2f}, clean_acc: {:.2f}%".format(np.mean(clean_losses), np.mean(clean_accs)*100))

    adv_losses = None
    adv_accs = None
    adv_successes = None

    for b, (inputs, targets) in enumerate(test_dataloader):
        if b >= max_batches:
            break

        inputs = inputs.cuda()
        targets = targets.cuda()
        # small perturbations
        adv_inputs = attacker.perturb(inputs, targets)
        with torch.no_grad():
            logits = model(adv_inputs)
        adv_losses = utils.numpy.concatenate(adv_losses, utils.torch.classification_loss(logits, targets, reduction='none').detach().cpu().numpy())
        adv_accs = utils.numpy.concatenate(adv_accs, 1 - utils.torch.classification_error(logits, targets, reduction='none').detach().cpu().numpy())

    logger.info("adv_loss: {:.2f}, adv_acc: {:.2f}%".format(np.mean(adv_losses), np.mean(adv_accs)*100))

def lr_schedule(t, lr_max):
    if t < 100:
        return lr_max
    elif t < 105:
        return lr_max / 10.
    else:
        return lr_max / 100.

def train_robust(model, 
                train_dataloader,
                attacker, 
                optimizer, 
                fraction,
                N_class,
                epoch,
                lr_max,
                use_schedule,
                print_freq,
                logger):
    
    num_training_iter = len(train_dataloader)
    for b, (inputs, targets) in enumerate(train_dataloader):
        
        if use_schedule:
            epoch_now = epoch + (b + 1) / len(train_dataloader)
            lr = lr_schedule(epoch_now, lr_max)
            optimizer.param_groups[0].update(lr=lr)
        
        inputs = inputs.cuda()
        targets = targets.cuda()
        split = int(fraction*inputs.size(0))
        # update fraction for correct loss computation
        fraction = split / float(inputs.size(0))

        clean_inputs = inputs[:split]
        adv_inputs = inputs[split:]
        clean_targets = targets[:split]
        adv_targets = targets[split:]

        adv_examples = attacker.perturb(adv_inputs, adv_targets)

        if adv_inputs.shape[0] < inputs.shape[0]: # fraction is not 1
            inputs = torch.cat((clean_inputs, adv_examples), dim=0)
        else:
            inputs = adv_examples

        model.train()
        optimizer.zero_grad()
        logits = model(inputs)
        clean_logits = logits[:split]
        adv_logits = logits[split:]

        adv_loss = utils.torch.classification_loss(adv_logits, adv_targets)
        adv_acc = 1 - utils.torch.classification_error(adv_logits, adv_targets)

        if adv_inputs.shape[0] < inputs.shape[0]:
            clean_loss = utils.torch.classification_loss(clean_logits, clean_targets)
            # clean_error = utils.torch.classification_error(clean_logits, clean_targets)
            loss = (1 - fraction) * clean_loss + fraction * adv_loss
        else:
            clean_loss = torch.zeros(1)
            # clean_error = torch.zeros(1)
            loss = adv_loss

        loss.backward()
        optimizer.step()

        if (b+1) % print_freq == 0:
            logger.info("Progress: {:d}/{:d}, adv_loss: {:.2f}, adv_acc: {:.2f}%".format(b+1, 
                                                                                        num_training_iter, 
                                                                                        adv_loss.item(),
                                                                                        adv_acc.item()*100,
                                                                                        ))
        

def get_args():
    parser = argparse.ArgumentParser(description='train robust model with detection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config-file', type=str, required=True, help='config file')
    parser.add_argument('--output-dir', type=str, required=True, help='output dir')
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
    lr = config['lr']
    optimizer_name = config['optimizer_name']
    epsilon = config['epsilon']
    eps_iter = config['eps_iter']
    nb_iter = config['nb_iter']
    fraction = config['fraction']
    print_freq = config['print_freq']
    max_batches = config['max_batches']
    nepoch = config['nepoch']
    batch_size = config['batch_size']
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train_output.log')),
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
        use_schedule = True

    elif dataset == 'mnist':
        N_class = 10
        resolution = (1, 28, 28)
        train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataset = datasets.MNIST(root='./datasets/mnist', train=False, transform=transforms.ToTensor(), download=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        use_schedule = False
  
    # Model Setup
    if model_arch == "lenet":
        model = models.FixedLeNet(N_class, resolution)
    elif model_arch == "resnet20":
        model = models.ResNet(N_class, resolution, blocks=[3, 3, 3])
    else:
        raise ValueError

    model.cuda()
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError

    attacker = LinfPGDAttack(model,  
                            epsilon=epsilon, 
                            nb_iter=nb_iter, 
                            eps_iter=eps_iter, 
                            rand_init_name="random+zero",
                            num_rand_init=1,
                            clip_min=0.0,
                            clip_max=1.0)

    for epoch in range(nepoch):
        logger.info("Epoch: {:d}".format(epoch))

        train_robust(model, 
                    train_dataloader,
                    attacker,
                    optimizer, 
                    fraction,
                    N_class,
                    epoch,
                    lr,
                    use_schedule,
                    print_freq,
                    logger)
                        
        test(model, attacker, test_dataloader, N_class, max_batches, logger)

    torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(output_dir, 'classifier.pth.tar'))


if __name__ == "__main__":
    main()
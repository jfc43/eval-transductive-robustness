import logging

import torch
import numpy as np
import time
import copy

from robustbench.data import load_cifar10
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from dent import Dent
from conf import cfg, load_cfg_fom_args

from utils.pgd_attack import *
from utils.torch import set_seed, reorder_data_points
from autoattack import AutoAttack

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    assert cfg.CORRUPTION.DATASET == 'cifar10'

    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       'cifar10', ThreatModel.Linf).cuda()
    saved_model = copy.deepcopy(base_model)
    
    if cfg.MODEL.ADAPTATION == "dent":
        assert cfg.MODEL.EPISODIC
        dent_model = Dent(base_model, cfg.OPTIM)
    
    # only evaluate on NUM_EX data points
    set_seed(0)
    origin_x_test, origin_y_test = load_cifar10(cfg.CORRUPTION.NUM_EX, cfg.DATA_DIR)
    N = origin_x_test.shape[0]
    indices = np.random.permutation(np.arange(N))[:cfg.NUM_EX]
    x_test = origin_x_test[indices]
    y_test = origin_y_test[indices]
    bs = cfg.TEST.BATCH_SIZE

    # reorder data points
    if cfg.ATTACK.REORDER:
        print("Reorder data points.")
        x_test, y_test = reorder_data_points(x_test, y_test, 2*bs)
    
    x_test, y_test = x_test.cuda(), y_test.cuda()
    
    # calculate accuracy
    n_batches = int(np.ceil(x_test.shape[0] / bs))
    robust_flags = torch.zeros(x_test.shape[0], dtype=torch.bool)

    if cfg.ATTACK.TYPE == "PGD":
        adversary = LinfConfPGDAttack(saved_model, eps=8/255.0, nb_iter=100,
            eps_iter=1/255.0, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=True, num_rand_init=5)
    elif cfg.ATTACK.TYPE == "Auto-Attack":
        adversary = AutoAttack(saved_model, norm='Linf', eps=8./255., version='standard', verbose=False)
    else:
        raise KeyError

    t0 = time.time()
    for batch_idx in range(n_batches):
        start_idx = batch_idx * bs
        end_idx = min( (batch_idx + 1) * bs, x_test.shape[0])

        x = x_test[start_idx:end_idx, :].clone().to('cuda')
        y = y_test[start_idx:end_idx].clone().to('cuda')
        
        if cfg.ATTACK.TYPE == "PGD":
            adv_x = adversary.perturb(x, y)
        elif cfg.ATTACK.TYPE == "Auto-Attack":
            adv_x = adversary.run_standard_evaluation(x, y)

        with torch.no_grad():
            adv_output = dent_model(adv_x)
        correct_batch = y.eq(adv_output.max(dim=1)[1]).detach().cpu()
        robust_flags[start_idx:end_idx] = correct_batch

    robust_accuracy = torch.sum(robust_flags).item() / x_test.shape[0]
    print(f'Robust accuracy under {cfg.ATTACK.TYPE} attack: {robust_accuracy:.2%}')
  

if __name__ == '__main__':
    evaluate('"CIFAR-10 transfer attack Linf 8/255.')

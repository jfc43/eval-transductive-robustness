import logging

import torch
import numpy as np

from autoattack import AutoAttack
from robustbench.data import load_cifar10
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from conf import cfg, load_cfg_fom_args
from utils.torch import set_seed

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    assert cfg.CORRUPTION.DATASET == 'cifar10'

    model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       'cifar10', ThreatModel.Linf).cuda()
    
    # only evaluate on NUM_EX data points
    set_seed(0)
    origin_x_test, origin_y_test = load_cifar10(cfg.CORRUPTION.NUM_EX, cfg.DATA_DIR)
    N = origin_x_test.shape[0]
    indices = np.random.permutation(np.arange(N))[:cfg.NUM_EX]
    x_test = origin_x_test[indices]
    y_test = origin_y_test[indices]
    x_test, y_test = x_test.cuda(), y_test.cuda()
    bs = cfg.TEST.BATCH_SIZE
    
    # calculate robust accuracy
    adversary = AutoAttack(
        model, norm='Linf', eps=8./255., version='standard')
    adversary.run_standard_evaluation(
        x_test, y_test, bs=bs)

    # calculate accuracy
    n_batches = int(np.ceil(x_test.shape[0] / bs))
    flags = torch.zeros(x_test.shape[0], dtype=torch.bool)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * bs
        end_idx = min( (batch_idx + 1) * bs, x_test.shape[0])

        x = x_test[start_idx:end_idx, :].clone().to('cuda')
        y = y_test[start_idx:end_idx].clone().to('cuda')
        with torch.no_grad():
            output = model(x)
        correct_batch = y.eq(output.max(dim=1)[1]).detach().cpu()
        flags[start_idx:end_idx] = correct_batch
     
    accuracy = torch.sum(flags).item() / x_test.shape[0]
    print('Accuracy: {:.2%}'.format(accuracy))


if __name__ == '__main__':
    evaluate('"CIFAR-10 Linf 8/255.')

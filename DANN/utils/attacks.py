
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GMSAMIN:
    """
    GMSA-MIN attack

    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param num_classes: the number of classes.
    :param elementwise_best: if the attack chooses the worst adversarial examples across iterations.
    """

    def __init__(
            self, models, eps=0.1, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=False):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.models = models
        self.num_classes = num_classes
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_loss(self, x, y):
        min_loss = None
        for model in self.models:
            outputs = model(x)
            curr_loss = self.loss_func(outputs, y)
            if min_loss is None:
                min_loss = curr_loss
            else:
                cond = curr_loss.data < min_loss.data
                min_loss[cond] = curr_loss[cond]

        return min_loss

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        for model in self.models:
            model.eval()

        x = x.detach().clone()
        y = y.detach().clone()
        y = y.cuda()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            with torch.no_grad():
                loss = self.get_loss(x, y)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

        for ii in range(self.nb_iter*len(self.models)):
            adv_x = x + delta
            loss = self.get_loss(adv_x, y)

            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta
            with torch.no_grad():
                loss = self.get_loss(adv_x, y)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]
            adv_x = x + worst_perb
        else:
            adv_x = x + delta.data

        return adv_x


class GMSAAVG:
    """
    GMSA-AVG attack

    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param num_classes: the number of classes.
    :param elementwise_best: if the attack chooses the worst adversarial examples across iterations.
    """

    def __init__(
            self, models, eps=0.1, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=False):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.models = models
        self.num_classes = num_classes
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_loss(self, x, y, update=False):
        loss = 0.0
        for model in self.models:
            outputs = model(x)
            if self.targeted:
                target = ((y + torch.randint(1, self.num_classes, y.shape).cuda()) % self.num_classes).long()
                curr_loss = -self.loss_func(outputs, target)
            else:
                curr_loss = self.loss_func(outputs, y)
            
            if update:
                curr_loss.mean().backward()
            loss += curr_loss.data

        return loss

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        for model in self.models:
            model.eval()

        x = x.detach().clone()
        y = y.detach().clone()
        y = y.cuda()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            with torch.no_grad():
                loss = self.get_loss(x, y, update=False)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

        for ii in range(self.nb_iter):
            adv_x = x + delta
            loss = self.get_loss(adv_x, y, update=True)
            
            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta
            with torch.no_grad():
                loss = self.get_loss(adv_x, y, update=False)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]
            adv_x = x + worst_perb
        else:
            adv_x = x + delta.data

        return adv_x
        

class LinfPGDAttack:
    """
    PGD Attack with order=Linf

    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param num_classes: the number of classes.
    :param elementwise_best: if the attack chooses the worst adversarial examples across iterations.
    """

    def __init__(
            self, model, eps=0.1, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=False):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.model = model
        self.num_classes = num_classes
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_loss(self, x, y):
        outputs = self.model(x)

        if self.targeted:
            target = ((y + torch.randint(1, self.num_classes, y.shape).cuda()) % self.num_classes).long()
            loss = -self.loss_func(outputs, target)
        else:
            loss = self.loss_func(outputs, y)

        return loss

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone()
        y = y.detach().clone()
        y = y.cuda()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            loss = self.get_loss(x, y)

            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

        for ii in range(self.nb_iter):
            adv_x = x + delta
            loss = self.get_loss(adv_x, y)

            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta
            loss = self.get_loss(adv_x, y)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]
            adv_x = x + worst_perb
        else:
            adv_x = x + delta.data

        return adv_x

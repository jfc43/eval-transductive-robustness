from __future__ import print_function
import torch
import torch.nn as nn
import random


class LinfPGDAttack:

    def __init__(self, 
                model, 
                epsilon=0.3, 
                nb_iter=100, 
                eps_iter=0.1, 
                rand_init_name="random",
                num_rand_init=1,
                clip_min=0.0,
                clip_max=1.0):

        self.model = model
        self.epsilon = epsilon
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init_name = rand_init_name
        self.num_rand_init = num_rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def random_init(self, delta, x, random_name="random"):
        
        if random_name == 'random':
            delta.data.normal_()
            u = torch.zeros(delta.size(0)).uniform_(0, 1).cuda()
            linf_norm = u / torch.max(delta.abs().view(delta.size(0), -1), dim=1)[0]
            delta.data = self.epsilon * delta.data * linf_norm.view(delta.size(0), 1, 1, 1).data
        elif random_name == 'zero':
            delta.data.zero_()
        else:
            raise ValueError

        delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

    def get_loss(self, x, delta, y):
        adv_x = x + delta
        outputs = self.model(adv_x)
        loss = self.loss_func(outputs, y)
        return loss

    def perturb_once(self, x, y):

        delta = torch.zeros_like(x)
        loss = self.get_loss(x, delta, y)
        success_errors = loss.data.clone()
        success_perturbs = delta.data.clone()

        if self.rand_init_name == 'random+zero':
            random_name = random.choice(["random", "zero"])
            self.random_init(delta, x, random_name)
        else:
            self.random_init(delta, x, self.rand_init_name)

        delta = nn.Parameter(delta)
        delta.requires_grad_()

        for ii in range(self.nb_iter):
            loss = self.get_loss(x, delta, y)

            cond = loss.data > success_errors
            success_errors[cond] = loss.data[cond]
            success_perturbs[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.epsilon, max=self.epsilon)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        loss = self.get_loss(x, delta, y)
        cond = loss.data > success_errors
        success_errors[cond] = loss.data[cond]
        success_perturbs[cond] = delta.data[cond]

        return success_errors, success_perturbs

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_loss = None
        worst_perb = None
        for k in range(self.num_rand_init):
            curr_worst_loss, curr_worst_perb = self.perturb_once(x, y)
            if worst_loss is None:
                worst_loss = curr_worst_loss
                worst_perb = curr_worst_perb
            else:
                cond = curr_worst_loss > worst_loss
                worst_loss[cond] = curr_worst_loss[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb




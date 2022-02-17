#!/bin/bash

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --config-file ./configs/mnist_lenet_eval.json
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/mnist_lenet_eval.json --attack-method auto-attack 
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/mnist_lenet_eval.json --attack-method pgd 
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/mnist_lenet_eval.json --attack-method fpa 
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/mnist_lenet_eval.json --attack-method gmsa-avg
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/mnist_lenet_eval.json --attack-method gmsa-min

CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --config-file ./configs/cifar10_resnet_eval.json
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/cifar10_resnet_eval.json --attack-method auto-attack 
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/cifar10_resnet_eval.json --attack-method pgd 
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/cifar10_resnet_eval.json --attack-method fpa 
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/cifar10_resnet_eval.json --attack-method gmsa-avg
CUDA_VISIBLE_DEVICES=$gpu_id python eval_robustness.py --config-file ./configs/cifar10_resnet_eval.json --attack-method gmsa-min
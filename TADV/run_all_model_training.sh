#!/bin/bash

gpu_id=0

for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_adv.py --config-file ./configs/mnist_lenet.json --seed $(( 100*(i+1) )) --output-dir saved_models/MNIST/model_$i/
done

for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_adv.py --config-file ./configs/cifar10_resnet.json --seed $(( 100*(i+1) )) --output-dir saved_models/CIFAR10/model_$i/
done

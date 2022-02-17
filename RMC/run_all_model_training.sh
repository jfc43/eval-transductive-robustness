#!/bin/bash

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py
CUDA_VISIBLE_DEVICES=$gpu_id python train_adv_model.py
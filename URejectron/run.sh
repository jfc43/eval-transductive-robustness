#!/bin/bash

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py
CUDA_VISIBLE_DEVICES=$gpu_id python eval_urejectron.py --attack-type pgd_attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_urejectron.py --attack-type cw_attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_urejectron.py --attack-type corruption
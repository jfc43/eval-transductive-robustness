#!/bin/bash

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python eval.py --attack-method None 
CUDA_VISIBLE_DEVICES=$gpu_id python eval.py --attack-method Auto-Attack 
CUDA_VISIBLE_DEVICES=$gpu_id python eval.py --attack-method PGD 
CUDA_VISIBLE_DEVICES=$gpu_id python eval.py --attack-method FPA 
CUDA_VISIBLE_DEVICES=$gpu_id python eval.py --attack-method GMSA-AVG 
CUDA_VISIBLE_DEVICES=$gpu_id python eval.py --attack-method GMSA-MIN

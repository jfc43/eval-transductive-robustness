#!/bin/bash

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python prepare_augmented_data.py --model-type nat_model 

CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model nat --attack-method none
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model nat --attack-method auto-attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model nat --attack-method pgd
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model nat --attack-method fpa
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model nat --attack-method gmsa-avg
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model nat --attack-method gmsa-min


CUDA_VISIBLE_DEVICES=$gpu_id python prepare_augmented_data.py --model-type adv_model 

CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model adv --attack-method none
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model adv --attack-method auto-attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model adv --attack-method pgd
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model adv --attack-method fpa
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model adv --attack-method gmsa-avg
CUDA_VISIBLE_DEVICES=$gpu_id python eval_rmc_sequence.py --base-model adv --attack-method gmsa-min

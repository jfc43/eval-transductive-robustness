#!/bin/bash

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra
CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled
CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra
CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving
CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using
CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast
CUDA_VISIBLE_DEVICES=$gpu_id python eval_static.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE Auto-Attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE Auto-Attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE Auto-Attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE Auto-Attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE Auto-Attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE Auto-Attack
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE Auto-Attack

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE PGD
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE PGD
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE PGD
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE PGD
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE PGD
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE PGD
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE PGD

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE FPA
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE FPA
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE FPA
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE FPA
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE FPA
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE FPA
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE FPA

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE GMSA-AVG
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE GMSA-AVG
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE GMSA-AVG
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE GMSA-AVG
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE GMSA-AVG
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE GMSA-AVG
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE GMSA-AVG

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE GMSA-MIN
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE GMSA-MIN
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE GMSA-MIN
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE GMSA-MIN
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE GMSA-MIN
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE GMSA-MIN
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE GMSA-MIN

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE Auto-Attack ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE Auto-Attack ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE Auto-Attack ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE Auto-Attack ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE Auto-Attack ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE Auto-Attack ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE Auto-Attack ATTACK.REORDER True

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE PGD ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE PGD ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE PGD ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE PGD ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE PGD ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE PGD ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_transfer.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE PGD ATTACK.REORDER True

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE FPA ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE FPA ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE FPA ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE FPA ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE FPA ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE FPA ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE FPA ATTACK.REORDER True

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE GMSA-AVG ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE GMSA-AVG ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE GMSA-AVG ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE GMSA-AVG ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE GMSA-AVG ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE GMSA-AVG ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE GMSA-AVG ATTACK.REORDER True

CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra ATTACK.TYPE GMSA-MIN ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled ATTACK.TYPE GMSA-MIN ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra ATTACK.TYPE GMSA-MIN ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving ATTACK.TYPE GMSA-MIN ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using ATTACK.TYPE GMSA-MIN ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast ATTACK.TYPE GMSA-MIN ATTACK.REORDER True
CUDA_VISIBLE_DEVICES=$gpu_id python eval_dent_gmsa.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA ATTACK.TYPE GMSA-MIN ATTACK.REORDER True
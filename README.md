# Towards Evaluating the Robustness of Neural Networks Learned by Transduction
This repository is the official implementation of [Towards Evaluating the Robustness of Neural Networks Learned by Transduction](https://openreview.net/pdf?id=_5js_8uTrx1). 

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requires some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)
* [scikit-learn](https://scikit-learn.org/)
* [yacs](https://pypi.org/project/yacs/)
* [iopath](https://pypi.org/project/iopath/)
* [adamod](https://pypi.org/project/adamod/)
* [robustbench](https://github.com/RobustBench/robustbench)

## RMC Experiments
The experiments are on CIFAR-10 dataset. 

### Overview of the Code
* `train_model.py`: train a base model using standard training.
* `train_adv_model.py`: train a base model using adversarial training. 
* `prepare_augmented_data.py`: generate augmented training data and extrate their features.
* `eval_rmc_sequence.py`: evaluate RMC on a sequence of adversarial examples generated by different attacks. 

### Running Experiments
You can run the following scripts to get the results reported in the paper:

* `run_all_model_training.sh`: train all base models. 
* `run_all_evaluation.sh`: evaluate RMC on all tasks.  

## DENT Experiments
The experiments are on CIFAR-10 dataset. 

### Overview of the Code
* `dent.py`: the implementation of DENT. 
* `conf.py`: the configurations for experiments. 
* `eval_dent_gmsa.py`: evaluate DENT under FPA, GMSA-AVG and GMSA-MIN attacks. 
* `eval_dent_transfer.py`: evaluate DENT under Auto-Attack and PGD attack in the transfer attack setting.  
* `eval_dent.py`: evaluate DENT under Auto-Attack (DENT-AA). 
* `eval_static.py`: evaluate the performance of static models. 

### Running Experiments
You can run the following scripts to get the results reported in the paper:

* `run_all_evaluation.sh`: evaluate DENT with different static base models on all tasks.  

## DANN Experiments
The experiments are on MNIST dataset. 

### Overview of the Code
* `train_model.py`: train a model using standard training.
* `eval.py`: evaluate DANN defenses under different attacks. 

### Running Experiments
You can run the following scripts to get the results reported in the paper:

* `run_model_training.sh`: train a standard base model. 
* `run_all_evaluation.sh`: evaluate DANN on all tasks.  

## TADV Experiments
The experiments are on MNIST and CIFAR-10 dataset. 

### Overview of the Code
* `eval_static.py`: evaluate the performance of static models. 
* `eval_robustness.py`: evaluate TADV under different attacks. 

### Running Experiments
You can run the following scripts to get the results reported in the paper:

* `run_all_model_training.sh`: train all models needed. 
* `run_all_evaluation.sh`: evaluate TADV on all tasks.  

## URejectron Experiments
### Downloading Datasets
* [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html): we provide a script `prepare_data.sh` to download it.

### Overview of the Code
* `train_model.py`: train a model for classification.
* `eval_urejectron.py`: evaluate URejectron under different attacks.  

### Running Experiments
You can run the following scripts to get the results reported in the paper: 

* `prepare_data.sh`: prepare data. 
* `run.sh`: train a model for classification and then evaluate URejectron with this model under PGD attack, CW attack and image corruption. 

## Acknowledgements
Part of this code is inspired by [Robustness](https://github.com/hendrycks/robustness), [Adversarial-Attacks-PyTorch](https://github.com/Harry24k/adversarial-attacks-pytorch), [URejectron](https://proceedings.neurips.cc/paper/2020/file/b6c8cf4c587f2ead0c08955ee6e2502b-Supplemental.zip), [DENT](https://github.com/DequanWang/dent) and [Runtime-Masking-and-Cleansing](https://github.com/nthu-datalab/Runtime-Masking-and-Cleansing). 

## Citation 
Please cite our work if you use the codebase: 
```
@inproceedings{
chen2022towards,
title={Towards Evaluating the Robustness of Neural Networks Learned by Transduction},
author={Jiefeng Chen and Xi Wu and Yang Guo and Yingyu Liang and Somesh Jha},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=_5js_8uTrx1}
}
```

## License
Please refer to the [LICENSE](LICENSE).

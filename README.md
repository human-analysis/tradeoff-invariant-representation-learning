# On Characterizing the Trade-off in Invariant Representation Learning

By Bashir Sadeghi, Sepehr Dehdashtian, and Vishnu Naresh Boddeti

## Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Setup](#setup)
0. [Requirements](#requirements)
0. [Commands to Reproduce Results in Paper](#commands-to-reproduce-results-in-paper)

### Introduction

This code archive includes the Python implementation to Characterizing the Trade-off in Invariant Representation Learning

### Citation

If you think this work is useful to your research, please cite:

    @inproceedings{sadeghi2022tradeoff,
      title={On Characterizing the Trade-off in Invariant Representation Learning},
      author={Sadeghi, Bashir and Dehdashtian, Sepehr and Boddeti, Vishnu},
      booktitle={Transactions on Machine Learning Research},
      year={2022}
    }

**Link** to the paper: https://hal.cse.msu.edu/papers/characterizing-trade-off-invariant-representation-learning/

### Setup
First, you need to download PyTorchNet by calling the following command:
> git clone --recursive https://github.com/human-analysis/kernel-adversarial-representation-learning.git

### Requirements

1. Require `python 3.8.5`
2. Require `pyTorch 1.10.0`
3. Require `pytorch-lightning 1.4.5`
4. Check `requirements.yml` for detailed dependencies. To install the dependencies completely in a new conda environment, you can run `conda env create -f requirements.yml` command. 

### Commands to Reproduce Results in Paper
#### Toy Gaussian Dataset for our K-$\mathcal{T}_{\text{Opt}}$
~~~~
$ python main.py --args exps/gaussian/args_gaussian_kernel_classification.txt
~~~~
To see track training procedure in Tensorboard:
~~~~
$ tensorboard --logdir results/KTOPT-Classification
~~~~

#### Toy Gaussian Dataset for SARL
~~~~
$ tensorboard --logdir results/Linear-Classification
$ python main.py --args exps/gaussian/args_gaussian_linear_classification.txt
~~~~

#### Toy Gaussian Dataset for ARL
~~~~
$ tensorboard --logdir results/ARL-Classification
$ python main.py --args exps/gaussian/args_gaussian_arl_classification.txt
~~~~

#### Toy Gaussian Dataset for HSIC-IRepL
~~~~
$ tensorboard --logdir results/HSIC-Classification
$ python main.py --args exps/gaussian/args_gaussian_hsic_classification.txt
~~~~

# Boosting Adversarial Robustness of Reinforcement Learning via Vector Quantization

This is Pytorch implementation of **VQ-RL** 

## Installation

This code is tested with Python 3.7 on Ubuntu 20.04, CUDA 11.1.

To install requirements:
```
pip install -r requirements.txt
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```
Go to directory of `d3rlpy` source code and install:
```
cd d3rlpy
pip install -e .
```


## Training code

Run the bash script from the directory `d3rlpy`:
```
./scripts/train/walker2d/run_vq_walked2d.sh
```

## Evaluation code

Modify the checkpoint path in the evaluation bash script in `scripts/evals`. For example, to evaluate in `Walker2d-v2`, run: 
```
./scripts/evals/walker2d/eval_attack_walker_sac_vq.sh
```

## Pre-trained Models
The pretrained models for `Walker2d-v2` are attached in `checkpoints.zip`. You can unzip and copy it into `d3rlpy_logs/online/walker`.
We also provide scripts to evaluate methods: `SAC, SAC-SA, SAC-VQ, SAC-SA-VQ`.


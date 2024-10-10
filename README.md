# VQ-RL: Vector Quantization for robust RL

This is the official Pytorch implementation for the paper:

**Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantization**

[[Paper]](https://arxiv.org/pdf/2410.03376)

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
The pretrained models for `Walker2d-v2` are attached in `checkpoints.7z`. You can unzip and copy it into `d3rlpy_logs/online/walker`.
We also provide scripts to evaluate methods: `SAC, SAC-SA, SAC-VQ, SAC-SA-VQ`.

## Citation
If you use this repo in your research, please consider citing the paper as follows
```
@inproceedings{tung2024vqrl,
  title={Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantizations},
  author={Tung M. Luu and Thanh Nguyen and Tee Joshua Tian Jin and Sungwoon Kim and Chang D. Yoo},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
}
```

## Acknowledgements
This code is based on top of [d3rlpy](https://github.com/takuseno/d3rlpy).

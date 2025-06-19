<div><h2>[IROS'24] Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantization</h2></div>
<br>

**Tung M. Luu<sup>1</sup>, Thanh Nguyen<sup>1</sup>, Tee Joshua Tian Jin<sup>1</sup>, Sungwoon Kim<sup>2</sup>, and Chang D. Yoo<sup>1</sup>**
<br>
<sup>1</sup>KAIST, South Korea, <sup>2</sup>Korea University, South Korea
<br>
[[arXiv]](https://arxiv.org/abs/2410.03376) [[Paper]](https://ieeexplore.ieee.org/document/10802066) 

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
@inproceedings{luu2024mitigating,
  title={Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantization},
  author={Luu, Tung M and Nguyen, Thanh and Jin, Tee Joshua Tian and Kim, Sungwoon and Yoo, Chang D},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgements
- This work was partly supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea
government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to
Real Environments) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00184, Development and Study
of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).

- This code is based on top of [d3rlpy](https://github.com/takuseno/d3rlpy).

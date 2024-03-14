#!/bin/bash

GPU=0

ENV_NAME="walker2d-expert-v2"
ENV_NAME_DIR="walker2d"


CKPT_STEPS="model_3000000.pt"
N_SEEDS_WANT_TO_TEST=5
N_EVAL_EPISODES=50



ATTACK_TYPE_LIST=(
  "random"
  "actor_mad"
  "minQ"
)

ATTACK_EPSILON_LIST=(
  0.05
  0.1
  0.15
  0.2
  0.25
  0.3
)

CKPT="d3rlpy_logs/online/${ENV_NAME_DIR}/SAC"
for ATTACK_TYPE in ${ATTACK_TYPE_LIST[*]}; do
for ATTACK_EPSILON in ${ATTACK_EPSILON_LIST[*]}; do
  CUDA_VISIBLE_DEVICES=${GPU} python -W ignore evaluate_online_learned_policy_under_attacks.py --dataset ${ENV_NAME} --gpu 0 --ckpt ${CKPT} \
    --attack_type_list ${ATTACK_TYPE} --n_seeds_want_to_test ${N_SEEDS_WANT_TO_TEST} \
    --attack_epsilon ${ATTACK_EPSILON} --ckpt_steps ${CKPT_STEPS} --disable_clean --no_clip \
    --n_eval_episodes ${N_EVAL_EPISODES} --standardization
done
done

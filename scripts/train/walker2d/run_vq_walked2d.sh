#!/bin/bash
ROOT_DIR="d3rlpy_logs"

GPU=0

ENV_NAME="walker2d-expert-v2"
ENV_NAME_DIR="walker2d"

N_STEPS=3000000

N_EMBEDDINGS=8
CODEBOOK_UPDATE_TYPE="sgd"


RL_ALGO='SAC'
EXP_NAME="${RL_ALGO}_VQ_K${N_EMBEDDINGS}"
LOG_DIR="${ROOT_DIR}/online/${ENV_NAME_DIR}/${EXP_NAME}"

SEEDS=(1)
for SEED in ${SEEDS[*]}; do
  CUDA_VISIBLE_DEVICES=${GPU} python run_rl_online.py --dataset ${ENV_NAME} --gpu 0 --logdir ${LOG_DIR} --exp ${EXP_NAME} \
  --algo ${RL_ALGO} --n_steps ${N_STEPS} --standardization --no_replacement \
  --use_vq_in --n_embeddings ${N_EMBEDDINGS} \
  --codebook_update_type ${CODEBOOK_UPDATE_TYPE} --autoscale_vq_loss \
  --seed ${SEED}
done



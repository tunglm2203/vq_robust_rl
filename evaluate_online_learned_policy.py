import argparse
import gym
import time
import copy
import os
import json

import d3rlpy

from torch import multiprocessing as mp

import pandas as pd
import numpy as np

import d4rl

from rliable import metrics

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

IQM = lambda x: metrics.aggregate_iqm(x) # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0) # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)

from d3rlpy.preprocessing.scalers import StandardScaler
from d3rlpy.adversarial_training.utility import make_checkpoint_list, copy_file, EvalLogger, get_stats_from_ckpt
from d3rlpy.adversarial_training.eval_utility import (
    ENV_SEED,
    eval_clean_env,
    eval_env_under_attack,
    eval_multiprocess_wrapper,
)

ENV_NAME_MAPPING = {
    'walker2d-expert-v2': 'walker2d',
    'hopper-expert-v2': 'hopper',
    'halfcheetah-expert-v2': 'cheetah',
    'Ant-v4': 'ant',
    'InvertedPendulum-v4': 'pendulum',
    'InvertedDoublePendulum-v4': 'doublependulum',
    'Reacher-v4': 'reacher',
    'Swimmer-v4': 'swimmer',
}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n_eval_episodes', type=int, default=50)

SUPPORTED_ATTACKS = ['none', 'random']

parser.add_argument('--attack_type', type=str, default='random', choices=SUPPORTED_ATTACKS)
parser.add_argument('--attack_type_list', type=str, default=['random'], nargs='+')
parser.add_argument('--attack_epsilon', type=float, default=0.05)
parser.add_argument('--attack_iteration', type=int, default=0)

parser.add_argument('--no_clip', action='store_true', default=False)
parser.add_argument('--no_assert', action='store_true', default=False)

SUPPORTED_OPTIMS = ['pgd']
parser.add_argument('--optimizer', type=str, default='pgd', choices=SUPPORTED_OPTIMS)

parser.add_argument('--ckpt', type=str, default='.')
parser.add_argument('--n_seeds_want_to_test', type=int, default=1)
parser.add_argument('--ckpt_steps', type=str, default="model_500000.pt")

# Scope for Vector Quantization representation
parser.add_argument('--use_vq_in', action='store_true', default=False)
parser.add_argument('--codebook_update_type', type=str, default="ema", choices=["ema", "sgd"])
parser.add_argument('--n_embeddings', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=1)

parser.add_argument('--disable_clean', action='store_true')
parser.add_argument('--mp', action='store_true')
parser.add_argument('--n_processes', type=int, default=5)

parser.add_argument('--eval_logdir', type=str, default='eval_results')
parser.add_argument('--standardization', action='store_true', default=False)
args = parser.parse_args()


"""" Pre-defined constant for evaluation """
ATTACK_ITERATION=dict(
    none=1,
    random=1,
    minQ=10,
    minQ_rand=1,
    actor_mad=10,
    actor_mad_rand=1,
    sarsa=10,
)


def eval_func(algo, env, writer, attack_type, attack_epsilon, params):
    multiprocessing = params.mp
    _args = copy.deepcopy(params)
    _args.attack_type = attack_type
    _args.attack_epsilon = attack_epsilon

    _args.attack_iteration = ATTACK_ITERATION[attack_type]

    unorm_score, norm_score, unorm_score_attack, norm_score_attack = 0, 0, 0, 0
    if multiprocessing:
        print("[INFO] Multiple-processing evaluating...")
        env_list = []
        env_list.append(env)
        for i in range(_args.n_processes - 1):
            _env = gym.make(_args.dataset)
            _env.seed(ENV_SEED)
            env_list.append(_env)
        if not _args.disable_clean:
            unorm_score = eval_multiprocess_wrapper(algo, eval_clean_env, env_list, _args)
        # unorm_score_attack = eval_multiprocess_wrapper(algo, eval_env_under_attack, env_list, _args)

        del env_list

    else:
        print("[INFO] Normally evaluating...")
        # func_args = (0, algo, env, _args.seed, _args)  # algo, env, start_seed, args
        func_args = (0, algo, env, ENV_SEED, _args)  # algo, env, start_seed, args

        if not _args.disable_clean:
            unorm_score = eval_clean_env(func_args)["unorm_score"]
        # unorm_score_attack = eval_env_under_attack(func_args)


    if not _args.disable_clean:
        if env.env.spec.id in d4rl.infos.DATASET_URLS.keys():
            norm_score = env.env.wrapped_env.get_normalized_score(unorm_score) * 100
        else:
            norm_score = unorm_score
        writer.log(attack_type="clean", attack_epsilon=attack_epsilon,
                   attack_iteration=_args.attack_iteration,
                   unorm_score=unorm_score, norm_score=norm_score)
    # norm_score_attack = env.env.wrapped_env.get_normalized_score(unorm_score_attack) * 100

    # writer.log(attack_type=attack_type, attack_epsilon=attack_epsilon,
    #            attack_iteration=_args.attack_iteration,
    #            unorm_score=unorm_score_attack, norm_score=norm_score_attack)

    print("***** Env: %s - method: %s *****" % (_args.dataset, _args.ckpt.split('/')[-3]))
    if unorm_score is not None:
        print("Clean env: unorm = %.3f, norm = %.2f" % (unorm_score, norm_score))
    # print("Noise env: unorm = %.3f, norm = %.2f" % (unorm_score_attack, norm_score_attack))
    return unorm_score, norm_score, unorm_score_attack, norm_score_attack


def main(args):
    args.eval_logdir = f"./evaluation/evaluation_online/clean/{ENV_NAME_MAPPING[args.dataset]}"
    if not os.path.exists(args.eval_logdir):
        os.makedirs(args.eval_logdir)

    print("[INFO] Logging evalutation into: %s" % (args.eval_logdir))

    env = gym.make(args.dataset)

    d3rlpy.seed(args.seed)
    env.seed(ENV_SEED)

    if args.standardization:
        scaler = StandardScaler(mean=np.zeros(env.observation_space.shape), std=np.ones(env.observation_space.shape))
    else:
        scaler = None

    #################### Initialize algorithm ####################
    algorithm = d3rlpy.algos.SAC(use_gpu=args.gpu,
                                 scaler=scaler,
                                 env_name=args.dataset,
                                 use_vq_in=args.use_vq_in,
                                 codebook_update_type=args.codebook_update_type,
                                 number_embeddings=args.n_embeddings,
                                 embedding_dim=args.embedding_dim,
                                 )

    algorithm.build_with_env(env)  # Create policy/critic for env, must be performed after fitting scaler

    list_checkpoints = make_checkpoint_list(args.ckpt, args.n_seeds_want_to_test, args.ckpt_steps)

    print("[INFO] Evaluating %d checkpoint(s)\n" % (args.n_seeds_want_to_test))


    #################### Setup logger ####################
    # Initialize writer for first checkpoint, and append next checkpoints
    writer = EvalLogger(ckpt=list_checkpoints[0], eval_logdir=args.eval_logdir, prefix='eval_v1', eval_args=args)

    # Structure: NxRxC = N attack's types x C seeds
    N = 1
    C = args.n_seeds_want_to_test

    norm_score_clean = np.zeros((1, C))
    unorm_score_clean = np.zeros((1, C))
    norm_score_attack = np.zeros((N, C))
    unorm_score_attack = np.zeros((N, C))


    #################### Start evaluating ####################
    n_seeds = args.n_seeds_want_to_test if len(list_checkpoints) > args.n_seeds_want_to_test else len(list_checkpoints)
    for c, checkpoint in enumerate(list_checkpoints[:n_seeds]):
        if c > 0:
            # If only have 1 seed, do not write anything
            writer.init_info_from_ckpt(checkpoint)
            writer.write_header()

        algorithm.load_model(checkpoint)

        # Override the moving stats if any
        mean_ckpt, std_ckpt = get_stats_from_ckpt(checkpoint)
        if mean_ckpt is not None and std_ckpt is not None and algorithm._impl.scaler is not None:
            algorithm._impl.scaler._mean = mean_ckpt
            algorithm._impl.scaler._std = std_ckpt
            print("Updated statistical from checkpoint.")

        args.ckpt = checkpoint
        print("===> Eval checkpoint: %s" % (checkpoint))

        start = time.time()
        for n, attack_type in enumerate(args.attack_type_list):
            d3rlpy.seed(args.seed)
            args.disable_clean = not (n == 0)
            _unorm_score, _norm_score, _unorm_score_attack, _norm_score_attack = eval_func(algorithm, env, writer, attack_type, args.attack_epsilon, args)

            if not args.disable_clean:
                norm_score_clean[n, c] = _norm_score
                unorm_score_clean[n, c] = _unorm_score
            norm_score_attack[n, c] = _norm_score_attack
            unorm_score_attack[n, c] = _unorm_score_attack
        print("\n<=== Evaluation time for seed %d: %.3f (s)\n" % (c + 1, time.time() - start))


    #################### Log results into text, pkl, json file ####################
    writer.print("\n\n====================== Summary ======================\n")
    writer.print("Average clean: mean=%.2f, std=%.2f, median=%.2f, iqm=%.2f, og=%.2f (%d seeds)\n" %
                 (MEAN([norm_score_clean[0]]), np.std(norm_score_clean, axis=1).squeeze(),
                  MEDIAN([norm_score_clean[0]]), IQM([norm_score_clean[0]]), OG([norm_score_clean[0]]),
                  n_seeds))

    # Prepare data to save into .pkl and.json files
    columns = ["mean", "std", "median", "iqm", "og", "n_seeds"]
    data = [[MEAN([norm_score_clean[0]]), np.std(norm_score_clean, axis=1).squeeze(),
             MEDIAN([norm_score_clean[0]]), IQM([norm_score_clean[0]]), OG([norm_score_clean[0]]),
             n_seeds]]

    score_dict = {
        "env_name": args.dataset,
        "n_seeds": n_seeds,
        "clean": norm_score_clean[0].tolist(),
        "clean_unorm": unorm_score_clean[0].tolist()
    }

    summary = pd.DataFrame(data, columns=columns, index=["clean"])
    # for n in range(N):
    #     writer.print(
    #         "Attack: %15s [eps=%.4f]: mean=%.2f, std=%.2f, median=%.2f, iqm=%.2f, og=%.2f (%d seeds)\n" %
    #         (args.attack_type_list[n], args.attack_epsilon,
    #          MEAN([norm_score_attack[n]]), np.std(norm_score_attack, axis=1)[n],
    #          MEDIAN([norm_score_attack[n]]), IQM([norm_score_attack[n]]), OG([norm_score_attack[n]]),
    #          n_seeds)
    #     )
    #     data = [[
    #         MEAN([norm_score_attack[n]]), np.std(norm_score_attack, axis=1)[n],
    #         MEDIAN([norm_score_attack[n]]), IQM([norm_score_attack[n]]), OG([norm_score_attack[n]]),
    #         n_seeds
    #     ]]
    #
    #     index = ["%15s-[eps=%.4f]" % (args.attack_type_list[n], args.attack_epsilon)]
    #     _summary = pd.DataFrame(data, columns=columns, index=index)
    #     summary = summary.append(_summary)
    #
    #     score_dict.update({
    #         str(args.attack_type_list[n]) + '-' + str(args.attack_epsilon): norm_score_attack[n].tolist()
    #     })

    writer.close()

    pickle_filename = writer.logfile[:-3] + 'pkl'
    pickle_filename_no_date = writer.logfile_no_date[:-3] + 'pkl'
    summary.to_pickle(pickle_filename)
    json_filename = writer.logfile[:-3] + 'json'

    with open(json_filename, 'w') as fp:
        json.dump(score_dict, fp, sort_keys=True)

    # Always maintain latest files
    copy_file(src=writer.logfile, des=writer.logfile_no_date[:-18] + 'latest.txt')
    copy_file(src=pickle_filename, des=pickle_filename_no_date[:-18] + 'latest.pkl')
    copy_file(src=json_filename, des=writer.logfile_no_date[:-18] + 'latest.json')


if __name__ == '__main__':
    if args.mp:
        mp.set_start_method("spawn")
    main(args)

import os
import json
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import multiprocessing as mp

import numpy as np
from .utility import tensor
from .attackers import *

ENV_SEED = 2023  # Global env seed for evaluation


def make_sure_type_is_float32(x):
    assert isinstance(x, np.ndarray)
    x = x.astype(np.float32) if x.dtype == np.float64 else x
    assert x.dtype == np.float32
    return x

"""
##### Functions used to evaluate
"""
def eval_clean_env(params):
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    episode_rewards = []
    for i in tqdm(range(n_trials), disable=(rank != 0)):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        state = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            state = make_sure_type_is_float32(state)
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    # unorm_score = float(np.mean(episode_rewards))
    unorm_score = {
        "unorm_score": float(np.mean(episode_rewards)),
        "diff_outs": -1
    }
    return unorm_score


def eval_env_under_attack(params):
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    # Set seed
    # torch.manual_seed(params.seed)
    # np.random.seed(params.seed)

    attack_type = params.attack_type
    attack_epsilon = params.attack_epsilon
    attack_stepsize = attack_epsilon / params.attack_iteration
    if rank == 0:
        print("[INFO] Using %s attack: eps=%f, n_iters=%d, sz=%f" %
              (params.attack_type.upper(), attack_epsilon, params.attack_iteration, attack_stepsize))

    def perturb(state, type, attack_epsilon=None, attack_iteration=None, attack_stepsize=None,
               optimizer='pgd', clip=True, use_assert=True):
        """" NOTE: This state is taken directly from environment, so it is un-normalized, when we
        return the perturbed state, it must be un-normalized
        """""
        state_tensor = tensor(state, algo._impl.device)

        # Important: inside the attack functions, the state is assumed already normalized
        state_tensor = algo.scaler.transform(state_tensor)

        if type in ['random']:
            perturb_state = random_attack(
                state_tensor, attack_epsilon,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm, clip=clip, use_assert=use_assert
            )

        elif type in ['minQ', 'sarsa']:
            perturb_state = critic_normal_attack(
                state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm,
                optimizer=optimizer, clip=clip, use_assert=use_assert
            )

        elif type in ['actor_mad']:
            perturb_state = actor_state_attack(
                state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm,
                optimizer=optimizer, clip=clip, use_assert=use_assert
            )

        else:
            raise NotImplementedError

        # De-normalize state for return in original scale
        perturb_state = algo.scaler.reverse_transform(perturb_state)
        return perturb_state.squeeze().cpu().numpy()

    episode_rewards = []
    for i in tqdm(range(n_trials), disable=(rank != 0), desc="{} attack".format(attack_type.upper())):
        if start_seed is None:
            env.seed(i)
            torch.manual_seed(i)
            np.random.seed(i)
        else:
            env.seed(start_seed + i)
            torch.manual_seed(start_seed + i)
            np.random.seed(start_seed + i)
        state = env.reset()

        episode_reward = 0.0

        while True:
            # take action
            state = make_sure_type_is_float32(state)
            state = perturb(
                state,
                attack_type, attack_epsilon, params.attack_iteration, attack_stepsize,
                optimizer=params.optimizer, clip=not params.no_clip, use_assert=not params.no_assert
            )
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)

            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    unorm_score = float(np.mean(episode_rewards))
    return unorm_score


def eval_env_under_attack_v1(params):
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    # Set seed
    # torch.manual_seed(params.seed)
    # np.random.seed(params.seed)

    attack_type = params.attack_type
    attack_epsilon = params.attack_epsilon
    attack_stepsize = attack_epsilon / params.attack_iteration
    if rank == 0:
        print("[INFO] Using %s attack: eps=%f, n_iters=%d, sz=%f" %
              (params.attack_type.upper(), attack_epsilon, params.attack_iteration, attack_stepsize))

    def perturb(state, type, attack_epsilon=None, attack_iteration=None, attack_stepsize=None,
               optimizer='pgd', clip=True, use_assert=True):
        """" NOTE: This state is taken directly from environment, so it is un-normalized, when we
        return the perturbed state, it must be un-normalized
        """""
        state_tensor = tensor(state, algo._impl.device)

        # Important: inside the attack functions, the state is assumed already normalized
        state_tensor = algo.scaler.transform(state_tensor)

        if type in ['random']:
            perturb_state = random_attack(
                state_tensor, attack_epsilon,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm, clip=clip, use_assert=use_assert
            )

        elif type in ['minQ', 'sarsa']:
            perturb_state = critic_normal_attack(
                state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm,
                optimizer=optimizer, clip=clip, use_assert=use_assert
            )

        elif type in ['actor_mad']:
            # perturb_state = actor_state_attack(
            #     state_tensor, algo._impl._policy, algo._impl._q_func,
            #     attack_epsilon, attack_iteration, attack_stepsize,
            #     algo._impl._obs_min_norm, algo._impl._obs_max_norm,
            #     optimizer=optimizer, clip=clip, use_assert=use_assert
            # )
            perturb_state = actor_state_attack_mean(
                state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm,
                optimizer=optimizer, clip=clip, use_assert=use_assert
            )

        else:
            raise NotImplementedError

        # De-normalize state for return in original scale
        perturb_state = algo.scaler.reverse_transform(perturb_state)
        return perturb_state.squeeze().cpu().numpy()

    episode_rewards = []
    diff_outs = []
    for i in tqdm(range(n_trials), disable=(rank != 0), desc="{} attack".format(attack_type.upper())):
        if start_seed is None:
            env.seed(i)
            torch.manual_seed(i)
            np.random.seed(i)
        else:
            env.seed(start_seed + i)
            torch.manual_seed(start_seed + i)
            np.random.seed(start_seed + i)
        state = env.reset()

        episode_reward = 0.0

        diff_outs_per_episode = []
        while True:
            # take action
            clean_state = make_sure_type_is_float32(state)
            state = perturb(
                clean_state,
                attack_type, attack_epsilon, params.attack_iteration, attack_stepsize,
                optimizer=params.optimizer, clip=not params.no_clip, use_assert=not params.no_assert
            )
            clean_action = algo.predict([clean_state])[0]
            action = algo.predict([state])[0]

            diff = np.linalg.norm(clean_action - action, ord=2)
            diff_outs_per_episode.append(diff)

            state, reward, done, _ = env.step(action)

            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)
        diff_outs.append(np.mean(diff_outs_per_episode))

    # print("diff output: %.4f" % (float(np.mean(diff_outs))))
    # unorm_score = float(np.mean(episode_rewards))
    unorm_score={
        "unorm_score": float(np.mean(episode_rewards)),
        "diff_outs": float(np.mean(diff_outs))
    }
    return unorm_score

def eval_multiprocess_wrapper(algo, func, env_list, params):
    n_trials_per_each = int(params.n_eval_episodes / params.n_processes)
    n_trials_for_last = n_trials_per_each if params.n_eval_episodes % params.n_processes == 0 else \
        n_trials_per_each + params.n_eval_episodes % params.n_processes

    args_list = []
    for i in range(params.n_processes):
        params_tmp = copy.deepcopy(params)

        if i == params_tmp.n_processes - 1:  # last iteration
            params_tmp.n_eval_episodes = n_trials_for_last
        else:
            params_tmp.n_eval_episodes = n_trials_per_each

        start_seed = ENV_SEED + n_trials_per_each * i
        args_list.append((i, algo, env_list[i], start_seed, params_tmp))

    with mp.Pool(params.n_processes) as pool:
        unorm_score_dict = pool.map(func, args_list)

    unorm_score = 0.0
    diff_outs = 0.0
    for i in range(params.n_processes):
        unorm_score += unorm_score_dict[i]["unorm_score"]
        diff_outs += unorm_score_dict[i]["diff_outs"]
    unorm_score = unorm_score / params.n_processes
    diff_outs = diff_outs / params.n_processes

    print("diff output: %.4f" % (diff_outs))

    return unorm_score


def train_sarsa(algo, env, ckpt, buffer=None, n_sarsa_steps=150000, n_warmups=100000):

    # ckpt is in format: /path/to/model_500000.pt
    logdir_sarsa = os.path.join(ckpt[:ckpt.rfind('/')], 'sarsa_model')
    model_path = os.path.join(logdir_sarsa, 'sarsa_ntrains{}_warmup{}.pt'.format(n_sarsa_steps, n_warmups))


    # algo = make_bound_for_network(algo)

    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    # We need to re-initialize the critic, not using the old one (following SA-DDPG)
    # algo._impl._q_func.reset_weight()
    # algo._impl._targ_q_func.reset_weight()

    algo._impl._q_func.apply(weight_reset)
    algo._impl._targ_q_func.apply(weight_reset)

    if not os.path.exists(logdir_sarsa):
        os.mkdir(logdir_sarsa)
    if os.path.exists(model_path):
        print('Found pretrained SARSA: ', model_path)
        print('Loading pretrained SARSA... ')
        algo.load_model(model_path)
    else:
        print('Not found pretrained SARSA: ', model_path)
        algo.fit_sarsa(env, buffer, n_sarsa_steps, n_warmups)
        algo.save_model(model_path)
        print('Loading pretrained SARSA... ')
        algo.load_model(model_path)

    return algo

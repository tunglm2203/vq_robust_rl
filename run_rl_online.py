import os
import argparse
import gym
import glob
import h5py
import copy

import numpy as np

import d3rlpy
from d3rlpy.online.explorers import NormalNoise
from d3rlpy.preprocessing.scalers import StandardScaler
from d3rlpy.adversarial_training.utility import set_name_wandb_project


ENV_NAME_MAPPING = {
    'walker2d-random-v2': 'w2d2-r',
    'walker2d-medium-v2': 'w2d2-m',
    'walker2d-medium-replay-v2': 'w2d2-m-re',
    'walker2d-medium-expert-v2': 'w2d2-m-e',
    'walker2d-expert-v2': 'w2d2-e',
    'hopper-random-v2': 'hop2-r',
    'hopper-medium-v2': 'hop2-m',
    'hopper-medium-replay-v2': 'hop2-m-re',
    'hopper-medium-expert-v2': 'hop2-m-e',
    'hopper-expert-v2': 'hop2-e',
    'halfcheetah-random-v2': 'che2-r',
    'halfcheetah-medium-v2': 'che2-m',
    'halfcheetah-medium-replay-v2': 'che2-m-re',
    'halfcheetah-medium-expert-v2': 'che2-m-e',
    'halfcheetah-expert-v2': 'che2-e',
    'ant-random-v2': 'ant-r',
    'ant-medium-v2': 'ant-m',
    'ant-medium-replay-v2': 'ant-m-re',
    'ant-medium-expert-v2': 'ant-m-e',
    'ant-expert-v2': 'ant-e',
}

N_SEEDS = 5
def process_checkpoint(args):
    if os.path.isfile(args.ckpt):
        print(f"[INFO] Preparing to load checkpoint: {args.ckpt}")
        return args.ckpt

    elif os.path.isdir(args.ckpt):
        entries = os.listdir(args.ckpt)
        entries.sort()

        print("\tFound %d experiments." % (len(entries)))
        ckpt_list = []
        for entry in entries:
            ckpt_file = os.path.join(args.ckpt, entry, args.ckpt_steps)
            if not os.path.isfile(ckpt_file):
                print("\tCannot find checkpoint {} in {}".format(args.ckpt_steps, ckpt_file))
            else:
                ckpt_list.append(ckpt_file)
        assert 1 <= args.seed <= N_SEEDS and len(ckpt_list) == N_SEEDS
        print(f"[INFO] Preparing to load checkpoint: {ckpt_list[args.seed - 1]}")
        return ckpt_list[args.seed - 1]

    else:
        print("[INFO] Training from scratch.")
        return None

def load_buffer_from_checkpoint(args):
    from d3rlpy.dataset import MDPDataset
    assert os.path.isfile(args.ckpt)
    ckpt_dir = args.ckpt[:args.ckpt.rfind('/')]
    ckpt_step = int(args.ckpt_steps.split('.')[0].split('_')[-1])
    entries = glob.glob(os.path.join(ckpt_dir, "*.h5"))
    entries.sort()
    buffer_path = entries[-1]
    buffer_at_step = int(buffer_path.split('/')[-1].split('_')[-1][:-8])
    total_of_samples = buffer_at_step

    with h5py.File(buffer_path, 'r') as f:
        observations = f['observations'][()]
        actions = f['actions'][()]
        rewards = f['rewards'][()]
        terminals = f['terminals'][()]
        discrete_action = f['discrete_action'][()]

        # for backward compatibility
        if 'episode_terminals' in f:
            episode_terminals = f['episode_terminals'][()]
        else:
            episode_terminals = None

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_terminals=episode_terminals,
        discrete_action=discrete_action,
    )
    print(f"[INFO] Loaded previous buffer (n_episodes={dataset.size()}): {buffer_path}")

    stats_filename = copy.copy(args.ckpt)
    stats_filename = stats_filename.replace('model_', 'stats_')
    stats_filename = stats_filename.replace('.pt', '.npz')
    if os.path.isfile(stats_filename):
        data = np.load(stats_filename)
        mean, std = data['mean'], data['std']
    else:
        raise ValueError("Cannot find statistics of buffer.")

    buffer_state = dict(
        total_samples=total_of_samples,
        obs_mean=mean, obs_std=std,
        obs_sum=mean * total_of_samples,
        obs_sum_sq=(std ** 2 + mean ** 2) * total_of_samples
    )
    return dataset.episodes, buffer_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='TD3', choices=['TD3', 'SAC'])
    parser.add_argument('--dataset', type=str, default='walker2d-medium-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--project', type=str, default='WALKER')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--logdir', type=str, default='d3rlpy_logs')
    parser.add_argument('--n_steps', type=int, default=2000000)
    parser.add_argument('--n_steps_collect_data', type=int, default=10000000)
    parser.add_argument('--n_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--n_eval_episodes', type=int, default=10)

    parser.add_argument('--no_replacement', action='store_true', default=False)
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--stats_update_interval', type=int, default=1000)

    parser.add_argument('--loss_type', type=str, default="normal", choices=["normal", "mad_loss"])
    parser.add_argument('--attack_type', type=str, default="actor_state_linf")
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--actor_reg', type=float, default=0.5)

    # Scope for Vector Quantization representation
    parser.add_argument('--use_vq_in', action='store_true', default=False)
    parser.add_argument('--codebook_update_type', type=str, default="ema", choices=["ema", "sgd"])
    parser.add_argument('--n_embeddings', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=1)
    parser.add_argument('--vq_decay', type=float, default=0.99)
    parser.add_argument('--vq_loss_weight', type=float, default=1.0)
    parser.add_argument('--autoscale_vq_loss', action='store_true', default=False)
    parser.add_argument('--scale_factor', type=float, default=60.0)
    parser.add_argument('--n_steps_allow_update_cb', type=int, default=10000000)
    parser.add_argument('--n_steps_start_at', type=int, default=0)

    parser.add_argument('--vq_decay_scheduler', action='store_true', default=False)
    parser.add_argument('--vq_decay_start_val', type=float, default=0.5)
    parser.add_argument('--vq_decay_end_val', type=float, default=0.99)
    parser.add_argument('--vq_decay_start_step', type=int, default=0)
    parser.add_argument('--vq_decay_end_step', type=int, default=1000000)

    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--ckpt', type=str, default='none')
    parser.add_argument('--ckpt_steps', type=str, default='model_500000.pt')
    parser.add_argument('--load_buffer', action='store_true', default=False)
    parser.add_argument('--backup_file', action='store_true')

    args = parser.parse_args()

    if args.finetune:
        args.ckpt = process_checkpoint(args)

    env = gym.make(args.dataset)
    eval_env = gym.make(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    if args.standardization:
        scaler = StandardScaler(mean=np.zeros(env.observation_space.shape), std=np.ones(env.observation_space.shape))
    else:
        scaler = None

    if args.algo == 'TD3':
        raise NotImplementedError
    elif args.algo == 'SAC':
        if args.loss_type == "mad_loss":
            adv_params = dict(
                epsilon=args.epsilon,
                num_steps=args.num_steps,
                step_size=args.epsilon / args.num_steps,
                attack_type=args.attack_type,
                actor_reg=args.actor_reg,
            )
        else:
            adv_params = {}

        vq_decay_scheduler = dict(
            vq_decay_scheduler=args.vq_decay_scheduler,
            vq_decay_start_val=args.vq_decay_start_val, vq_decay_end_val=args.vq_decay_end_val,
            vq_decay_start_step=args.vq_decay_start_step, vq_decay_end_step=args.vq_decay_end_step,
        )
        sac = d3rlpy.algos.SAC(batch_size=256,
                               use_gpu=args.gpu,
                               scaler=scaler,
                               replacement=not args.no_replacement,
                               env_name=args.dataset,
                               use_vq_in=args.use_vq_in,
                               codebook_update_type=args.codebook_update_type,
                               n_steps_allow_update_cb=args.n_steps_allow_update_cb,
                               n_steps_start_at=args.n_steps_start_at,
                               number_embeddings=args.n_embeddings,
                               embedding_dim=args.embedding_dim,
                               decay=args.vq_decay,
                               vq_decay_scheduler=vq_decay_scheduler,
                               vq_loss_weight=args.vq_loss_weight,
                               autoscale_vq_loss=args.autoscale_vq_loss,
                               scale_factor=args.scale_factor,
                               loss_type=args.loss_type,
                               adv_params=adv_params,
                               )

        previous_buffer, buffer_state = None, None
        if args.finetune and args.load_buffer:
            previous_buffer, buffer_state = load_buffer_from_checkpoint(args)

        buffer_size = 1000000
        if args.n_steps < buffer_size:
            buffer_size = args.n_steps

        # replay buffer for experience replay
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_size, env=env,
                                                    compute_statistical=args.standardization, episodes=previous_buffer)
        if args.finetune and buffer_state is not None:
            buffer._total_samples = buffer_state["total_samples"]
            buffer._obs_sum = buffer_state["obs_sum"]
            buffer._obs_sum_sq = buffer_state["obs_sum_sq"]
            buffer._obs_mean = buffer_state["obs_mean"]
            buffer._obs_std = buffer_state["obs_std"]

        # start training
        sac.fit_online(
            env,
            buffer,
            eval_env=eval_env,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            update_interval=1,
            update_start_step=1000,
            save_interval=args.save_interval,
            logdir=args.logdir,
            wandb_project=set_name_wandb_project(args.dataset),
            use_wandb=args.wandb,
            experiment_name=f"{args.exp}",
            eval_interval=args.eval_interval,
            standardization=args.standardization,
            stats_update_interval=args.stats_update_interval,
            finetune=args.finetune,
            checkpoint=args.ckpt,
            backup_file=True,
        )


if __name__ == '__main__':
    main()

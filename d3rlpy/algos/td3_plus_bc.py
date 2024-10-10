from typing import Any, Dict, Optional, Sequence, Callable

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.td3_plus_bc_impl import TD3PlusBCImpl

# Additional imports for custom online training
import gym
from tqdm import tqdm
from ..base import LearnableBase
from ..constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
)
from ..online.buffers import Buffer, ReplayBuffer
from ..online.iterators import _setup_algo
from ..torch_utility import TorchMiniBatch

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import numpy as np
from d3rlpy.adversarial_training.attackers import critic_action_attack


def _assert_action_space(algo: LearnableBase, env: gym.Env) -> None:
    if isinstance(env.action_space, gym.spaces.Box):
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR
    elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
        assert (
            algo.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        action_space = type(env.action_space)
        raise ValueError(f"The action-space is not supported: {action_space}")


class TD3PlusBC(AlgoBase):
    r"""TD3+BC algorithm.

    TD3+BC is an simple offline RL algorithm built on top of TD3.
    TD3+BC introduces BC-reguralized policy objective function.

    .. math::

        J(\phi) = \mathbb{E}_{s,a \sim D}
            [\lambda Q(s, \pi(s)) - (a - \pi(s))^2]

    where

    .. math::

        \lambda = \frac{\alpha}{\frac{1}{N} \sum_(s_i, a_i) |Q(s_i, a_i)|}

    References:
        * `Fujimoto et al., A Minimalist Approach to Offline Reinforcement
          Learning. <https://arxiv.org/abs/2106.06860>`_

    Args:
        actor_learning_rate (float): learning rate for a policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        alpha (float): :math:`\alpha` value.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.td3_impl.TD3Impl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _target_smoothing_sigma: float
    _target_smoothing_clip: float
    _alpha: float
    _update_actor_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[TD3PlusBCImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        target_smoothing_sigma: float = 0.2,
        target_smoothing_clip: float = 0.5,
        alpha: float = 2.5,
        update_actor_interval: int = 2,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = "standard",
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[TD3PlusBCImpl] = None,
        env_name: str = '',
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip
        self._alpha = alpha
        self._update_actor_interval = update_actor_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        self._env_name = env_name

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = TD3PlusBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            target_smoothing_sigma=self._target_smoothing_sigma,
            target_smoothing_clip=self._target_smoothing_clip,
            alpha=self._alpha,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            env_name=self._env_name,
        )
        self._impl.build()
        assert self.scaler._mean is not None and self.scaler._std is not None
        self._impl.init_range_of_norm_obs()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        critic_loss, q_target, current_q1, current_q2 = self._impl.update_critic(batch)
        metrics.update({
            "critic_loss": critic_loss,
            "q_target": q_target,
            "q1_prediction": current_q1,
            "q2_prediction": current_q2,
        })

        # delayed policy update
        if self._grad_step % self._update_actor_interval == 0:
            actor_loss, main_actor_loss, bc_loss = self._impl.update_actor(batch)
            metrics.update({
                "actor_loss": actor_loss,
                "actor_loss_main": main_actor_loss,
                "bc_loss": bc_loss
            })
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


    def fit_sarsa(
        self,
        env: gym.Env,
        buffer: Optional[Buffer] = None,
        n_steps: int = 30000,
        update_start_step: int = 100000,
        timelimit_aware: bool = True,
        batch_size: int = 256,
        log_interval: int = 2000,
        expl_noise: float = 0.1,
        sarsa_reg: float = 1,
        attack_epsilon: float = 0.05,
        attack_iteration: int = 5,
    ) -> None:

        attack_stepsize = attack_epsilon / attack_iteration

        if buffer is None:
            buffer = BufferSarsaWrapper(update_start_step, env=env)

        # check action-space
        _assert_action_space(self, env)

        _setup_algo(self, env)

        observation_shape = env.observation_space.shape
        assert len(observation_shape) == 1, "Do not support image env."

        max_action = float(env.action_space.high[0])
        action_dim = env.action_space.shape[0]

        # Collecting data using learned policy
        observation = env.reset()
        rollout_return = 0.0
        n_episodes = 0
        for _ in tqdm(range(1, update_start_step + 1)):
            action = self.sample_action([observation])[0]
            nosie = np.random.normal(0, max_action * expl_noise, size=action_dim).astype(action.dtype)
            action = (action + nosie).clip(-max_action, max_action)

            next_observation, reward, terminal, info = env.step(action)
            rollout_return += reward

            # special case for TimeLimit wrapper
            if timelimit_aware and "TimeLimit.truncated" in info:
                clip_episode = True
                terminal = False
            else:
                clip_episode = terminal

            # store observation
            buffer.append(
                observation=observation,
                action=action,
                reward=reward,
                terminal=terminal,
                clip_episode=clip_episode,
            )

            if clip_episode:
                n_episodes += 1
                if n_episodes % 5 == 0:
                    print("[INFO] Return: ", rollout_return)
                observation = env.reset()
                rollout_return = 0.0
            else:
                observation = next_observation

        robust_beta = 0.0
        # Start training loop
        for total_step in tqdm(range(1, n_steps + 1), desc="SARSA training"):
            batch, next_action = buffer.sample(
                batch_size=batch_size,
                n_frames=1,
                n_steps=1,
                gamma=self.gamma,
            )

            batch = TorchMiniBatch(
                batch,
                self._impl.device,
                scaler=self.scaler,
            )

            next_action = default_collate(next_action)
            next_action = next_action.to(self._impl.device)

            # For debug
            with torch.no_grad():
                q_prediction = self._impl._q_func(batch.observations, batch.actions, reduction="none")
                q1_pred = q_prediction[0].cpu().detach().numpy().mean()
                q2_pred = q_prediction[1].cpu().detach().numpy().mean()

            """
            training Q-value as SARSA style
            """
            self._impl._critic_optim.zero_grad()

            with torch.no_grad():
                # Still use double-q style from TD3 for compute Q target but without
                # smoothing target action
                q_tpn = self._impl._targ_q_func.compute_target(
                    batch.next_observations,
                    next_action,
                    reduction="min"
                )

                # This is the ground truth value to compute robust SARSA
                gt_qval = self._impl._q_func(batch.observations, batch.actions, "none")

            # Normal Bellman error
            loss = self._impl._q_func.compute_error(
                observations=batch.observations,
                actions=batch.actions,
                rewards=batch.rewards,
                target=q_tpn,
                terminals=batch.terminals,
                gamma=self._gamma ** batch.n_steps,
            )

            if sarsa_reg > 1e-5:
                # TODO: Do we need to scale `attack_epsilon` by state_std and action_std ?
                # q_lb, q_ub = self._impl._q_func.compute_bound(
                #     x_lb=batch.observations - attack_epsilon,
                #     x_ub=batch.observations + attack_epsilon,
                #     a_lb=batch.actions - attack_epsilon,
                #     a_ub=batch.actions + attack_epsilon,
                #     x=batch.observations, a=batch.actions,
                #     beta=robust_beta,
                # )
                # critic_reg_loss = ((q_ub[0] - q_lb[0]).mean() + (q_ub[1] - q_lb[1]).mean()) / 2
                # loss += sarsa_reg * critic_reg_loss

                a_adv = critic_action_attack(batch.observations, batch.actions,
                                             self._impl._policy, self._impl._q_func,
                                             attack_epsilon, attack_iteration, attack_stepsize,
                                             self._impl._obs_min_norm, self._impl._obs_max_norm,
                                             )

                qval_adv = self._impl._q_func(batch.observations, a_adv, "none")
                q1_reg_loss = F.mse_loss(qval_adv[0], gt_qval[0])
                q2_reg_loss = F.mse_loss(qval_adv[1], gt_qval[1])
                critic_reg_loss = (q1_reg_loss + q2_reg_loss) / 2
                loss += sarsa_reg * critic_reg_loss



            loss.backward()
            self._impl._critic_optim.step()

            if total_step % self._update_actor_interval == 0:
                self._impl.update_critic_target()

            if total_step % log_interval == 0:
                print("Iter: %d, q1_pred=%.2f, q2_pred=%.2f, critic_loss=%.4f, reg_loss=%.4f" %
                      (total_step, q1_pred, q2_pred, loss.item(), critic_reg_loss.item()))


class BufferSarsaWrapper(ReplayBuffer):

    def __init__(
        self,
        maxlen: int,
        env: Optional[gym.Env] = None,
    ):
        super().__init__(maxlen, env)


    def sample(
        self, batch_size: int,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        indices = np.random.choice(len(self._transitions), batch_size)
        transitions = []
        next_action = []

        for index in indices:
            # Avoid last sample in buffer to take 'next_action'
            if index == len(self._transitions) - 1:
                index = len(self._transitions) - 2

            transitions.append(self._transitions[index])

            if self._transitions[index].terminal != 1.0:
                next_action.append(self._transitions[index + 1].action)
            else:
                # This is arbitrary, since it will be ignored latter when computing target
                next_action.append(self._transitions[index].action)

        batch = TransitionMiniBatch(transitions, n_frames, n_steps, gamma)
        return batch, next_action

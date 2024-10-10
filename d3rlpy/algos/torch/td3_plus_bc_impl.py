# pylint: disable=too-many-ancestors

from typing import Optional, Sequence, Tuple

import torch
import numpy as np

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .td3_impl import TD3Impl
from ...adversarial_training import ENV_OBS_RANGE

class TD3PlusBCImpl(TD3Impl):

    _alpha: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        env_name: str = '',
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

        env_name_ = env_name.split('-')
        self.env_name = env_name_[0] + '-' + env_name_[-1]
        self._obs_max = torch.Tensor(ENV_OBS_RANGE[self.env_name]['max']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))
        self._obs_min = torch.Tensor(ENV_OBS_RANGE[self.env_name]['min']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))

        self._obs_max_norm = self._obs_min_norm = None

    def init_range_of_norm_obs(self):
        self._obs_max_norm = self.scaler.transform(
            torch.Tensor(ENV_OBS_RANGE[self.env_name]['max']).to('cuda:{}'.format(
                self._use_gpu.get_id()))
        )
        self._obs_min_norm = self.scaler.transform(
            torch.Tensor(ENV_OBS_RANGE[self.env_name]['min']).to('cuda:{}'.format(
                self._use_gpu.get_id()))
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        actor_loss = lam * -q_t.mean()
        bc_loss = ((batch.actions - action) ** 2).mean()
        total_loss = actor_loss + bc_loss
        return total_loss, actor_loss, bc_loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_policy is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._critic_optim is not None
        with torch.no_grad():
            q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
            q1_pred = q_prediction[0].cpu().detach().numpy().mean()
            q2_pred = q_prediction[1].cpu().detach().numpy().mean()

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy(), q_tpn.cpu().detach().numpy().mean(), q1_pred, q2_pred

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss, actor_loss, bc_loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy(), actor_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy()

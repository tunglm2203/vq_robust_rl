import copy
import math
from typing import Optional, Sequence, Tuple, Any, Dict

import numpy as np
import torch
from torch.optim import Optimizer
from torch.distributions import kl_divergence

from ...gpu import Device
from ...models.builders import (
    create_categorical_policy,
    create_discrete_q_function,
    create_parameter,
    create_squashed_normal_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    CategoricalPolicy,
    EnsembleDiscreteQFunction,
    EnsembleQFunction,
    Parameter,
    Policy,
    SquashedNormalPolicy,
)
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, hard_sync, torch_api, train_api
from .base import TorchImplBase
from .ddpg_impl import DDPGBaseImpl
from .utility import DiscreteQFunctionMixin

from ...adversarial_training import ENV_OBS_RANGE
from ...adversarial_training.attackers import (
    random_attack,
    actor_state_attack,
)


class SACImpl(DDPGBaseImpl):

    _policy: Optional[SquashedNormalPolicy]
    _targ_policy: Optional[SquashedNormalPolicy]
    _temp_learning_rate: float
    _temp_optim_factory: OptimizerFactory
    _initial_temperature: float
    _log_temp: Optional[Parameter]
    _temp_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        env_name: str = '',
        vq_loss_weight: float = 1.0,
        autoscale_vq_loss: float = False,
        scale_factor: float = 60.0,
        loss_type: str = 'normal',
        adv_params: dict = {},
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
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._temp_learning_rate = temp_learning_rate
        self._temp_optim_factory = temp_optim_factory
        self._initial_temperature = initial_temperature

        # initialized in build
        self._log_temp = None
        self._temp_optim = None

        if env_name not in ["Ant-v4", "InvertedPendulum-v4", "Reacher-v4", "InvertedDoublePendulum-v4", "Swimmer-v4"]:
            env_name_ = env_name.split('-')
            self.env_name = env_name_[0] + '-' + env_name_[-1]
        else:
            self.env_name = env_name

        self._obs_max_norm = self._obs_min_norm = None

        self.vq_loss_weight = vq_loss_weight
        self._autoscale_vq_loss = autoscale_vq_loss
        self._scale_factor = scale_factor

        self._loss_type = loss_type
        self._adv_params = adv_params

        self.allow_update_codebook = True
        self.allow_adversarial_training = False
        self.total_update_steps = 0

    def sync_codebook_from_policy(self):
        assert self._policy.vq_input is not None
        with torch.no_grad():
            self._targ_policy.vq_input.codebooks.data.copy_(self._policy.vq_input.codebooks.data)

    def build(self, policy_args: dict = {}) -> None:
        self._build_temperature()
        super().build(policy_args)
        self._build_temperature_optim()

    def _build_actor(self, use_vq_in: bool = False, codebook_update_type: str = "ema",
                     number_embeddings: int = 128, embedding_dim: int = 1, decay: float = 0.99) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            use_vq_in=use_vq_in, codebook_update_type=codebook_update_type,
            number_embeddings=number_embeddings, embedding_dim=embedding_dim, decay=decay
        )

    def _build_temperature(self) -> None:
        initial_val = math.log(self._initial_temperature)
        self._log_temp = create_parameter((1, 1), initial_val)

    def _build_temperature_optim(self) -> None:
        assert self._log_temp is not None
        self._temp_optim = self._temp_optim_factory.create(
            self._log_temp.parameters(), lr=self._temp_learning_rate
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> Dict:
        assert self._critic_optim is not None
        summary_logs = {}

        batch_mean = batch.observations.mean(dim=0).mean().item()
        batch_std = batch.observations.std(dim=0).mean().item()

        with torch.no_grad():
            q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
            q1_pred = q_prediction[0].cpu().detach().numpy().mean()
            q2_pred = q_prediction[1].cpu().detach().numpy().mean()

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        summary_logs.update({
            "critic_total_loss": loss.cpu().detach().numpy(),
            "q_target": q_tpn.cpu().detach().numpy().mean(),
            "q1_prediction": q1_pred,
            "q2_prediction": q2_pred,
            "batch_mean": batch_mean,
            "batch_std": batch_std,
        })

        return summary_logs

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
            gamma=self._gamma ** batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> Dict:
        assert self._q_func is not None
        assert self._actor_optim is not None
        summary_logs = {}

        if self._loss_type == "normal":

            # Q function should be inference mode for stability
            self._q_func.eval()
            self._actor_optim.zero_grad()

            if self.policy.use_vq_in and self.allow_update_codebook:
                self.policy.vq_input.enable_update_codebook()
            loss, actor_q_loss, entropy, extra_outs = self.compute_actor_loss(batch)
            if self.policy.use_vq_in:
                self.policy.vq_input.disable_update_codebook()

            vq_loss = extra_outs.get("vq_loss", -1.0)  # -1 is no meaning, just log
            scale = torch.tensor(0.0)
            if vq_loss != -1.0 and self._policy.codebook_update_type == "sgd":
                if self._autoscale_vq_loss:
                    scale = (actor_q_loss.abs().mean().detach() / self._scale_factor).detach()
                    loss += scale * vq_loss
                else:
                    loss += self.vq_loss_weight * vq_loss

            loss.backward()
            self._actor_optim.step()

        elif self._loss_type == "mad_loss":
            if self.allow_adversarial_training:
                batch_aug = self.do_augmentation(batch, self._adv_params["epsilon"])
            else:
                batch_aug = None

            # Q function should be inference mode for stability
            self._q_func.eval()
            self._actor_optim.zero_grad()

            if self.policy.use_vq_in and self.allow_update_codebook:
                self.policy.vq_input.enable_update_codebook()
            loss, actor_q_loss, entropy, tanh_dist, extra_outs = self.compute_actor_loss_for_at(batch)
            if self.policy.use_vq_in:
                self.policy.vq_input.disable_update_codebook()

            vq_loss = extra_outs.get("vq_loss", -1.0)  # -1 is no meaning, just log
            scale = torch.tensor(0.0)
            if vq_loss != -1.0 and self._policy.codebook_update_type == "sgd":
                if self._autoscale_vq_loss:
                    scale = (actor_q_loss.abs().mean().detach() / self._scale_factor).detach()
                    loss += scale * vq_loss
                else:
                    loss += self.vq_loss_weight * vq_loss

            if self.allow_adversarial_training:
                tanh_dist_noise, _ = self._policy.dist(batch_aug.observations)
                actor_reg_loss = kl_divergence(tanh_dist._dist, tanh_dist_noise._dist).sum(axis=-1)
                actor_reg_loss = actor_reg_loss.mean()
                loss += self._adv_params["actor_reg"] * actor_reg_loss
            else:
                actor_reg_loss = torch.tensor(0.0)

            loss.backward()
            self._actor_optim.step()

            summary_logs.update({
                "actor_reg_loss": actor_reg_loss.item(),
            })

        else:
            raise NotImplementedError

        if self.policy.use_vq_in:
            summary_logs.update({
                "debug_sum_cb_value": self.policy.vq_input.codebooks.detach().mean().item()
            })

        summary_logs.update({
            "actor_total_loss": loss.cpu().detach().numpy(),
            "actor_q_loss": actor_q_loss.cpu().detach().numpy().mean(),
            "entropy": entropy.cpu().detach().numpy().mean(),
            "vq_loss": vq_loss.item(),
            "scale_for_vq_loss": scale.item(),
        })

        return summary_logs

    def compute_actor_loss(self, batch: TorchMiniBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._q_func is not None
        action, log_prob, extra_outs = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp().detach() * log_prob
        q_t = self._q_func(batch.observations, action, "min")
        loss = (entropy - q_t).mean()
        return loss, -q_t, entropy, extra_outs

    def compute_actor_loss_for_at(self, batch: TorchMiniBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any]:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._q_func is not None
        dist, extra_outs = self._policy.dist(batch.observations)
        action, log_prob = dist.sample_with_log_prob()
        entropy = self._log_temp().exp().detach() * log_prob
        q_t = self._q_func(batch.observations, action, "min")
        loss = (entropy - q_t).mean()
        return loss, -q_t, entropy, dist, extra_outs

    @train_api
    @torch_api()
    def update_temp(
        self, batch: TorchMiniBatch
    ) -> Dict:
        assert self._temp_optim is not None
        assert self._policy is not None
        assert self._log_temp is not None
        summary_logs = {}

        self._temp_optim.zero_grad()

        with torch.no_grad():
            _, log_prob, _ = self._policy.sample_with_log_prob(batch.observations)
            targ_temp = log_prob - self._action_size

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        summary_logs.update({
            "temp_loss": loss.cpu().detach().numpy(),
            "temp": cur_temp,
        })
        return summary_logs

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action, log_prob, _ = self._policy.sample_with_log_prob(
                batch.next_observations
            )
            entropy = self._log_temp().exp() * log_prob
            target = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy

    def do_augmentation(self, batch: TorchMiniBatch, epsilon=None):
        """" NOTE: Assume obs, next_obs are already normalized """""

        batch_aug = copy.deepcopy(batch)

        num_steps = self._adv_params["num_steps"]
        step_size = self._adv_params["step_size"]
        attack_type = self._adv_params["attack_type"]

        assert (epsilon is not None) and (num_steps is not None) and \
               (step_size is not None) and (attack_type is not None)

        if attack_type == 'rand_state':
            adv_x = random_attack(batch_aug._observations, epsilon,
                                  self._obs_min_norm, self._obs_max_norm,
                                  clip=False, use_assert=True)
            batch_aug._observations = adv_x

        elif attack_type == 'actor_state_linf':
            adv_x = actor_state_attack(batch_aug._observations,
                                       self._policy, None,
                                       epsilon, num_steps, step_size,
                                       self._obs_min_norm, self._obs_max_norm,
                                       clip=False, use_assert=True)

            batch_aug._observations = adv_x

        else:
            raise NotImplementedError

        return batch_aug

class DiscreteSACImpl(DiscreteQFunctionMixin, TorchImplBase):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _initial_temperature: float
    _use_gpu: Optional[Device]
    _policy: Optional[CategoricalPolicy]
    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]
    _log_temp: Optional[Parameter]
    _actor_optim: Optional[Optimizer]
    _critic_optim: Optional[Optimizer]
    _temp_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        initial_temperature: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=None,
            reward_scaler=reward_scaler,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._initial_temperature = initial_temperature
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._policy = None
        self._targ_q_func = None
        self._log_temp = None
        self._actor_optim = None
        self._critic_optim = None
        self._temp_optim = None

    def build(self) -> None:
        self._build_critic()
        self._build_actor()
        self._build_temperature()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()
        self._build_temperature_optim()

    def _build_critic(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )

    def _build_actor(self) -> None:
        self._policy = create_categorical_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    def _build_temperature(self) -> None:
        initial_val = math.log(self._initial_temperature)
        self._log_temp = create_parameter((1, 1), initial_val)

    def _build_temperature_optim(self) -> None:
        assert self._log_temp is not None
        self._temp_optim = self._temp_optim_factory.create(
            self._log_temp.parameters(), lr=self._temp_learning_rate
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            log_probs = self._policy.log_probs(batch.next_observations)
            probs = log_probs.exp()
            entropy = self._log_temp().exp() * log_probs
            target = self._targ_q_func.compute_target(batch.next_observations)
            keepdims = True
            if target.dim() == 3:
                entropy = entropy.unsqueeze(-1)
                probs = probs.unsqueeze(-1)
                keepdims = False
            return (probs * (target - entropy)).sum(dim=1, keepdim=keepdims)

    def compute_critic_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._log_temp is not None
        with torch.no_grad():
            q_t = self._q_func(batch.observations, reduction="min")
        log_probs = self._policy.log_probs(batch.observations)
        probs = log_probs.exp()
        entropy = self._log_temp().exp() * log_probs
        return (probs * (entropy - q_t)).sum(dim=1).mean()

    @train_api
    @torch_api()
    def update_temp(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._temp_optim is not None
        assert self._policy is not None
        assert self._log_temp is not None

        self._temp_optim.zero_grad()

        with torch.no_grad():
            log_probs = self._policy.log_probs(batch.observations)
            probs = log_probs.exp()
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = 0.98 * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.best_action(x)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.sample(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    @property
    def policy(self) -> Policy:
        assert self._policy
        return self._policy

    @property
    def policy_optim(self) -> Optimizer:
        assert self._actor_optim
        return self._actor_optim

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._critic_optim
        return self._critic_optim

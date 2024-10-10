import math
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, cast, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from .distributions import GaussianDistribution, SquashedGaussianDistribution
from .encoders import Encoder, EncoderWithAction
from .vector_quantization import *


def squash_action(
    dist: torch.distributions.Distribution, raw_action: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    squashed_action = torch.tanh(raw_action)
    jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=-1, keepdims=True)
    return squashed_action, log_prob


class Policy(nn.Module, metaclass=ABCMeta):  # type: ignore
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample_with_log_prob(x)[0]

    @abstractmethod
    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_n_with_log_prob(x, n)[0]

    @abstractmethod
    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DeterministicPolicy(Policy):

    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int,
                 use_vq_in: bool = False, codebook_update_type: str = "ema",
                 number_embeddings: int = 128, embedding_dim: int = 1,
                 commitment_cost: float = 0.25, decay: float = 0.99
                 ):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

        # Create VectorQuantization module
        self.use_vq_in = use_vq_in
        self.codebook_update_type = codebook_update_type
        if use_vq_in:
            n_codebooks = self._encoder._observation_shape[0]
            if codebook_update_type == "ema":
                self.vq_input = VectorQuantizerEMA_unshared_codebook(number_embeddings, embedding_dim,
                                                                     commitment_cost, decay,
                                                                     n_codebooks, update_codebook=True)
            elif codebook_update_type == "sgd":
                self.vq_input = VectorQuantizer_unshared_codebook(number_embeddings, embedding_dim,
                                                                  commitment_cost, decay,
                                                                  n_codebooks, update_codebook=True)
            else:
                raise NotImplementedError
        else:
            self.vq_input = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        if self.use_vq_in:
            vq_loss, quantized_x = self.vq_input(x)  # EMA
        else:
            quantized_x = x
            vq_loss = None

        h = self._encoder(quantized_x)
        extra_outs = {"vq_loss": vq_loss}
        return torch.tanh(self._fc(h)), extra_outs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        action, _ = self.forward(x)
        return action


class DeterministicResidualPolicy(Policy):

    _encoder: EncoderWithAction
    _scale: float
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, scale: float):
        super().__init__()
        self._scale = scale
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), encoder.action_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        residual_action = self._scale * torch.tanh(self._fc(h))
        return (action + cast(torch.Tensor, residual_action)).clamp(-1.0, 1.0)

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def best_residual_action(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, action)

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "residual policy does not support best_action"
        )

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )


class NormalPolicy(Policy):

    _encoder: Encoder
    _action_size: int
    _min_logstd: float
    _max_logstd: float
    _use_std_parameter: bool
    _mu: nn.Linear
    _logstd: Union[nn.Linear, nn.Parameter]

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
        squash_distribution: bool,
        use_vq_in: bool = False, codebook_update_type: str = "ema",
        number_embeddings: int = 128, embedding_dim: int = 1,
        commitment_cost: float = 0.25, decay: float = 0.99
    ):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._use_std_parameter = use_std_parameter
        self._squash_distribution = squash_distribution
        self._mu = nn.Linear(encoder.get_feature_size(), action_size)
        if use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self._logstd = nn.Parameter(initial_logstd)
        else:
            self._logstd = nn.Linear(encoder.get_feature_size(), action_size)

        # Create VectorQuantization module
        self.use_vq_in = use_vq_in
        self.codebook_update_type = codebook_update_type
        if use_vq_in:
            n_codebooks = self._encoder._observation_shape[0]
            if codebook_update_type == "ema":
                self.vq_input = VectorQuantizerEMA_unshared_codebook(number_embeddings, embedding_dim,
                                                                     commitment_cost, decay,
                                                                     n_codebooks, update_codebook=True)
            elif codebook_update_type == "sgd":
                self.vq_input = VectorQuantizer_unshared_codebook(number_embeddings, embedding_dim,
                                                                  commitment_cost, decay,
                                                                  n_codebooks, update_codebook=True)
            else:
                raise NotImplementedError
        else:
            self.vq_input = None

    def _compute_logstd(self, h: torch.Tensor) -> torch.Tensor:
        if self._use_std_parameter:
            clipped_logstd = self.get_logstd_parameter()
        else:
            logstd = cast(nn.Linear, self._logstd)(h)
            clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return clipped_logstd

    def dist(
        self, x: torch.Tensor
    ) -> Tuple[Union[GaussianDistribution, SquashedGaussianDistribution], Any]:
        if self.use_vq_in:
            vq_loss, quantized_x = self.vq_input(x)  # EMA
        else:
            quantized_x = x
            vq_loss = torch.tensor(-1.0)

        h = self._encoder(quantized_x)
        mu = self._mu(h)
        clipped_logstd = self._compute_logstd(h)
        if self._squash_distribution:
            extra_outs = {"vq_loss": vq_loss}
            return SquashedGaussianDistribution(mu, clipped_logstd.exp()), extra_outs
        else:
            extra_outs = {"vq_loss": vq_loss}
            return GaussianDistribution(
                torch.tanh(mu),
                clipped_logstd.exp(),
                raw_loc=mu,
            ), extra_outs

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        dist, extra_outs = self.dist(x)
        if deterministic:
            action, log_prob = dist.mean_with_log_prob()
        else:
            action, log_prob = dist.sample_with_log_prob()
        return (action, log_prob, extra_outs) if with_log_prob else (action, extra_outs)

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        action, log_prob, extra_outs = self.forward(x, with_log_prob=True)
        return action, log_prob, extra_outs

    def sample_n_with_log_prob(
        self,
        x: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self.dist(x)

        action_T, log_prob_T = dist.sample_n_with_log_prob(n)

        # (n, batch, action) -> (batch, n, action)
        transposed_action = action_T.transpose(0, 1)
        # (n, batch, 1) -> (batch, n, 1)
        log_prob = log_prob_T.transpose(0, 1)

        return transposed_action, log_prob

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        dist, _ = self.dist(x)
        action = dist.sample_n_without_squash(n)
        return action.transpose(0, 1)

    def onnx_safe_sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        h = self._encoder(x)
        mean = self._mu(h)
        std = self._compute_logstd(h).exp()

        if not self._squash_distribution:
            mean = torch.tanh(mean)

        # expand shape
        # (batch_size, action_size) -> (batch_size, N, action_size)
        expanded_mean = mean.view(-1, 1, self._action_size).repeat((1, n, 1))
        expanded_std = std.view(-1, 1, self._action_size).repeat((1, n, 1))

        # sample noise from Gaussian distribution
        noise = torch.randn(x.shape[0], n, self._action_size, device=x.device)

        if self._squash_distribution:
            return torch.tanh(expanded_mean + noise * expanded_std)
        else:
            return expanded_mean + noise * expanded_std

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        action, _ = self.forward(x, deterministic=True, with_log_prob=False)
        return cast(torch.Tensor, action)

    def get_logstd_parameter(self) -> torch.Tensor:
        assert self._use_std_parameter
        logstd = torch.sigmoid(cast(nn.Parameter, self._logstd))
        base_logstd = self._max_logstd - self._min_logstd
        return self._min_logstd + logstd * base_logstd


class SquashedNormalPolicy(NormalPolicy):
    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
        use_vq_in: bool = False, codebook_update_type: str = "ema",
        number_embeddings: int = 128, embedding_dim: int = 1,
        commitment_cost: float = 0.25, decay: float = 0.99
    ):
        super().__init__(
            encoder=encoder,
            action_size=action_size,
            min_logstd=min_logstd,
            max_logstd=max_logstd,
            use_std_parameter=use_std_parameter,
            squash_distribution=True,
            use_vq_in=use_vq_in, codebook_update_type=codebook_update_type,
            number_embeddings=number_embeddings, embedding_dim=embedding_dim,
            commitment_cost=commitment_cost, decay=decay
        )


class NonSquashedNormalPolicy(NormalPolicy):
    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
    ):
        super().__init__(
            encoder=encoder,
            action_size=action_size,
            min_logstd=min_logstd,
            max_logstd=max_logstd,
            use_std_parameter=use_std_parameter,
            squash_distribution=False,
        )


class CategoricalPolicy(Policy):

    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x: torch.Tensor) -> Categorical:
        h = self._encoder(x)
        h = self._fc(h)
        return Categorical(torch.softmax(h, dim=1))

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dist = self.dist(x)

        if deterministic:
            action = cast(torch.Tensor, dist.probs.argmax(dim=1))
        else:
            action = cast(torch.Tensor, dist.sample())

        if with_log_prob:
            return action, dist.log_prob(action)

        return action

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)

        action_T = cast(torch.Tensor, dist.sample((n,)))
        log_prob_T = dist.log_prob(action_T)

        # (n, batch) -> (batch, n)
        action = action_T.transpose(0, 1)
        # (n, batch) -> (batch, n)
        log_prob = log_prob_T.transpose(0, 1)

        return action, log_prob

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.forward(x, deterministic=True))

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        return cast(torch.Tensor, dist.logits)

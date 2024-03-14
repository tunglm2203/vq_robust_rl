import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

import numpy as np

from .utility import clamp

def preprocess_state(x):
    if len(x.shape) == 1:
        x = x.reshape((1,) + x.shape)

    assert len(x.shape) == 2, "Currently only support the low-dimensional state"
    return x

def random_attack(x, epsilon, _obs_min_norm, _obs_max_norm, clip=True, use_assert=True):
    """" NOTE: x must be normalized """""
    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())
    noise = torch.zeros_like(ori_x).uniform_(-epsilon, epsilon)
    adv_x = ori_x + noise

    # This clamp is performed in normalized scale
    if clip:
        adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm)

    perturbed_state = adv_x  # already normalized

    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
            epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"

    return adv_x

def critic_normal_attack(x, _policy, _q_func, epsilon, num_steps, step_size,
                         _obs_min_norm, _obs_max_norm,
                         q_func_id=0, optimizer='pgd', clip=True, use_assert=True):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())                   # already normalized

    adv_x = ori_x.clone().detach()               # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                   # Add noise in `normalized space`

    if clip:
        adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            action, _ = _policy(adv_x, deterministic=True, with_log_prob=False)
            qval = _q_func(ori_x, action, "none")[q_func_id]

            cost = -qval.mean()

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta       # This is adversarial example

            # This clamp is performed in normalized scale
            if clip:
                adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()
            else:
                adv_x = adv_x.detach()

    elif optimizer == 'sgld':
        raise NotImplementedError
    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
            epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"
    return perturbed_state

def critic_action_attack(x, a, _q_func,
                         epsilon, num_steps, step_size,
                         _obs_min_norm=None, _obs_max_norm=None,
                         q_func_id=0, optimizer='pgd', clip=True, use_assert=True):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor) and isinstance(a, torch.Tensor), "input x & a must be tensor."
    ori_a = preprocess_state(a.clone().detach())                   # already normalized
    ori_x = preprocess_state(x.clone().detach())                   # already normalized

    adv_a = ori_a.clone().detach()               # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_a).uniform_(-epsilon, epsilon)
    adv_a = adv_a + noise                                   # Add noise in `normalized space`

    adv_a = torch.clamp(adv_a, min=-1.0, max=1.0).detach()

    with torch.no_grad():
        gt_qval = _q_func(ori_x, ori_a, "none")[q_func_id].detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_a.requires_grad = True

            qval_adv = _q_func(ori_x, adv_a, "none")[q_func_id]

            cost = F.mse_loss(qval_adv, gt_qval)    # TODO: Should it be minus: qval - gt_qval ?

            grad = torch.autograd.grad(cost, adv_a, retain_graph=False, create_graph=False)[0]

            adv_a = adv_a.detach() + step_size * torch.sign(grad.detach())

            delta = torch.clamp(adv_a - ori_a, min=-epsilon, max=epsilon)
            adv_a = ori_a + delta       # This is adversarial example

            # This clamp is performed in normalized scale
            adv_a = torch.clamp(adv_a, min=-1.0, max=1.0).detach()

    elif optimizer == 'sgld':
        raise NotImplementedError
    else:
        raise NotImplementedError

    perturbed_action = adv_a     # already normalized
    if use_assert:
        assert np.max(np.linalg.norm(perturbed_action.cpu() - ori_a.cpu(), np.inf, 1)) < (
            epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_action.cpu() - ori_a.cpu(), np.inf, 1))}\n Origin: {ori_a.cpu()}, perturb: {perturbed_action.cpu()}"
    return perturbed_action


def critic_state_attack(x, a, _q_func,
                        epsilon, num_steps, step_size,
                        _obs_min_norm=None, _obs_max_norm=None,
                        q_func_id=0, optimizer='pgd', clip=True, use_assert=True):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor) and isinstance(a, torch.Tensor), "input x & a must be tensor."
    ori_a = preprocess_state(a.clone().detach())                   # already normalized
    ori_x = preprocess_state(x.clone().detach())                   # already normalized

    adv_x = ori_x.clone().detach()               # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise  # Add noise in `normalized space`

    if clip:
        adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    with torch.no_grad():
        gt_qval = _q_func(ori_x, ori_a, "none")[q_func_id].detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            qval = _q_func(adv_x, ori_a, "none")[q_func_id]

            cost = F.mse_loss(qval, gt_qval)    # TODO: Should it be minus: qval - gt_qval ?

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta       # This is adversarial example

            # This clamp is performed in normalized scale
            if clip:
                adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()
            else:
                adv_x = adv_x.detach()

    elif optimizer == 'sgld':
        raise NotImplementedError
    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
            epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"
    return perturbed_state


def actor_state_attack(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min_norm, _obs_max_norm,
                     optimizer='pgd', clip=True, use_assert=True):
    """" NOTE: x must be normalized """""

    def get_policy_kl(policy, obs, noised_obs):
        tanh_dist, _ = policy.dist(obs)
        tanh_dist_noise, _ = policy.dist(noised_obs)

        kl_loss = kl_divergence(tanh_dist._dist, tanh_dist_noise._dist).sum(axis=-1)
                  # + kl_divergence(tanh_dist_noise._dist, tanh_dist._dist).sum(axis=-1)
        return kl_loss.mean()

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())                           # already normalized

    adv_x = ori_x.clone().detach()  # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                          # Add noise in `normalized space`

    if clip:
        adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    for _ in range(num_steps):
        adv_x.requires_grad = True

        cost = get_policy_kl(_policy, ori_x, adv_x)

        grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

        adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())

        delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
        adv_x = ori_x + delta         # This is adversarial example

        if clip:
            # This clamp is performed in normalized scale
            adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()
        else:
            adv_x = adv_x.detach()


    perturbed_state = adv_x     # already normalized
    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
                epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"
    return perturbed_state


def actor_state_attack_mean(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min_norm, _obs_max_norm,
                            optimizer='pgd', clip=True, use_assert=True):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())                           # already normalized

    adv_x = ori_x.clone().detach()  # already normalized

    with torch.no_grad():
        gt_action, _ = _policy(ori_x, deterministic=True, with_log_prob=False)              # ground truth
        gt_action = gt_action.detach()

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                          # Add noise in `normalized space`

    if clip:
        adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    for _ in range(num_steps):
        adv_x.requires_grad = True

        adv_a, _ = _policy(adv_x, deterministic=True, with_log_prob=False)

        cost = F.mse_loss(adv_a, gt_action)

        grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

        adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())

        delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
        adv_x = ori_x + delta         # This is adversarial example

        if clip:
            # This clamp is performed in normalized scale
            adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()
        else:
            adv_x = adv_x.detach()


    perturbed_state = adv_x     # already normalized
    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
                epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"
    return perturbed_state

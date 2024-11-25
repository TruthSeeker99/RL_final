from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule, GymEnv
from stable_baselines3.ppo import PPO
import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
import numpy as np
import torch as th


class IPPO(PPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range,
                         clip_range_vf, normalize_advantage, ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq,
                         rollout_buffer_class, rollout_buffer_kwargs, target_kl, stats_window_size, tensorboard_log,
                         policy_kwargs, verbose, seed, device, _init_setup_model)

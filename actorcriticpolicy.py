from stable_baselines3.common.policies import BasePolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from nets import FeatureExtractorNetwork, ActionNetwork, ValueNetwork, MLPNetwork


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        self.mlp_extractor = MLPNetwork()
        self.features_extractor = FeatureExtractorNetwork(in_channels=5, out_dim=512 + 32)
        self.pi_features_extractor = FeatureExtractorNetwork(in_channels=5, out_dim=512 + 32)
        self.vf_features_extractor = FeatureExtractorNetwork(in_channels=5, out_dim=512 + 32)
        self.action_net = ActionNetwork()
        self.value_net = ValueNetwork()

    # def _build_mlp_extractor(self) -> None:
    #     self.mlp_extractor = FeatureExtractorNetwork(in_channels=5, out_dim=512+32)
    #     self.features_extractor = FeatureExtractorNetwork(in_channels=5, out_dim=512+32)
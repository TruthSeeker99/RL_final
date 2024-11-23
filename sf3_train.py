#!/usr/bin/env python3
import diambra.arena
import numpy as np
import time
import gymnasium
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.engine import SpaceTypes
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from action_wrap import MultiDiscreteToDiscreteWrapper
from actorcriticpolicy import CustomActorCriticPolicy

def main():
    # Settings
    settings = EnvironmentSettings()
    settings.frame_shape = (56, 96, 1)
    settings.step_ratio = 3  # action every 3 frames
    settings.difficulty = 1
    settings.characters = 'Ken'
    # settings.frame_shape = (224, 384, 1)
    # settings.hardcore = True
    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 10
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = []

    # Environment creation
    env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode="human")
    # env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings)
    # env = MultiDiscreteToDiscreteWrapper(env)
    # env.action_space = gymnasium.spaces.Discrete(90)
    print('>>>action space:', env.action_space, int(len(env.action_space.nvec)))
    print('>>>obs space:', env.observation_space)

    # Instantiate the agent
    policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
    # agent = PPO(CustomActorCriticPolicy, env, verbose=1)
    agent = PPO('MultiInputPolicy', env, verbose=1,
                n_steps=512,
                batch_size=1024, # 512,
                n_epochs=4,
                gamma=0.94)
    print(agent.policy.action_net)
    # Train the agent
    # for _ in range(200):
    agent.learn(total_timesteps=10000)
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)
    print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

if __name__ == '__main__':
    main()
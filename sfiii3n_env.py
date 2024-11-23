#!/usr/bin/env python3
import diambra.arena
import numpy as np
import time
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings

from action_wrap import MultiDiscreteToDiscreteWrapper


def main():
    # Settings
    settings = EnvironmentSettings()
    # settings.frame_shape = (56*8, 96*8, 3)
    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 5
    wrappers_settings.scale = True

    # Environment creation
    env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode="human")
    env = MultiDiscreteToDiscreteWrapper(env)

    # Environment reset
    observation, info = env.reset(seed=42)

    # Agent-Environment interaction loop
    while True:
        # (Optional) Environment rendering
        env.render()

        # Action random sampling
        actions = env.action_space.sample()
        print(actions)
        # print(type(actions))

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        print(observation['frame'].shape)
        print(observation["P1"]["health"])
        print(observation["P2"]["health"])
        if terminated or truncated or reward != 0.0:
            print(reward)

        # Episode end (Done condition) check
        if terminated or truncated:
            observation, info = env.reset()
            break

    # Environment shutdown
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()
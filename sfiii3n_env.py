#!/usr/bin/env python3
import diambra.arena
import numpy as np
import time
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings

from action_wrap import MultiDiscreteToDiscreteWrapper


def main():
    # Settings
    settings = EnvironmentSettings()
    settings.step_ratio = 1
    settings.difficulty = 1
    settings.characters = 'Ken'

    # settings.frame_shape = (56*8, 96*8, 3)
    # Wrappers Settings
    wrappers_settings = WrappersSettings()

    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 5
    wrappers_settings.scale = True
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 5
    # wrappers_settings.repeat_action = 4

    # Environment creation
    env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode="human")
    print(env.observation_space)
    # env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings, render_mode="human")
    # env = MultiDiscreteToDiscreteWrapper(env)

    # Environment reset
    observation, info = env.reset(seed=42)

    # Agent-Environment interaction loop
    count = 0
    while True:
        # (Optional) Environment rendering
        env.render()

        # Action random sampling
        # actions = env.action_space.sample()

        # print(type(actions))
        # if (count % 5 == 0) or (count % 5 == 1) or (count % 5 == 2) or (count % 5 == 3):

        # if (count % 5 == 0):
        #     actions = np.array([3, 0])
        # else:
        #     actions = np.array([0, 0])
        # print(actions)
        combo_list = [
            # special Moves
            [[7, 0], [6, 0], [5, 2], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],  # Hadoken
            [[7, 0], [8, 0], [1, 0], [0, 2]],  # Hadoken
            [[7, 0], [6, 0], [5, 5]],  # Hurricane Kick
            [[7, 0], [8, 0], [1, 5]],  # Hurricane Kick
            [[5, 0], [7, 0], [6, 2]],  # Shoryuken
            [[1, 0], [7, 0], [8, 2]],  # Shoryuken
            # only moves
            [[7, 0], [6, 0], [5, 0]],  # Hurricane Kick
            [[7, 0], [8, 0], [1, 0]],  # Hurricane Kick
            [[5, 0], [7, 0], [6, 0]],  # Shoryuken
            [[1, 0], [7, 0], [8, 0]],  # Shoryuken
        ]

        combo = combo_list[0]
        # combo = [[4, 0], [4, 0], [4, 0], [7, 0], [8, 0], [1, 4], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        cur_step = count % len(combo)
        actions = np.array(combo[cur_step])

        count += 1

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        # print(observation)
        # print(observation['frame'].shape)
        # print(observation['action'].shape)
        # print(observation["P1"]["health"])
        # print(observation["P2"]["health"])
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
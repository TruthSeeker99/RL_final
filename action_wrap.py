import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import queue
from time import sleep


class ComboWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ComboWrapper, self).__init__(env)
        # self.combo_list = np.array([
        #     # special Moves
        #     [[7, 0], [6, 0], [5, 2]],  # Hadoken
        #     [[7, 0], [8, 0], [1, 2]],  # Hadoken
        #     [[7, 0], [6, 0], [5, 5]],  # Hurricane Kick
        #     [[7, 0], [8, 0], [1, 5]],  # Hurricane Kick
        #     [[5, 0], [7, 0], [6, 2]],  # Shoryuken
        #     [[1, 0], [7, 0], [8, 2]],  # Shoryuken
        #     # only moves
        #     [[7, 0], [6, 0], [5, 0]],
        #     [[7, 0], [8, 0], [1, 0]],
        #     [[5, 0], [7, 0], [6, 0]],
        #     [[1, 0], [7, 0], [8, 0]],
        # ])
        self.combo_list = np.array([
            [[7, 0], [6, 0], [5, 0]],
            [[7, 0], [8, 0], [1, 0]],
            [[5, 0], [7, 0], [6, 2]],  # Shoryuken
            [[1, 0], [7, 0], [8, 2]],  # Shoryuken
            [[5, 0], [7, 0], [6, 0]],
            [[7, 0], [6, 0], [5, 5]],  # Hurricane Kick
            [[7, 0], [8, 0], [1, 5]],  # Hurricane Kick
            [[1, 0], [7, 0], [8, 0]],
            [[7, 0], [6, 0], [5, 2]],  # Hadoken
            [[7, 0], [8, 0], [1, 2]],  # Hadoken
        ])
        self.action_len = 3

    def step(self, action):
        # print(type(action), action, action[0] == 9)
        done, truncated = False, False
        all_reward = 0.0
        # print(action)
        if action[0] == 9:
            # push combo in
            # combo = queue.Queue()
            combo_idx = action[1]
            combo_to_use = self.combo_list[combo_idx]
            # print(combo_to_use)
            for i in range(self.action_len):
                if not (done or truncated):
                    obs, reward, done, truncated, info = self.env.step(combo_to_use[i])
                    # print(reward)
                    all_reward += reward
                else:
                    break
        else:
            for i in range(self.action_len):
                # print(i, action)
                if not (done or truncated):
                    obs, reward, done, truncated, info = self.env.step(action)
                    all_reward += reward
                else:
                    break
        # print(obs['action'].shape)
        # print(all_reward)
        return obs, all_reward, done, truncated, info


class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(MultiDiscreteToDiscreteWrapper, self).__init__(env)

        # Ensure the original action space is MultiDiscrete
        # assert isinstance(env.action_space, spaces.MultiDiscrete)
        # self.combo_list = [
        #     # [[0,1], [0,4], [0,0]], # Hiza-Geri
        #     # [[5,7], [0,0]], # Seoi-Nage
        #     # [[1,7], [0,0]], # Jigoku-Guruma
        #     # [[1,5], [0,0]], # Inazuma-Kakato-Wari
        #     # [[0,2], [0,3], [0,0]], # Target-Combo
        #     # special Moves
        #     [[7,0], [6,0], [5,0], [0,3], [0,0]], # Hadoken
        #     [[7,0], [8,0], [1,0], [0,3], [0,0]], # Hadoken
        #     [[7,0], [6,0], [5,0], [0,6], [0,0]], # Hadoken
        #     [[7,0], [8,0], [1,0], [0,6], [0,0]], # Hadoken
        #     # super art
        #     [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,3], [0,0]], # Shoryu-Reppa
        #     [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,3], [0,0]], # Shoryu-Reppa
        #     [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,6], [0,0]], # Shinryu-Ken
        #     [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,6], [0,0]], # Shinryu-Ken
        #     [[7,0], [6,0], [5,0], [7,0], [6,0], [5,0], [0,6], [0,6], [0,0]], # Shinryu-Jinrai-Kyaku
        #     [[7,0], [8,0], [1,0], [7,0], [8,0], [1,0], [0,6], [0,6], [0,0]], # Shinryu-Jinrai-Kyaku
        # ]

        # # Store the original MultiDiscrete action space
        # self.multi_discrete_space = env.action_space

        # # Calculate the total number of discrete actions needed
        # self.n_discrete_actions = np.prod(self.multi_discrete_space.nvec)

        # # Define the new discrete action space
        # self.action_space = spaces.Discrete(self.n_discrete_actions)
        self.action_space = spaces.MultiDiscrete([10, 10])
        # self.reset_without_seed = copy.deepcopy(env.reset)
        # self.combo = queue.Queue()

    def action(self, action):
        return action
        # # print(action.shape)
        # if self.combo.empty():
        #     if action[0] < 9:
        #         return action
        #     else:
        #         # push combo in
        #         combo_idx = action[1]
        #         combo_to_use = self.combo_list[combo_idx]
        #         for i in range(len(combo_to_use)):
        #             self.combo.put(combo_to_use[i])
        #         return self.combo.get()
        # else:
        #     return self.combo.get()

    # def reset(self, seed):
    #     self.seed(seed)
    #     self.reset_without_seed()
    #     return self.state

    # def reverse_action(self, action):
    #     # Convert multi-discrete action to discrete action
    #     discrete_action = 0
    #     multiplier = 1
    #     for act, n in zip(reversed(action), reversed(self.multi_discrete_space.nvec)):
    #         discrete_action += act * multiplier
    #         multiplier *= n
    #     return discrete_action


class CustomMultiDiscreteEnv(gym.Env):
    def __init__(self, env):
        super(CustomMultiDiscreteEnv, self).__init__()
        # Your initialization code here

    def reset(self, seed):
        self.seed(seed)
        self.reset()
        return self.state

# # Example usage with a custom environment
# class CustomMultiDiscreteEnv(gym.Env):
#     def __init__(self):
#         super(CustomMultiDiscreteEnv, self).__init__()
#         self.action_space = spaces.MultiDiscrete([2, 3, 2])  # Example multi-discrete action space
#         self.observation_space = spaces.Discrete(5)  # Example observation space
#
#     def step(self, action):
#         # Implement the environment's step function
#         obs = self.observation_space.sample()
#         reward = 1.0
#         done = False
#         info = {}
#         return obs, reward, done, info
#
#     def reset(self):
#         # Implement the environment's reset function
#         return self.observation_space.sample()
#
#
# # Instantiate the custom environment
# env = CustomMultiDiscreteEnv()
#
# # Wrap the environment to convert MultiDiscrete actions to Discrete actions
# wrapped_env = MultiDiscreteToDiscreteWrapper(env)
#
# # Test the wrapped environment
# obs = wrapped_env.reset()
# done = False
#
# while not done:
#     action = wrapped_env.action_space.sample()
#     obs, reward, done, info = wrapped_env.step(action)
#     wrapped_env.render()
#
# wrapped_env.close()

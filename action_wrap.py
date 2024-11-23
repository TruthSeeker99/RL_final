import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(MultiDiscreteToDiscreteWrapper, self).__init__(env)

        # Ensure the original action space is MultiDiscrete
        assert isinstance(env.action_space, spaces.MultiDiscrete)

        # Store the original MultiDiscrete action space
        self.multi_discrete_space = env.action_space

        # Calculate the total number of discrete actions needed
        self.n_discrete_actions = np.prod(self.multi_discrete_space.nvec)

        # Define the new discrete action space
        self.action_space = spaces.Discrete(self.n_discrete_actions)

    def action(self, action):
        # Convert discrete action to multi-discrete action
        return [action // 10, action % 10]

    # def reverse_action(self, action):
    #     # Convert multi-discrete action to discrete action
    #     discrete_action = 0
    #     multiplier = 1
    #     for act, n in zip(reversed(action), reversed(self.multi_discrete_space.nvec)):
    #         discrete_action += act * multiplier
    #         multiplier *= n
    #     return discrete_action


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

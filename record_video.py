#!/usr/bin/env python3
import os

import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings, \
    RecordingSettings
from diambra.arena.stable_baselines3.sb3_utils import AutoSave
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import record_video, record_single_video, build_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == '__main__':
    all_settings = {
        'basic': {
            'step_ratio': 3,
            'difficulty': 4,
            'frame_shape': (112, 192, 1),
            'characters': 'Ken'
        },
        'wrapper': {
            'normalize_reward': True,
            'stack_frames': 10,
            'scale': True,
            'exclude_image_scaling': True,
            'flatten': True,
            'filter_keys': [],
            'role_relative': True,
            'add_last_action': True
        }
    }
    sb3 = False
    env, num_envs = build_env(False, sb3=sb3, render_mode='rgb_array', all_settings=all_settings)
    # env = None
    agent = PPO('MultiInputPolicy', env, verbose=1, device='cuda')
    env.close()
    test_env, num_envs = build_env(True, sb3=sb3, render_mode='rgb_array', all_settings=all_settings)
    # print(agent.policy.action_net)
    # Train the agent
    # for _ in range(200):

    model_folder = './ckpts/'
    model_checkpoint = 'PPO'
    new_model_checkpoint = model_checkpoint + "_final"
    # new_model_checkpoint = model_checkpoint + "_autosave_50000"
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.load(model_path)
    record_single_video(test_env,
                        agent, video_folder='./videos/', env_id=model_checkpoint)

#!/usr/bin/env python3
import os

import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings, \
    RecordingSettings
from diambra.arena.stable_baselines3.sb3_utils import AutoSave, linear_schedule
from diambra.engine import SpaceTypes
from stable_baselines3.common.vec_env import SubprocVecEnv

from settings import all_settings
from utils import record_video, build_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def main(all_settings):
    # Settings
    # settings = EnvironmentSettings()
    # settings.frame_shape = (112, 192, 1)
    # settings.step_ratio = 3  # action every 3 frames
    # settings.difficulty = 1
    # settings.characters = 'Ken'
    # # settings.frame_shape = (224, 384, 1)
    # # settings.hardcore = True
    # # Wrappers Settings
    # wrappers_settings = WrappersSettings()
    # wrappers_settings.normalize_reward = True
    # wrappers_settings.stack_frames = 10
    # wrappers_settings.scale = True
    # wrappers_settings.exclude_image_scaling = True
    # wrappers_settings.flatten = True
    # wrappers_settings.filter_keys = []
    # wrappers_settings.role_relative = True
    # wrappers_settings.add_last_action = True

    # Environment creation
    # env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode="human")
    # Create environment
    # env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings, render_mode="human")
    # print('Rank:', env.env_settings.rank)
    # print("Activated {} environment(s)".format(num_envs))
    # print('>>>action space:', env.action_space, int(len(env.action_space.nvec)))
    # print('>>>obs space:', env.observation_space)
    env, num_envs = build_env(False, sb3=True, render_mode='human', all_settings=all_settings)
    # Instantiate the agent
    # policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
    # agent = PPO(CustomActorCriticPolicy, env, verbose=1)
    # agent = PPO('MultiInputPolicy', env, verbose=1,
    #             device='cuda',
    #             n_steps=1024,
    #             batch_size=512, # 512,
    #             n_epochs=10,
    #             gamma=0.94)
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    model_folder = all_settings['agent']['model_folder']
    model_checkpoint = all_settings['agent']['model_checkpoint']
    autosave_freq = all_settings['agent']['autosave_freq']
    time_steps = all_settings['agent']['time_steps']


    agent = PPO('MultiInputPolicy', env, verbose=1, device='cuda',
                learning_rate=lr_schedule,
                clip_range=clip_range_schedule,
                n_steps=4096,
                # n_steps=1024,
                batch_size=1024,  # 512,
                n_epochs=10,
                gamma=0.94,
                tensorboard_log=all_settings['agent']['log_dir']
                )
    # print(agent.policy.action_net)
    # Train the agent
    # for _ in range(200):
    # Create the callback: autosave every USER DEF steps

    auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                  save_path=model_folder, filename_prefix=model_checkpoint + "_")

    # Train the agent

    agent.learn(total_timesteps=time_steps, progress_bar=True, callback=auto_save_callback)

    # Save the agent
    # new_model_checkpoint = str(int(model_checkpoint) + time_steps)
    new_model_checkpoint = model_checkpoint + "_final"
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # rank = env.env_settings.rank
    # if rank == 0:
    # record_video(agent.get_env(),
    #              agent, video_folder='./videos/', env_id=model_checkpoint, num_envs=num_envs)
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10, render=True)
    print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))


if __name__ == '__main__':
    main(all_settings)

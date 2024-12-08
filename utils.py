import os
import time
from typing import List, OrderedDict

import cv2
import diambra
import numpy as np
from diambra.arena import EnvironmentSettings, WrappersSettings, load_settings_flat_dict, RecordingSettings
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv

from action_wrap import MultiDiscreteToDiscreteWrapper, ComboWrapper


def convert2order(d):
    k = list(d.keys())
    k.sort()
    result = OrderedDict()
    for i in k:
        result[i] = d[i]
    return result

def make_sb3_env_new(game_id: str, env_settings: EnvironmentSettings=EnvironmentSettings(),
                 wrappers_settings: WrappersSettings=WrappersSettings(),
                 episode_recording_settings: RecordingSettings=RecordingSettings(),
                 render_mode: str="rgb_array", seed: int=None, start_index: int=0,
                 allow_early_resets: bool=True, start_method: str=None, no_vec: bool=False,
                 use_subprocess: bool=True, log_dir_base: str="/tmp/DIAMBRALog/"):
    """
    Create a wrapped, monitored VecEnv.
    :param game_id: (str) the game environment ID
    :param env_settings: (EnvironmentSettings) parameters for DIAMBRA Arena environment
    :param wrappers_settings: (WrappersSettings) parameters for environment wrapping function
    :param episode_recording_settings: (RecordingSettings) parameters for environment recording wrapping function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses. See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv`
    :param no_vec: (bool) Whether to avoid usage of Vectorized Env or not. Default: False
    :return: (VecEnv) The diambra environment
    """

    env_addresses = os.getenv("DIAMBRA_ENVS", "").split()
    if len(env_addresses) == 0:
        raise Exception("ERROR: Running script without DIAMBRA CLI.")

    num_envs = len(env_addresses)

    def _make_sb3_env(rank, seed):
        # Seed management
        env_settings.seed = int(time.time()) if seed is None else seed
        env_settings.seed += rank

        def _init():
            env = diambra.arena.make(game_id, env_settings, wrappers_settings,
                                     episode_recording_settings, render_mode, rank=rank)

            # Create log dir
            log_dir = os.path.join(log_dir_base, str(rank))
            os.makedirs(log_dir, exist_ok=True)
            env = MultiDiscreteToDiscreteWrapper(env)
            env = ComboWrapper(env)
            env = Monitor(env, log_dir, allow_early_resets=allow_early_resets)
            return env
        set_random_seed(env_settings.seed)
        return _init

    # If not wanting vectorized envs
    if no_vec and num_envs == 1:
        env = _make_sb3_env(0, seed)()
    else:
        # When using one environment, no need to start subprocesses
        if num_envs == 1 or not use_subprocess:
            env = DummyVecEnv([_make_sb3_env(i + start_index, seed) for i in range(num_envs)])
        else:
            env = SubprocVecEnv([_make_sb3_env(i + start_index, seed) for i in range(num_envs)],
                                start_method=start_method)

    return env, num_envs

def build_env(no_resize: bool = False, sb3: bool = True, render_mode=None, all_settings=None, test=False):
    # Settings

    # settings = EnvironmentSettings()
    # if not no_resize:
    #     settings.frame_shape = all_settings['basic']['frame_shape']
    settings = load_settings_flat_dict(EnvironmentSettings, all_settings['basic'])
    if test:
        settings.continue_game = 0.0
    if no_resize:
        settings.frame_shape = (0, 0, 0)
    # settings.step_ratio = all_settings['basic']['step_ratio']  # action every 3 frames
    # settings.difficulty = all_settings['basic']['difficulty']
    # settings.characters = all_settings['basic']['characters']
    # settings.frame_shape = (224, 384, 1)
    # settings.hardcore = True
    # Wrappers Settings
    # wrappers_settings = WrappersSettings()
    # wrappers_settings.normalize_reward = all_settings['wrapper']['normalize_reward']
    # wrappers_settings.normalization_factor = all_settings['wrapper']['normalization_factor']
    # wrappers_settings.stack_frames = all_settings['wrapper']['stack_frames']
    # wrappers_settings.stack_actions = all_settings['wrapper']['stack_actions']
    # wrappers_settings.scale = all_settings['wrapper']['scale']
    # wrappers_settings.exclude_image_scaling = all_settings['wrapper']['exclude_image_scaling']
    # wrappers_settings.flatten = all_settings['wrapper']['flatten']
    # wrappers_settings.filter_keys = all_settings['wrapper']['filter_keys']
    # wrappers_settings.role_relative = all_settings['wrapper']['role_relative']
    # wrappers_settings.add_last_action = all_settings['wrapper']['add_last_action']
    wrappers_settings = load_settings_flat_dict(WrappersSettings, all_settings['wrapper'])

    config = {"policy_type": "MultiInputPolicy"}

    # Create environment
    if sb3:
        env, num_envs = make_sb3_env_new("sfiii3n", settings, wrappers_settings, render_mode=render_mode)
    else:
        env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode=render_mode)
        num_envs = 1
        env = MultiDiscreteToDiscreteWrapper(env)
        env = ComboWrapper(env)
    return env, num_envs


def preprocess(obs, width=192, height=112):
    # TODO: Only support one environment
    frames = obs["frame"]
    fs = frames.shape[-1]
    result = []
    # print(frames.shape)
    squeeze_indicator = False
    if len(frames.shape) > 3:
        frames = frames.squeeze(axis=0)
        squeeze_indicator = True
    for i in range(0, fs, 3):
        frame = frames[:, :, i:(i + 3)]
        # apply grayscaled first, then resize
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)[:, :, None]
        result.append(frame.squeeze(-1))
    frames = np.stack(result, axis=-1)
    if squeeze_indicator:
        frames = frames[np.newaxis, :]
    # print(frames.shape)
    obs["frame"] = frames
    return frames



class CustomVecVideoRecorder(VecVideoRecorder):
    def _capture_frame(self) -> None:
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        # print(type(frame), frame.shape)
        if isinstance(frame, List):
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self._stop_recording()
            # logger.warn(
            #     f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
            # )

def record_video(env, agent, video_folder, video_length=10240, env_id=None, num_envs=1):
    env = CustomVecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"trained-agent-{env_id}"
    )

    # env.reset()
    observation = env.reset()
    cumulative_reward = [0.0 for _ in range(num_envs)]

    while True:
        action, _state = agent.predict(observation, deterministic=True)  # 获取动作
        observation, reward, done, info = env.step(action)
        cumulative_reward = [cr + r for cr, r in zip(cumulative_reward, reward)]  # 累计奖励
        # cumulative_reward += reward

        # if any(r != 0 for r in reward):
        # print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            observation = env.reset()
            break

    env.close()


def record_single_video(env, agent, video_folder, video_length=10240, env_id=None, all_settings=None, episodes=1):
    # env = DummyVecEnv([lambda: env])
    if not isinstance(env, DummyVecEnv) and not isinstance(env, SubprocVecEnv):
        env = DummyVecEnv([lambda: env])
    env = CustomVecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length * episodes,
        name_prefix=f"trained-agent-{env_id}"
    )

    # env.reset()
    observation = env.reset()
    states = None
    cumulative_reward = 0.0
    height = all_settings['basic']['frame_shape'][0]
    width = all_settings['basic']['frame_shape'][1]
    episode_count = 0

    while episode_count < episodes:
        preprocess(observation, width=width, height=height)
        observation = convert2order(observation)
        action, _ = agent.predict(
            observation,
            # state=states,
            deterministic=True)
        # print(action, type(action))
        # action = [[9, 1]]
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward

        print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            # observation = env.reset()
            # break
            episode_count += 1
            cumulative_reward = 0

    env.close()

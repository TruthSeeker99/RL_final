from typing import List, OrderedDict

import cv2
import diambra
import numpy as np
from diambra.arena import EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv


def convert2order(d):
    k = list(d.keys())
    k.sort()
    result = OrderedDict()
    for i in k:
        result[i] = d[i]
    return result

def build_env(no_resize: bool = False, sb3: bool = True, render_mode=None, all_settings=None):
    # Settings

    settings = EnvironmentSettings()
    if not no_resize:
        settings.frame_shape = all_settings['basic']['frame_shape']
    settings.step_ratio = all_settings['basic']['step_ratio']  # action every 3 frames
    settings.difficulty = all_settings['basic']['difficulty']
    settings.characters = all_settings['basic']['characters']
    # settings.frame_shape = (224, 384, 1)
    # settings.hardcore = True
    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = all_settings['wrapper']['normalize_reward']
    wrappers_settings.stack_frames = all_settings['wrapper']['stack_frames']
    wrappers_settings.scale = all_settings['wrapper']['scale']
    wrappers_settings.exclude_image_scaling = all_settings['wrapper']['exclude_image_scaling']
    wrappers_settings.flatten = all_settings['wrapper']['flatten']
    wrappers_settings.filter_keys = all_settings['wrapper']['filter_keys']
    wrappers_settings.role_relative = all_settings['wrapper']['role_relative']
    wrappers_settings.add_last_action = all_settings['wrapper']['add_last_action']

    config = {"policy_type": "MultiInputPolicy"}

    # Create environment
    if sb3:
        env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings, render_mode=render_mode)
    else:
        env = diambra.arena.make("sfiii3n", settings, wrappers_settings, render_mode=render_mode)
        num_envs = 1
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


def record_single_video(env, agent, video_folder, video_length=10240, env_id=None):
    # env = DummyVecEnv([lambda: env])
    if not isinstance(env, DummyVecEnv) and not isinstance(env, SubprocVecEnv):
        env = DummyVecEnv([lambda: env])
    env = CustomVecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"trained-agent-{env_id}"
    )

    # env.reset()
    observation = env.reset()
    states = None
    cumulative_reward = 0.0

    while True:
        preprocess(observation)
        observation = convert2order(observation)
        action, states = agent.predict(
            observation,
            state=states,
            deterministic=True)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward

        print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            # observation = env.reset()
            break

    env.close()

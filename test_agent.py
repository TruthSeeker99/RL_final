import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from settings import all_settings
from utils import preprocess, convert2order, build_env


def test_agent(env, agent, seed=42, width=128, height=128):
    # Environment reset
    observation = env.reset()[0]
    # Agent-Environment interaction loop
    while True:
        preprocess(observation)
        observation = convert2order(observation)
        # (Optional) Environment rendering
        env.render()

        # Action random sampling
        actions, _state = agent.predict(observation, deterministic=True)
        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        if reward != 0:
            print(reward)
        # Episode end (Done condition) check
        if terminated or truncated:
            # observation = env.reset()
            break

    # # Environment shutdown
    # env.close()

    # Return success
    return 0


if __name__ == '__main__':
    env, num_envs = build_env(False, sb3=True, render_mode='human', all_settings=all_settings)
    # env = None
    agent = PPO('MultiInputPolicy', env, verbose=1, device='cuda')
    # env.close()
    # test_env, num_envs = build_env(True, sb3=False, render_mode='human', all_settings=all_settings)
    # print(agent.policy.action_net)
    # Train the agent
    # for _ in range(200):

    model_folder = './ckpts/'
    model_checkpoint = 'PPO'
    new_model_checkpoint = model_checkpoint + "_final"
    # new_model_checkpoint = model_checkpoint + "_autosave_50000"
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.load(model_path)
    # test_agent(test_env, agent, seed=42)
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10, render=True)
    print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

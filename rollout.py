from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from eval import MetricCallback  # your callback
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from env import WrappedVacuumEnv, VacuumEnv
import matplotlib.pyplot as plt
from gymnasium.spaces.utils import flatten
import os


register(
    id="VacuumEnv-v0",
    entry_point="env:VacuumEnv",
    kwargs={"grid_size": (20, 20), 
    "render_mode": "plot"},
)

# Load model
model = PPO.load("./logs/ppo_20*20_wall_tuned/models/best_model.zip")

eval_factory = WrappedVacuumEnv(grid_size=[20, 20], dirt_num=0, max_steps=3000, walls=[], algo='ppo')
eval_env = DummyVecEnv([eval_factory])
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)

eval_base_env = eval_factory.base_env
eval_env = VecNormalize.load("logs/ppo_20*20_wall_tuned/vec_normalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

def rollout_and_save_last_frame(env, model, filename="last_frame.png", max_steps=100, walls=None, dir_name="./logs/ppo", algo='ppo'):
    obs, _ = env.reset(options={"walls": walls})
    last_frame = None
    total_reward = 0
    episode_length = 0

    if isinstance(obs, dict):
        if algo == 'dqn':
            obs = flatten(env.observation_space, obs)

    for step in range(max_steps):
        last_frame = env.unwrapped.render_frame()

        try:
            action, _ = model.predict(obs)
        except:
            action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_length += 1

        if terminated or truncated:
            break
    print(env.unwrapped.compute_metrics()["coverage_rate"])

    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    plt.figure()
    plt.imshow(last_frame)
    plt.axis("off")
    plt.savefig(os.path.join(dir_name, filename), bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Last frame saved to {filename}")

    # Example of custom metric from `info`
    success = info.get("success", None)

    # Return metrics
    return {
        "total_reward": total_reward,
        "episode_length": episode_length,
        "success": success,
    }

rollout_and_save_last_frame(eval_base_env, model, filename="test.png", walls=[])

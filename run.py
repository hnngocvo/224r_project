import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
from gymnasium.spaces.utils import flatten

import os

from env import VacuumEnv
from eval import evaluate_random_agent_steps, RandomAgent
from eval import export_random_metrics
from eval import evaluate_model, save_metrics_with_summary, evaluate_vec_model
from eval import MetricWrapper, MetricCallback
from env import WrappedVacuumEnv

import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --------------------------------------
# Register the robot vacuum environment
# --------------------------------------
register(
    id="VacuumEnv-v0",
    entry_point="env:VacuumEnv",
    kwargs={"grid_size": (20, 20), 
    "render_mode": "plot"},
)

# --------------------------------------
# DQN Callback for logging
# --------------------------------------
class DQNLoggingCallback(BaseCallback):
    def __init__(self, verbose=1, log_freq=5000):
        super().__init__(verbose)
        self.episode_rewards = []
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Collect episode rewards from Monitor-wrapped env
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                self.episode_rewards.append(reward)
                if self.verbose > 0:
                    print(f"[Episode] Reward: {reward:.2f} | Length: {length}")

        # Log every `log_freq` timesteps
        if self.num_timesteps % self.log_freq == 0:
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else float("nan")
            train_loss = self.model.logger.name_to_value.get("train/loss")

            # Print to terminal
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps}] Avg Reward (last 10): {avg_reward:.2f} | Loss: {train_loss}")

            # Log to TensorBoard
            self.logger.record("rollout/ep_rew_mean", avg_reward)
            if train_loss is not None:
                self.logger.record("train/loss", train_loss)
            self.logger.dump(self.num_timesteps)

        return True

# --------------------------------------
# Fixed wall layout generator
# --------------------------------------
def generate_1b1b_layout_grid():
    wall_positions = set()

    for x in [5, 25, 32, 35]:
        for y in range(3):
            wall_positions.add((x, y))

    for x in range(5, 27):
        wall_positions.add((x, 12))

    wall_positions.add((25, 7))
    wall_positions.add((25, 8))
    wall_positions.add((25, 11))

    for y in range(7, 13):
        wall_positions.add((28, y))
    wall_positions.add((27, 12))
    wall_positions.add((25, 7))
    wall_positions.add((26, 7))
    wall_positions.add((27, 7))

    for y in [10, 15, 20]:
        for x in range(36, 41):
            wall_positions.add((x, y))

    wall_positions.add((32, 7))
    wall_positions.add((32, 8))
    wall_positions.add((32, 9))
    wall_positions.add((32, 10))
    wall_positions.add((33, 10))

    for x in [22, 25, 28]:
        for y in range(11, 18):
            wall_positions.add((x, y))

    for x in range(20, 28):
        wall_positions.add((x, 17))
    
    # remove the living room closest doors
    wall_positions.discard((23, 17))
    wall_positions.discard((24, 17))
    wall_positions.discard((28, 14))
    wall_positions.discard((28, 15))

    # living room and dinning room blocker
    for x in [25, 30, 31, 35]:
        for y in range(27, 31):
            wall_positions.add((x, y))

    return list(wall_positions)


# --------------------------------------
# Alternate fixed wall layout for evaluation
# --------------------------------------
def generate_eval_layout_grid():
    wall_positions = set()

    for x in range(2, 4):
        for y in range(6, 9):
            wall_positions.add((x, y))

    for x in range(32, 36):
        wall_positions.add((x, 3))
    for y in range(2, 6):
        wall_positions.add((34, y))
    wall_positions.discard((33, 3))
    wall_positions.discard((32, 3))

    for x in range(4, 9):
        wall_positions.add((x, 26))
    for y in range(25, 29):
        wall_positions.add((4, y))
    wall_positions.discard((6, 26))
    wall_positions.discard((7, 26))

    for x in range(30, 34):
        wall_positions.add((x, 27))
    wall_positions.add((32, 28))

    for y in range(10, 20):
        wall_positions.add((8, y))
    wall_positions.discard((8, 14))
    wall_positions.discard((8, 15))

    for x in range(10, 30):
        wall_positions.add((x, 5))
    wall_positions.discard((17, 5))
    wall_positions.discard((18, 5))
    wall_positions.discard((19, 5))

    for x in range(12, 25):
        wall_positions.add((x, 24))
    wall_positions.discard((16, 24))
    wall_positions.discard((17, 24))

    wall_positions.discard((0, 0))

    return list(wall_positions)

# --------------------------------------
# Rollout and save animation
# --------------------------------------
def rollout_and_record(env, model, filename="vacuum_run.mp4", max_steps=100, walls=None, algo='dqn'):
    obs, _ = env.reset(options={"walls": walls})
    frames = []
    if isinstance(obs, dict):
        if algo == 'dqn':
            obs = flatten(env.observation_space, obs)

    if isinstance(obs, dict):
        obs = FlattenObservation(env).observation(obs)

    for _ in range(max_steps):
        fig = env.unwrapped.render_frame()
        frames.append(fig)

        try:
            action, _ = model.predict(obs)
        except:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    # Save animation
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=100, blit=True
    )

    ani.save(filename, writer="ffmpeg")
    plt.close(fig)
    print(f"Video saved to {filename}")

def rollout_and_save_last_frame(env, model, filename="last_frame.png", max_steps=100, walls=None, dir_name = "logs", algo='ppo'):
    obs, _ = env.reset(options={"walls": walls})
    last_frame = None

    if isinstance(obs, dict):
        if algo == 'dqn':
            obs = flatten(env.observation_space, obs)

    for _ in range(max_steps):
        last_frame = env.unwrapped.render_frame()  # Only keep the latest frame

        try:
            action, _ = model.predict(obs)
        except:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break
    
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)

    plt.figure()
    plt.imshow(last_frame)
    plt.axis("off")
    plt.savefig(os.path.join(dir_name, filename), bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Last frame saved to {filename}")


if __name__ == "__main__":
    # ------------------------------
    # Parse command-line arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description="Train PPO or DQN on VacuumEnv")
    parser.add_argument("--algo", choices=["ppo", "dqn", "her", "random"], default="ppo", help="RL algorithm to run")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Number of training timesteps")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[20, 20],
                        help="Grid size as two integers (e.g., 40 30)")
    parser.add_argument("--wall_mode", choices=["random", "hardcoded", "none"], default="random",
                        help="Wall layout: 'none', 'random' or 'hardcoded' (only applies to 40x30)")
    parser.add_argument("--dirt_num", type=float, default=0,
                        help="Number of dirt clusters; 0 for all dirty")
    args = parser.parse_args()

    algo = args.algo
    total_timesteps = args.timesteps
    grid_size = tuple(args.grid_size)
    wall_mode = args.wall_mode
    dirt_num = args.dirt_num

    # Determine wall layout for training and eval
    if wall_mode == "hardcoded" and grid_size == (40, 30):
        walls = generate_1b1b_layout_grid()
        eval_walls = generate_eval_layout_grid()
    elif wall_mode == "random":
        walls = []
        eval_walls = []
    else:
        walls = None
        eval_walls = None

    # Load best hyperparameters if available
    param_path = Path(f"optuna_results/{algo}_best_params.json")
    if param_path.exists():
        with open(param_path, "r") as f:
            best_params = json.load(f)
    else:
        best_params = {}

    if algo == "random":
        # ---------------
        # Random baseline
        # ---------------
        max_steps = 3000

        base_env = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot", dirt_num=dirt_num)
        base_env = TimeLimit(base_env, max_episode_steps=max_steps)
        base_env = ExplorationBonusWrapper(base_env, bonus=0.3)
        base_env = ExploitationPenaltyWrapper(base_env, time_penalty=-0.002, stay_penalty=-0.1)

        # monitor for stats logging
        base_env = MetricWrapper(base_env)

        # reset before training
        obs, _ = base_env.reset(options={"walls": eval_walls})
        check_env(base_env, warn=True)

        # evaluate the random agent
        random_agent = RandomAgent(base_env.action_space)
        metrics = evaluate_random_agent_steps(random_agent, base_env)
        export_random_metrics(metrics, "best_param/random")
        with open("random_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        rollout_and_record(base_env.unwrapped, random_agent, filename="random_agent.mp4", max_steps=max_steps, walls=eval_walls)

    elif algo == "ppo":
        # --------------------------------------
        # PPO Training with wrappers and Monitor
        # --------------------------------------
        max_steps = 3000

        train_factory = WrappedVacuumEnv(grid_size=grid_size, dirt_num=dirt_num, max_steps=max_steps, walls=walls, algo=algo)
        train_env = DummyVecEnv([train_factory])
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

        eval_factory = WrappedVacuumEnv(grid_size=grid_size, dirt_num=dirt_num, max_steps=max_steps, walls=walls, algo=algo)
        eval_env = DummyVecEnv([eval_factory])
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)

        base_env = train_factory.base_env
        eval_base_env = eval_factory.base_env

        # reset before training
        #obs, _ = base_env.reset(options={"walls": walls})
        #obs, _ = eval_base_env.reset(options={"walls": walls})
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./logs/ppo/models/",
            log_path="./logs/ppo/",
            eval_freq=10000,             # Evaluate every N steps
            deterministic=True,
            render=False,
            n_eval_episodes=10,           # Evaluate using multiple episodes
        )

        check_env(base_env, warn=True)
        check_env(eval_base_env, warn=True)

        # PPO agent with best hyperparameters if available
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            **best_params
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, MetricCallback()],
        )

        # Save VecNormalize stats
        train_env.save("logs/ppo/vec_normalize.pkl")

        # Load stats into eval env
        eval_env = VecNormalize.load("logs/ppo/vec_normalize.pkl", eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        # Save the final trajectory
        print("Saving PPO training trajectory...")
        rollout_and_save_last_frame(base_env, model, filename="ppo_train.png", max_steps=3000, walls=walls, dir_name="./logs/ppo", algo='ppo')

        # Save best eval trajectory
        print("Saving PPO eval trajectory...")
        rollout_and_save_last_frame(eval_base_env, model, filename="ppo_eval.png", max_steps=3000, walls=eval_walls, dir_name = "./logs/ppo", algo='ppo')
        #rollout_and_record(eval_env, model, filename="ppo_eval.mp4", max_steps=3000, walls=eval_walls)

        # Save to file with summary
        #metrics = evaluate_vec_model(model, eval_env, n_episodes=20)
        #save_metrics_with_summary(metrics, output_path="./logs/ppo/evaluation_metrics.json")


    elif algo == "dqn":
        # --------------------------------------
        # DQN Training with wrappers and Monitor
        # --------------------------------------
        max_steps = 3000

        train_factory = WrappedVacuumEnv(grid_size=grid_size, dirt_num=dirt_num, max_steps=max_steps, walls=walls, algo=algo)
        train_env = DummyVecEnv([train_factory])
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

        eval_factory = WrappedVacuumEnv(grid_size=grid_size, dirt_num=dirt_num, max_steps=max_steps, walls=walls, algo=algo)
        eval_env = DummyVecEnv([eval_factory])
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)

        base_env = train_factory.base_env
        eval_base_env = eval_factory.base_env

        check_env(base_env, warn=True)
        check_env(eval_base_env, warn=True)

        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            **best_params
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=MetricCallback(),
        )

        # Save VecNormalize stats
        train_env.save("logs/dqn/vec_normalize.pkl")

        # Load stats into eval env
        eval_env = VecNormalize.load("logs/dqn/vec_normalize.pkl", eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        # Save the final trajectory
        print("Saving DQN training trajectory...")
        rollout_and_save_last_frame(base_env, model, filename="dqn_train.png", max_steps=3000, walls=walls, dir_name="./logs/dqn", algo='dqn')

        # Save best eval trajectory
        print("Saving DQN eval trajectory...")
        rollout_and_save_last_frame(eval_base_env, model, filename="dqn_eval.png", max_steps=3000, walls=eval_walls, dir_name = "./logs/dqn", algo='dqn')
        #rollout_and_record(eval_env, model, filename="ppo_eval.mp4", max_steps=3000, walls=eval_walls)

        # Save to file with summary
        metrics = evaluate_vec_model(model, eval_env, n_episodes=20)
        save_metrics_with_summary(metrics, output_path="./logs/dqn/evaluation_metrics.json")
    
    elif algo == "her":
        # -----------------------------------------------
        # DQN with HER Training with wrappers and Monitor
        # -----------------------------------------------
        max_steps = 500

        base_env = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot", dirt_num=dirt_num)
        base_env = TimeLimit(base_env, max_episode_steps=max_steps)
        base_env = ExplorationBonusWrapper(base_env, bonus=0.3)
        base_env = ExploitationPenaltyWrapper(base_env, time_penalty=-0.002, stay_penalty=-0.1)

        # Wrap with Goal Wrapper for HER
        goal_env = VacuumGoalWrapper(base_env)
        monitored_env = MetricWrapper(goal_env)

        # Eval env (same setup)
        eval_base = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot", dirt_num=dirt_num)
        eval_base = TimeLimit(eval_base, max_episode_steps=max_steps)
        eval_base = ExplorationBonusWrapper(eval_base, bonus=0.3)
        eval_base = ExploitationPenaltyWrapper(eval_base, time_penalty=-0.002, stay_penalty=-0.1)
        eval_env = MetricWrapper(VacuumGoalWrapper(eval_base))

        # Set fixed wall layout
        obs, _ = goal_env.reset(options={"walls": walls})
        obs, _ = eval_env.reset(options={"walls": eval_walls})
        check_env(monitored_env, warn=True)
        check_env(eval_env, warn=True)

        # default DQN uses Double DQN already
        model = DQN(
            "MultiInputPolicy",
            monitored_env,
            verbose=1,
            tensorboard_log="./tensorboard_dqn_her/",
            learning_starts=1000,
            train_freq=4,
            target_update_interval=1000,
            replay_buffer_class=HerReplayBufferForDQN,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
                env=goal_env,
            ),
            **best_params
        )

        # Training
        model.learn(
            total_timesteps=total_timesteps,
            callback=MetricCallback()
        )

        # Save training run video
        print("Saving final training trajectory...")
        rollout_and_record(monitored_env.unwrapped, model, filename="her_train.mp4", max_steps=3000, walls=walls)

        # Save evaluation run video
        print("Saving evaluation trajectory...")
        rollout_and_record(eval_env.unwrapped, model, filename="dqn_her_eval.mp4", max_steps=3000, walls=eval_walls)

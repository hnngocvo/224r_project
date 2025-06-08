import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from typing import Dict, List
import json
import os

class MetricCallback(BaseCallback):
    def __init__(self, verbose=1, log_dir="./logs/"):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            "coverage_ratio": [],
            "path_efficiency": [],
            "revisit_ratio": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "cleaning_time": []
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.metrics["episode_rewards"].append(info["episode"]["r"])
                self.metrics["episode_lengths"].append(info["episode"]["l"])
            
            if "coverage_ratio" in info:
                self.metrics["coverage_ratio"].append(info["coverage_ratio"])
                self.metrics["path_efficiency"].append(info["path_efficiency"])
                self.metrics["revisit_ratio"].append(info["revisit_ratio"])
                if "cleaning_time" in info:
                    self.metrics["cleaning_time"].append(info["cleaning_time"])

                # Log to tensorboard
                self.logger.record("metrics/coverage_ratio", info["coverage_ratio"])
                self.logger.record("metrics/path_efficiency", info["path_efficiency"])
                self.logger.record("metrics/revisit_ratio", info["revisit_ratio"])
                if "cleaning_time" in info:
                    self.logger.record("metrics/cleaning_time", info["cleaning_time"])

        return True

    def _on_training_end(self) -> None:
        # Save metrics to file
        metrics_file = os.path.join(self.log_dir, "training_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f)

        # Plot metrics
        self._plot_metrics()

    def _plot_metrics(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Training Metrics")

        # Plot episode rewards
        axes[0, 0].plot(self.metrics["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        # Plot episode lengths
        axes[0, 1].plot(self.metrics["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")

        # Plot coverage ratio
        axes[0, 2].plot(self.metrics["coverage_ratio"])
        axes[0, 2].set_title("Coverage Ratio")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Ratio")

        # Plot path efficiency
        axes[1, 0].plot(self.metrics["path_efficiency"])
        axes[1, 0].set_title("Path Efficiency")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Efficiency")

        # Plot revisit ratio
        axes[1, 1].plot(self.metrics["revisit_ratio"])
        axes[1, 1].set_title("Revisit Ratio")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Ratio")

        # Plot cleaning time if available
        if self.metrics["cleaning_time"]:
            axes[1, 2].plot(self.metrics["cleaning_time"])
            axes[1, 2].set_title("Cleaning Time")
            axes[1, 2].set_xlabel("Episode")
            axes[1, 2].set_ylabel("Steps")

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_metrics.png"))
        plt.close()

def compute_coverage_ratio(env):
    total_dirt = np.sum(env.dirt_map == 1) + np.sum(env.cleaned_map == 1)
    cleaned = np.sum(env.cleaned_map == 1)
    return cleaned / total_dirt if total_dirt > 0 else 0

def compute_redundancy_rate(env):
    path_visits = env.path_map[env.obstacle_map == 0]
    return np.mean(path_visits) if path_visits.size > 0 else 0

def compute_revisit_ratio(env):
    """Calculate the ratio of revisits to total steps.
    A revisit is counted when a cell is visited more than once."""
    # Convert to float64 to avoid overflow
    path_map = env.path_map.astype(np.float64) if hasattr(env, 'path_map') else env.unwrapped.path_map.astype(np.float64)
    total_steps = np.sum(path_map)
    
    if total_steps == 0:
        return 0.0
    
    # Calculate total revisits
    # For each cell, count how many times it was visited beyond the first visit
    revisits = np.sum(np.maximum(path_map - 1, 0))
    
    # Revisit ratio is the number of revisit steps divided by total steps
    return revisits / total_steps

class MetricWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.metrics = {}
        self.env = env
        self.base_env = env.unwrapped # env.unwrapped

    def reset(self, **kwargs):
        self.metrics = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info_from_lower_env = self.env.step(action)

        if terminated or truncated:
            path_map_from_info = info_from_lower_env.get("path_map")
            path_map_from_direct_access = self.base_env.path_map

            print(f"[MetricWrapper DEBUG] id(self.base_env): {id(self.base_env)}")
            if path_map_from_info is not None:
                print(f"  path_map from info_from_lower_env: sum: {np.sum(path_map_from_info)}, max: {np.max(path_map_from_info)}, shape: {path_map_from_info.shape}")
            else:
                print("  path_map NOT FOUND in info_from_lower_env.")
            print(f"  path_map from direct_access (self.base_env.path_map): sum: {np.sum(path_map_from_direct_access)}, max: {np.max(path_map_from_direct_access)}, shape: {path_map_from_direct_access.shape}")
            
            # The crucial part: metrics are computed using self.base_env
            # So, the state of path_map_from_direct_access is what determines the computed metrics.
            
            # Update info with the path_map that will be used for metrics (direct access one)
            info_from_lower_env["path_map"] = path_map_from_direct_access 
            
            self.metrics['coverage_ratio'] = compute_coverage_ratio(self.base_env)
            self.metrics['path_efficiency'] = compute_redundancy_rate(self.base_env)
            self.metrics['revisit_ratio'] = compute_revisit_ratio(self.base_env)
            info_from_lower_env.update(self.metrics)

        return obs, reward, terminated, truncated, info_from_lower_env

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

def evaluate_model(model, env, n_episodes=10, render=False) -> Dict[str, List[float]]:
    """
    Evaluate a trained model and return detailed metrics
    """
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "coverage_ratio": [],
        "path_efficiency": [],
        "revisit_ratio": [],
        "cleaning_time": []
    }

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if done:
                metrics["episode_rewards"].append(episode_reward)
                metrics["episode_lengths"].append(episode_length)
                metrics["coverage_ratio"].append(info.get("coverage_ratio", 0))
                metrics["path_efficiency"].append(info.get("path_efficiency", 0))
                metrics["revisit_ratio"].append(info.get("revisit_ratio", 0))
                if "cleaning_time" in info:
                    metrics["cleaning_time"].append(info["cleaning_time"])

                print(f"Episode {episode + 1}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Length: {episode_length}")
                print(f"Coverage: {info.get('coverage_ratio', 0):.2%}")
                print(f"Path Efficiency: {info.get('path_efficiency', 0):.2f}")
                print(f"Revisit Ratio: {info.get('revisit_ratio', 0):.2f}")
                if "cleaning_time" in info:
                    print(f"Cleaning Time: {info['cleaning_time']}")
                print("---")

    return metrics

def save_metrics_with_summary(metrics, output_path):
    """
    Save both raw metrics and summary statistics to JSON file.
    """
    summary = {}
    for key, values in metrics.items():
        values_np = np.array(values, dtype=np.float32)
        for key in metrics:
            metrics[key] = [float(v) for v in metrics[key]]
        summary[key] = {
            "mean": float(np.mean(values_np)) if len(values_np) > 0 else 0.0,
            "std": float(np.std(values_np)) if len(values_np) > 0 else 0.0
        }

    result = {
        "raw_metrics": metrics,
        "summary_statistics": summary
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Metrics saved to {output_path}")

def evaluate_vec_model(model, env, n_episodes=10, render=False) -> Dict[str, List[float]]:
    """
    Evaluate a trained model and return detailed metrics.
    Works with VecNormalize + DummyVecEnv.
    """
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "coverage_ratio": [],
        "path_efficiency": [],
        "revisit_ratio": [],
        "cleaning_time": []
    }

    for episode in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # remove info if present

        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step([action])

            # Unpack from VecEnv (batched)
            obs = obs[0]
            reward = reward[0]
            done = done[0]
            info = info[0]

            episode_reward += reward
            episode_length += 1

            if done:
                metrics["episode_rewards"].append(episode_reward)
                metrics["episode_lengths"].append(episode_length)
                metrics["coverage_ratio"].append(info.get("coverage_ratio", 0))
                metrics["path_efficiency"].append(info.get("path_efficiency", 0))
                metrics["revisit_ratio"].append(info.get("revisit_ratio", 0))
                if "cleaning_time" in info:
                    metrics["cleaning_time"].append(info["cleaning_time"])

                is_truncated = info.get("TimeLimit.truncated", False)
                print(f"Episode done. Truncated: {is_truncated}")

                print(f"Episode {episode + 1}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Length: {episode_length}")
                print(f"Coverage: {info.get('coverage_ratio', 0):.2%}")
                print(f"Path Efficiency: {info.get('path_efficiency', 0):.2f}")
                print(f"Revisit Ratio: {info.get('revisit_ratio', 0):.2f}")
                if "cleaning_time" in info:
                    print(f"Cleaning Time: {info['cleaning_time']}")
                print("---")

    return metrics


class RandomAgent:
    """Simple random agent for baseline comparison"""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample(), None
    
def evaluate_random_agent_steps(agent, env, max_total_steps=500_000, max_episode_steps=3000) -> Dict[str, List[float]]:
    """
    Run random agent for up to `max_total_steps` across multiple episodes,
    limiting each episode to `max_episode_steps`, and collect evaluation metrics.
    """
    total_steps = 0

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "coverage_ratio": [],
        "path_efficiency": [],
        "revisit_ratio": [],
        "cleaning_time": []
    }

    episode = 0
    while total_steps < max_total_steps:
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done and episode_length < max_episode_steps and total_steps < max_total_steps:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            total_steps += 1

        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_length)
        metrics["coverage_ratio"].append(info.get("coverage_ratio", 0))
        metrics["path_efficiency"].append(info.get("path_efficiency", 0))
        metrics["revisit_ratio"].append(info.get("revisit_ratio", 0))
        if "cleaning_time" in info:
            metrics["cleaning_time"].append(info["cleaning_time"])

        episode += 1
        print(f"Episode {episode}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Length: {episode_length}")
        print(f"Coverage: {info.get('coverage_ratio', 0):.2%}")
        print(f"Path Efficiency: {info.get('path_efficiency', 0):.2f}")
        print(f"Revisit Ratio: {info.get('revisit_ratio', 0):.2f}")
        if "cleaning_time" in info:
            print(f"Cleaning Time: {info['cleaning_time']}")
        print("---")

    return metrics

def export_random_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(values) + 1), values)
        plt.title(metric_name)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)

        safe_name = metric_name.replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
        plt.close()

def evaluate_random_agent(agent, env, episodes=10) -> Dict[str, List[float]]:
    """
    Evaluate a random agent as baseline
    """
    return evaluate_model(agent, env, n_episodes=episodes)

def plot_comparison_metrics(rainbow_metrics: Dict[str, List[float]], 
                          random_metrics: Dict[str, List[float]], 
                          save_path: str = "comparison_metrics.png"):
    """
    Plot comparison between Rainbow DQN and random agent metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Rainbow DQN vs Random Agent")

    metrics = [
        ("episode_rewards", "Episode Rewards", "Reward"),
        ("episode_lengths", "Episode Lengths", "Steps"),
        ("coverage_ratio", "Coverage Ratio", "Ratio"),
        ("path_efficiency", "Path Efficiency", "Efficiency"),
        ("revisit_ratio", "Revisit Ratio", "Ratio"),
        ("cleaning_time", "Cleaning Time", "Steps")
    ]

    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in rainbow_metrics and metric in random_metrics:
            ax.boxplot([rainbow_metrics[metric], random_metrics[metric]], 
                      labels=['Rainbow', 'Random'])
            ax.set_title(title)
            ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


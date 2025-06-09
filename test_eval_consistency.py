#!/usr/bin/env python3
"""
Test script to compare evaluation results between:
1. Training + immediate evaluation
2. Loading saved model + evaluation

This will help identify if there are systematic differences.
"""

import subprocess
import os
import json
import sys

def run_command(cmd):
    """Run command and return the process result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None
    return result

def extract_metrics(log_dir):
    """Extract evaluation metrics from the log directory"""
    metrics_file = os.path.join(log_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def main():
    # Test configuration
    grid_size = "6 6"
    dirt_num = 5
    eval_episodes = 10
    timesteps = 50000  # Short training for quick test
    seed = 42
    temperature = 1.0
    
    print("=" * 60)
    print("TESTING EVALUATION CONSISTENCY")
    print("=" * 60)
    print(f"Grid Size: {grid_size}")
    print(f"Dirt Num: {dirt_num}")
    print(f"Eval Episodes: {eval_episodes}")
    print(f"Training Steps: {timesteps}")
    print(f"Seed: {seed}")
    print(f"Temperature: {temperature}")
    print()
    
    # Step 1: Train + Evaluate
    print("STEP 1: Training + Immediate Evaluation")
    print("-" * 40)
    train_cmd = f"python rainbow_dqn.py --mode train --grid_size {grid_size} --dirt_num {dirt_num} --eval_episodes {eval_episodes} --timesteps {timesteps} --seed {seed} --temperature {temperature}"
    
    train_result = run_command(train_cmd)
    if train_result is None:
        print("Training failed!")
        return
    
    # Find the most recent log directory
    log_dirs = [d for d in os.listdir("logs") if d.startswith("rainbow_dqn_6x6")]
    if not log_dirs:
        print("No log directories found!")
        return
    
    latest_log_dir = max([os.path.join("logs", d) for d in log_dirs], key=os.path.getctime)
    print(f"Training completed. Log dir: {latest_log_dir}")
    
    # Extract metrics from training+eval
    train_eval_metrics = extract_metrics(latest_log_dir)
    if train_eval_metrics is None:
        print("Failed to extract training+eval metrics!")
        return
    
    print("\nTraining + Eval Results:")
    avg_reward_train = sum(train_eval_metrics["episode_rewards"]) / len(train_eval_metrics["episode_rewards"])
    print(f"  Average Reward: {avg_reward_train:.3f}")
    print(f"  Episode Rewards: {train_eval_metrics['episode_rewards']}")
    
    # Step 2: Load Model + Evaluate (using best_model.pth)
    print("\n" + "=" * 60)
    print("STEP 2: Loading Model + Evaluation")
    print("-" * 40)
    
    best_model_path = os.path.join(latest_log_dir, "best_model.pth")
    eval_cmd = f"python rainbow_dqn.py --mode eval --model_path \"{best_model_path}\" --grid_size {grid_size} --dirt_num {dirt_num} --eval_episodes {eval_episodes} --temperature {temperature}"
    
    eval_result = run_command(eval_cmd)
    if eval_result is None:
        print("Evaluation failed!")
        return
    
    # Extract metrics from eval-only
    eval_only_metrics = extract_metrics(latest_log_dir)
    if eval_only_metrics is None:
        print("Failed to extract eval-only metrics!")
        return
    
    print(f"\nEval-Only completed. Results saved to: {latest_log_dir}")
    print("\nEval-Only Results:")
    avg_reward_eval = sum(eval_only_metrics["episode_rewards"]) / len(eval_only_metrics["episode_rewards"])
    print(f"  Average Reward: {avg_reward_eval:.3f}")
    print(f"  Episode Rewards: {eval_only_metrics['episode_rewards']}")
    
    # Step 3: Compare Results
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    
    reward_diff = avg_reward_eval - avg_reward_train
    reward_diff_pct = (reward_diff / abs(avg_reward_train)) * 100 if avg_reward_train != 0 else 0
    
    print(f"Training + Eval Average: {avg_reward_train:.3f}")
    print(f"Eval-Only Average:      {avg_reward_eval:.3f}")
    print(f"Difference:             {reward_diff:+.3f} ({reward_diff_pct:+.1f}%)")
    
    # Episode-by-episode comparison
    print("\nEpisode-by-Episode Comparison:")
    print("Episode | Train+Eval | Eval-Only | Difference")
    print("-" * 48)
    for i in range(min(len(train_eval_metrics["episode_rewards"]), len(eval_only_metrics["episode_rewards"]))):
        train_r = train_eval_metrics["episode_rewards"][i]
        eval_r = eval_only_metrics["episode_rewards"][i]
        diff = eval_r - train_r
        print(f"   {i+1:2d}   |   {train_r:6.2f}   |  {eval_r:6.2f}   |   {diff:+6.2f}")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if abs(reward_diff) < 0.1:
        print("✅ Results are VERY SIMILAR - likely no significant difference")
    elif abs(reward_diff) < 1.0:
        print("⚠️  Results show SMALL DIFFERENCES - may be within normal variance")
    else:
        print("🔥 Results show SIGNIFICANT DIFFERENCES - confirms your theory!")
        print("   This suggests training+eval vs eval-only behave differently.")
    
    print(f"\nKey factors to investigate:")
    print(f"1. Model training state (training=True vs training=False)")
    print(f"2. Noise layer states in NoisyLinear layers")
    print(f"3. Action selection method (argmax vs probabilistic sampling)")
    print(f"4. Random state seeding")
    
    print(f"\nTest completed. Full logs available in: {latest_log_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
from pathlib import Path

# Import from existing codebase
import gymnasium as gym
from rainbow_dqn import RainbowDQN
from env import VacuumEnv  
from wrappers import DumbWrapper, SmartExplorationWrapper
from eval import MetricWrapper
from gymnasium.wrappers import TimeLimit

# Register environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

def load_trial_hyperparameters(trial_dir):
    """Load hyperparameters from trial directory"""
    hyperparam_file = trial_dir / "hyperparameters.json"
    with open(hyperparam_file, 'r') as f:
        return json.load(f)

def create_env_and_wrapper(wrapper_type, grid_size=(20, 20), dirt_num=5, max_steps=3000):
    """Create environment with appropriate wrapper (matching hyperparameter search setup)"""
    env = gym.make("Vacuum-v0", 
                  grid_size=grid_size, 
                  dirt_num=dirt_num)
    env = TimeLimit(env, max_episode_steps=max_steps)
    
    # Apply chosen wrapper
    if wrapper_type == 'smart':
        env = SmartExplorationWrapper(env)
    else:  # 'dumb'
        env = DumbWrapper(env)
        
    env = MetricWrapper(env)
    
    # Use random walls by passing empty list to trigger random room generation
    env.reset(options={"walls": []})
    return env

def evaluate_model(model_path, hyperparams, device='cuda'):
    """Evaluate a trained model and return average reward"""
    print(f"Evaluating model: {model_path}")
    
    # Create environment with same wrapper as training
    env = create_env_and_wrapper(hyperparams['wrapper_type'])
    
    # Calculate observation dimensions
    obs_dim = {
        'agent_orient': 1,
        'knowledge_map': 20 * 20  # 20x20 grid for hyperparameter search
    }
    n_actions = env.action_space.n
    
    # Load model
    model = RainbowDQN(obs_dim, n_actions).to(device)
    
    # Load trained weights (handle the save format from rainbow_dqn.py)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Handle old format if any
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Run evaluation episodes
    eval_episodes = 10
    total_rewards = []
    
    for episode in range(eval_episodes):
        state, _ = env.reset()  # env.reset() returns (observation, info)
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action = model.act(state, epsilon=0.0)  # No exploration during evaluation
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return avg_reward, std_reward

def evaluate_all_models(study_dir):
    """Evaluate all trained models and create results summary"""
    study_dir = Path(study_dir)
    
    print(f"Evaluating models from: {study_dir}")
    
    # Process each trial directory
    trial_dirs = sorted([d for d in study_dir.iterdir() if d.name.startswith('trial_')])
    
    results = []
    
    for trial_dir in trial_dirs:
        trial_num = int(trial_dir.name.split('_')[1])
        print(f"\n--- Processing Trial {trial_num} ---")
        
        # Load hyperparameters
        hyperparams = load_trial_hyperparameters(trial_dir)
        print(f"Hyperparameters: {hyperparams}")
        
        # Find the best model (prefer best_model.pth, fallback to final_model.pth)
        training_dir = trial_dir / "training"
        best_model_path = training_dir / "best_model.pth"
        final_model_path = training_dir / "final_model.pth"
        
        if best_model_path.exists():
            model_path = best_model_path
            model_type = "best"
        elif final_model_path.exists():
            model_path = final_model_path
            model_type = "final"
        else:
            print(f"No model found for trial {trial_num}, skipping...")
            continue
        
        try:
            # Evaluate the model
            avg_reward, std_reward = evaluate_model(model_path, hyperparams)
            
            # Load training metrics if available
            metrics_file = training_dir / "training_metrics.json"
            training_metrics = {}
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    training_metrics = json.load(f)
            
            results.append({
                'trial': trial_num,
                'params': hyperparams,
                'eval_reward_mean': avg_reward,
                'eval_reward_std': std_reward,
                'model_type': model_type,
                'training_metrics': training_metrics
            })
            
            print(f"Trial {trial_num} completed with reward: {avg_reward:.2f} ± {std_reward:.2f}")
            
        except Exception as e:
            print(f"Error evaluating trial {trial_num}: {e}")
            continue
    
    # Save results to JSON file
    results_file = study_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {results_file}")
    
    # Print summary
    if results:
        best_result = max(results, key=lambda x: x['eval_reward_mean'])
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Total trials evaluated: {len(results)}")
        print(f"Best trial: {best_result['trial']}")
        print(f"Best reward: {best_result['eval_reward_mean']:.2f} ± {best_result['eval_reward_std']:.2f}")
        print(f"Best parameters: {best_result['params']}")
        
        # Show top 5 results
        sorted_results = sorted(results, key=lambda x: x['eval_reward_mean'], reverse=True)
        print(f"\n=== TOP 5 TRIALS ===")
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. Trial {result['trial']}: {result['eval_reward_mean']:.2f} ± {result['eval_reward_std']:.2f}")
            print(f"   {result['params']}")
        
        # Analyze hyperparameter patterns
        print(f"\n=== HYPERPARAMETER ANALYSIS ===")
        wrapper_performance = {}
        for result in results:
            wrapper = result['params']['wrapper_type']
            if wrapper not in wrapper_performance:
                wrapper_performance[wrapper] = []
            wrapper_performance[wrapper].append(result['eval_reward_mean'])
        
        for wrapper, rewards in wrapper_performance.items():
            print(f"{wrapper.title()} wrapper: {np.mean(rewards):.2f} ± {np.std(rewards):.2f} (n={len(rewards)})")
    
    return results

if __name__ == "__main__":
    study_dir = "logs/optuna_studies/rainbow_dqn_study_20250608_195648"
    
    if not os.path.exists(study_dir):
        print(f"Study directory not found: {study_dir}")
        exit(1)
    
    print("Starting evaluation of trained models...")
    results = evaluate_all_models(study_dir)
    print("Evaluation completed!") 

import optuna
import os
import torch
import numpy as np
from rainbow_dqn import train, evaluate, RainbowDQN, get_log_dir
import json
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from env import VacuumEnv
from wrappers import DumbWrapper, SmartExplorationWrapper
from eval import MetricWrapper

# Base directory for all Optuna studies
OPTUNA_STUDIES_DIR = os.path.join('logs', 'optuna_studies')
os.makedirs(OPTUNA_STUDIES_DIR, exist_ok=True)

# Register the environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

def make_env(grid_size=(20, 20), max_steps=3000, dirt_num=5, wrapper_type='dumb'):
    """Create a wrapped vacuum environment with specified parameters.
    
    Args:
        grid_size (tuple): Size of the grid (width, height)
        max_steps (int): Maximum steps per episode
        dirt_num (int): Number of dirt clusters to place
        wrapper_type (str): Type of exploration wrapper ('dumb' or 'smart')
    
    Returns:
        callable: A function that creates and returns the wrapped environment
    """
    
    def _env():
        env = gym.make("Vacuum-v0", 
                      grid_size=grid_size, 
                      dirt_num=dirt_num)
        env = TimeLimit(env, max_episode_steps=max_steps)
        
        # Apply chosen wrapper
        if wrapper_type == 'smart':
            env = SmartExplorationWrapper(env)
        else:
            env = DumbWrapper(env)
            
        env = MetricWrapper(env)
        
        # Use random walls by passing empty list to trigger random room generation
        env.reset(options={"walls": []})
        return env
    
    return _env

def objective(trial):
    # Create study directory for this trial
    study_name = trial.study.study_name
    study_dir = os.path.join(OPTUNA_STUDIES_DIR, study_name)
    trial_dir = os.path.join(study_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)
    
    # Focus on the 5 most impactful hyperparameters + wrapper choice
    params = {
        'wrapper_type': trial.suggest_categorical('wrapper_type', ['dumb', 'smart']),
        'lr': trial.suggest_float('lr', 1e-5, 5e-4, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.995),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'n_step': trial.suggest_categorical('n_step', [1, 3, 5])
    }
    
    # Save hyperparameters
    with open(os.path.join(trial_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    try:
        # Override global variables in rainbow_dqn.py (only the ones we're tuning)
        import rainbow_dqn as rdqn
        rdqn.LR = 0.0004067366522568581
        rdqn.BATCH_SIZE = params['batch_size']
        rdqn.GAMMA = 0.9619576978096083
        rdqn.N_STEP = 1
        
        # Environment parameters - optimized for 20x20 grid with random walls and 5 dirt spots
        grid_size = (20, 20)
        total_timesteps = 300000  # Target for promising trials
        eval_episodes = 5  # More episodes for reliable evaluation
        max_steps = 3000
        dirt_num = 5
        
        # Create environment factory with consistent settings
        env_fn = make_env(
            grid_size=grid_size,
            max_steps=max_steps,
            dirt_num=dirt_num,
            wrapper_type=params['wrapper_type']
        )
        
        # Create training and eval environments from the same factory
        train_env = env_fn()
        eval_env = env_fn()  # This ensures same layout and settings
        
        # Create training and evaluation directories
        train_dir = os.path.join(trial_dir, 'training')
        eval_dir = os.path.join(trial_dir, 'evaluation')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Train the model with intermediate reporting for pruning
        # We'll modify this to report intermediate rewards every 50k steps
        intermediate_rewards = []
        checkpoint_steps = 50000  # Report every 50k steps
        
        for checkpoint in range(1, (total_timesteps // checkpoint_steps) + 1):
            current_steps = checkpoint * checkpoint_steps
            
            # Train for this checkpoint
            model = train(
                env=train_env,
                grid_size=grid_size,
                total_timesteps=current_steps,
                save_freq=current_steps,  # Only save at the end of each checkpoint
                output_dir=train_dir,
                wrapper=params['wrapper_type']
            )
            
            # Quick evaluation to get intermediate performance
            quick_metrics = evaluate(
                model,
                env=eval_env,
                grid_size=grid_size,
                episodes=2,  # Quick evaluation with fewer episodes
                render=False,
                output_dir=eval_dir,
                wrapper=params['wrapper_type']
            )
            
            intermediate_reward = np.mean(quick_metrics['episode_rewards'])
            intermediate_rewards.append(intermediate_reward)
            
            # Report to Optuna for pruning decision
            trial.report(intermediate_reward, checkpoint - 1)
            
            # Check if trial should be pruned
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at {current_steps} steps with reward {intermediate_reward:.2f}")
                train_env.close()
                eval_env.close()
                raise optuna.TrialPruned()
        
        # Final evaluation with full episodes
        print(f"Trial {trial.number} completed full training")
        metrics = evaluate(
            model,
            env=eval_env,
            grid_size=grid_size,
            episodes=eval_episodes,
            render=False,
            output_dir=eval_dir,
            wrapper=params['wrapper_type']
        )
        

        
        # Calculate mean reward as the objective
        mean_reward = np.mean(metrics['episode_rewards'])
        
        # Save all metrics for analysis, but only use mean reward for optimization
        results = {
            'mean_reward': float(mean_reward),
            'metrics': {
                'episode_rewards': metrics['episode_rewards'],
                'episode_lengths': metrics['episode_lengths'],
                'coverage_ratio': metrics['coverage_ratio'],
                'path_efficiency': metrics['path_efficiency'],
                'revisit_ratio': metrics['revisit_ratio']
            }
        }
        
        with open(os.path.join(trial_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Clean up environments
        train_env.close()
        eval_env.close()
        
        return mean_reward  # Only return mean reward as the objective
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
        return float('-inf')  # Return worst possible value on failure

def main():
    # Create study name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_name = f"rainbow_dqn_study_{timestamp}"
    study_dir = os.path.join(OPTUNA_STUDIES_DIR, study_name)
    os.makedirs(study_dir, exist_ok=True)
    
    # Create database in the study directory
    storage_name = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    
    # Create and run study with pruning
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials (need baseline)
            n_warmup_steps=1,    # Wait 1 checkpoints before pruning
            interval_steps=1     # Check for pruning every checkpoint
        )
    )
    
    n_trials = 30  # Many trials with early pruning to stay within budget
    study.optimize(objective, n_trials=n_trials)
    
    # Print best trial information
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Save study results in the study directory
    study_results = {
        'best_value': trial.value,
        'best_params': trial.params,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in study.trials
        ]
    }
    
    with open(os.path.join(study_dir, 'study_results.json'), 'w') as f:
        json.dump(study_results, f, indent=4)
    
    # Create visualization plots in the study directory
    try:
        # Plot optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(os.path.join(study_dir, 'optimization_history.html'))
        
        # Plot parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(os.path.join(study_dir, 'param_importances.html'))
        
        # Plot parallel coordinate
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(study_dir, 'parallel_coordinate.html'))
    except Exception as e:
        print(f"Warning: Could not create some visualizations: {str(e)}")

if __name__ == "__main__":
    main() 

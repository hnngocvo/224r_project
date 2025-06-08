#!/usr/bin/env python3
"""
Integration test for the knowledge map environment + Rainbow DQN + SmartExplorationWrapper
Tests basic functionality without training to ensure all components work together.
"""

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# Import our components
from env import VacuumEnv
from constants import OBSTACLE, CLEAN, ROBOT, UNKNOWN, DIRTY, RETURN_TARGET
from rainbow_dqn import RainbowDQN, device
from wrappers import SmartExplorationWrapper
from eval import MetricWrapper

# Register environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

def print_knowledge_map(knowledge_map, step_num, action_taken=None):
    """Pretty print the knowledge map with semantic meanings"""
    print(f"\n--- Step {step_num} Knowledge Map ---")
    if action_taken is not None:
        action_names = ["FORWARD", "LEFT", "RIGHT"]
        print(f"Action taken: {action_names[action_taken]}")
    
    h, w = knowledge_map.shape
    symbols = {
        OBSTACLE: '■',    # Black square for obstacles
        CLEAN: '·',       # Dot for clean/visited
        ROBOT: '🤖',      # Robot emoji
        UNKNOWN: '?',     # Question mark for unknown
        DIRTY: '💩',      # Dirt emoji  
        RETURN_TARGET: '🏠' # Home emoji
    }
    
    print("Legend: ■=Obstacle, ·=Clean, 🤖=Robot, ?=Unknown, 💩=Dirty, 🏠=Home")
    for i in range(h):
        row = ""
        for j in range(w):
            value = knowledge_map[i, j]
            # Find closest semantic value
            closest_symbol = '?'
            min_diff = float('inf')
            for semantic_val, symbol in symbols.items():
                diff = abs(value - semantic_val)
                if diff < min_diff:
                    min_diff = diff
                    closest_symbol = symbol
            row += closest_symbol + " "
        print(row)
    print()

def log_reward_breakdown(base_reward, exploration_reward, env):
    """Log detailed reward information"""
    print(f"Base reward: {base_reward:.3f}")
    print(f"Exploration reward: {exploration_reward:.3f}")
    print(f"Total reward: {base_reward + exploration_reward:.3f}")
    
    # Try to get more details from the wrapper if possible
    if hasattr(env, 'unwrapped'):
        agent_pos = env.unwrapped.agent_pos
        agent_orient = env.unwrapped.agent_orient
        print(f"Agent position: {agent_pos}, orientation: {agent_orient}")
        
        if hasattr(env.unwrapped, 'dirt_remaining'):
            dirt_left = env.unwrapped.dirt_remaining
            print(f"Dirt remaining: {dirt_left}")

def test_network_forward_pass(network, obs):
    """Test that the network can process observations without errors"""
    try:
        # Convert observation to the format expected by network
        state = {
            'agent_orient': torch.FloatTensor(obs['agent_orient']).to(device),
            'knowledge_map': torch.FloatTensor(obs['knowledge_map']).unsqueeze(0).to(device)
        }
        
        with torch.no_grad():
            network.eval()  # Set to eval mode
            dist = network(state)
            q_values = (dist * network.support).sum(dim=2)
            
        print(f"Network forward pass successful!")
        print(f"Q-values shape: {q_values.shape}")
        print(f"Q-values: {q_values[0].cpu().numpy()}")
        return True
        
    except Exception as e:
        print(f"Network forward pass failed: {e}")
        return False

def run_integration_test():
    """Run the main integration test"""
    print("="*60)
    print("INTEGRATION TEST: Knowledge Map + Rainbow DQN + Smart Wrapper")
    print("="*60)
    
    # Test parameters
    grid_size = (6, 6)
    dirt_num = 3
    max_steps = 50
    num_episodes = 3
    
    print(f"Test setup:")
    print(f"- Grid size: {grid_size}")
    print(f"- Dirt clusters: {dirt_num}")
    print(f"- Max steps per episode: {max_steps}")
    print(f"- Number of test episodes: {num_episodes}")
    print(f"- Device: {device}")
    
    # Create environment with all wrappers
    try:
        print("\n1. Creating environment...")
        env = gym.make("Vacuum-v0", grid_size=grid_size, dirt_num=dirt_num)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = SmartExplorationWrapper(env)
        # env = ExploitationPenaltyWrapper(env, time_penalty=-0.002, stay_penalty=-0.05)
        env = MetricWrapper(env)
        print("✓ Environment created successfully")
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False
    
    # Create network
    try:
        print("\n2. Creating Rainbow DQN network...")
        obs_dim = {
            'agent_orient': 1,
            'knowledge_map': grid_size[0] * grid_size[1]
        }
        n_actions = env.action_space.n
        network = RainbowDQN(obs_dim, n_actions).to(device)
        print(f"✓ Network created successfully")
        print(f"   Input dims: {obs_dim}")
        print(f"   Actions: {n_actions}")
        print(f"   Parameters: {sum(p.numel() for p in network.parameters()):,}")
        
    except Exception as e:
        print(f"✗ Network creation failed: {e}")
        return False
    
    # Run test episodes
    for episode in range(num_episodes):
        print(f"\n{'='*40}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*40}")
        
        try:
            # Reset environment
            obs, info = env.reset()
            print("✓ Environment reset successful")
            
            # Test initial observation structure
            print(f"\nObservation keys: {list(obs.keys())}")
            print(f"Knowledge map shape: {obs['knowledge_map'].shape}")
            print(f"Agent orient shape: {obs['agent_orient'].shape}")
            print(f"Agent orient value: {obs['agent_orient']}")
            
            # Print initial knowledge map
            print_knowledge_map(obs['knowledge_map'], 0)
            
            # Test network on initial observation
            network_ok = test_network_forward_pass(network, obs)
            if not network_ok:
                return False
                
            # Run episode
            total_reward = 0
            step = 0
            
            while step < max_steps:
                step += 1
                
                # Take random action for testing
                action = env.action_space.sample()
                
                # Step environment  
                prev_obs = obs.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # Log step info
                print(f"\nStep {step}:")
                log_reward_breakdown(reward, 0, env)  # Note: reward includes wrapper modifications
                
                # Print knowledge map if it changed significantly
                map_changed = not np.array_equal(prev_obs['knowledge_map'], obs['knowledge_map'])
                if map_changed or step <= 3:  # Always show first few steps
                    print_knowledge_map(obs['knowledge_map'], step, action)
                
                # Test network on new observation
                if step % 10 == 0:  # Test every 10 steps
                    network_ok = test_network_forward_pass(network, obs)
                    if not network_ok:
                        return False
                
                if terminated or truncated:
                    print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
                    break
            
            print(f"\n--- Episode {episode + 1} Summary ---")
            print(f"Steps taken: {step}")
            print(f"Total reward: {total_reward:.3f}")
            print(f"Final metrics: {info}")
            
        except Exception as e:
            print(f"✗ Episode {episode + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*60}")
    print("✓ INTEGRATION TEST PASSED!")
    print("All components working together successfully.")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    success = run_integration_test()
    if not success:
        print("\n✗ Integration test failed!")
        exit(1)
    else:
        print("\n🎉 Ready for training!") 

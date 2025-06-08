#!/usr/bin/env python3
"""
Quick test script to debug SmartExplorationWrapper
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from debug_smart_wrapper import DebugSmartExplorationWrapper
from eval import MetricWrapper

# Register environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

def debug_test():
    print("=== DEBUG TEST: SmartExplorationWrapper ===")
    
    # Create environment with debug wrapper
    env = gym.make("Vacuum-v0", grid_size=(6, 6), use_counter=False, dirt_num=3)
    env = TimeLimit(env, max_episode_steps=10)  # Very short for debugging
    env = DebugSmartExplorationWrapper(env)
    env = MetricWrapper(env)
    
    print("\nStarting debug test...")
    obs, info = env.reset()
    
    print(f"\nInitial observation keys: {list(obs.keys())}")
    print(f"Knowledge map shape: {obs['knowledge_map'].shape}")
    print(f"Agent position: {env.unwrapped.agent_pos}")
    print(f"Agent orientation: {env.unwrapped.agent_orient}")
    
    # Take a few actions to see what happens
    actions = [0, 1, 0, 2, 0]  # forward, left, forward, right, forward
    
    for i, action in enumerate(actions):
        print(f"\n{'='*50}")
        print(f"STEP {i+1}: Taking action {action}")
        print(f"{'='*50}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Episode ended: terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            break
    
    env.close()
    print("\nDebug test complete!")

if __name__ == "__main__":
    debug_test() 

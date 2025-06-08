#!/usr/bin/env python3
"""
Debug version of SmartExplorationWrapper with extensive logging
"""

import gymnasium as gym
import numpy as np
from collections import deque

# Import semantic values from constants.py
from constants import OBSTACLE, CLEAN, ROBOT, UNKNOWN, DIRTY, RETURN_TARGET

class DebugSmartExplorationWrapper(gym.Wrapper):
    """
    Debug version of SmartExplorationWrapper with extensive logging
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.grid_size
        self.orientations = self.env.unwrapped.orientations
        
        # History tracking (past 5 moves)
        self.action_history = deque(maxlen=5)
        self.position_orientation_history = deque(maxlen=5)
        
        # Previous state for reward calculation
        self.prev_pos = None
        self.prev_orient = None
        self.prev_knowledge_map = None
        
        print(f"[DEBUG] SmartExplorationWrapper initialized")
        print(f"[DEBUG] Grid size: {self.grid_size}")
        print(f"[DEBUG] Orientations: {self.orientations}")
        
    def reset(self, **kwargs):
        print(f"[DEBUG] SmartExplorationWrapper.reset() called")
        self.action_history.clear()
        self.position_orientation_history.clear()
        self.prev_pos = None
        self.prev_orient = None
        self.prev_knowledge_map = None
        
        obs, info = self.env.reset(**kwargs)
        
        # Initialize tracking
        self.prev_pos = tuple(self.env.unwrapped.agent_pos)
        self.prev_orient = self.env.unwrapped.agent_orient
        self.prev_knowledge_map = obs['knowledge_map'].copy()
        
        print(f"[DEBUG] Reset - Initial pos: {self.prev_pos}, orient: {self.prev_orient}")
        print(f"[DEBUG] Reset - Knowledge map shape: {self.prev_knowledge_map.shape}")
        
        return obs, info
    
    def step(self, action):
        print(f"\n[DEBUG] === SmartExplorationWrapper.step(action={action}) ===")
        
        # Store state before action
        old_pos = tuple(self.env.unwrapped.agent_pos)
        old_orient = self.env.unwrapped.agent_orient
        
        print(f"[DEBUG] Before action - pos: {old_pos}, orient: {old_orient}")
        
        # Take the action
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Get new state
        new_pos = tuple(self.env.unwrapped.agent_pos)
        new_orient = self.env.unwrapped.agent_orient
        knowledge_map = obs['knowledge_map']
        
        print(f"[DEBUG] After action - pos: {new_pos}, orient: {new_orient}")
        print(f"[DEBUG] Base reward: {base_reward}")
        print(f"[DEBUG] Knowledge map available: {knowledge_map is not None}")
        
        # Calculate exploration reward
        exploration_reward = self._calculate_exploration_reward(
            action, old_pos, old_orient, new_pos, new_orient, knowledge_map, base_reward
        )
        
        print(f"[DEBUG] Calculated exploration reward: {exploration_reward}")
        
        # Update history
        self.action_history.append(action)
        self.position_orientation_history.append((old_pos, old_orient))
        
        # Update tracking variables
        self.prev_pos = new_pos
        self.prev_orient = new_orient
        self.prev_knowledge_map = knowledge_map.copy()
        
        total_reward = base_reward + exploration_reward
        print(f"[DEBUG] Total reward: {base_reward} + {exploration_reward} = {total_reward}")
        
        return obs, total_reward, terminated, truncated, info
    
    def _calculate_exploration_reward(self, action, old_pos, old_orient, new_pos, new_orient, knowledge_map, base_reward):
        """Calculate dense exploration rewards with debug logging"""
        print(f"[DEBUG] _calculate_exploration_reward called")
        reward = 0.0
        
        try:
            # 1. History rewards - penalize revisiting recent position+orientation combinations
            print(f"[DEBUG] Calculating history reward...")
            history_reward = self._history_reward(action, old_pos, old_orient, new_pos, new_orient)
            print(f"[DEBUG] History reward: {history_reward}")
            reward += history_reward
            
            # 2. Distance rewards - reward moves toward objectives
            print(f"[DEBUG] Calculating distance reward...")
            distance_reward = self._distance_reward(old_pos, new_pos, knowledge_map)
            print(f"[DEBUG] Distance reward: {distance_reward}")
            reward += distance_reward
            
            # 3. Tweaked base rewards - counteract harsh base penalties
            print(f"[DEBUG] Calculating tweaked base rewards...")
            base_tweaks = self._tweaked_base_rewards(action, old_pos, new_pos, knowledge_map, base_reward)
            print(f"[DEBUG] Base tweaks: {base_tweaks}")
            reward += base_tweaks
            
            # 4. Rotation rewards - encourage smart turning
            print(f"[DEBUG] Calculating rotation reward...")
            rotation_reward = self._rotation_reward(action, old_pos, old_orient, new_orient, knowledge_map)
            print(f"[DEBUG] Rotation reward: {rotation_reward}")
            reward += rotation_reward
            
            print(f"[DEBUG] Total exploration reward components: {history_reward} + {distance_reward} + {base_tweaks} + {rotation_reward} = {reward}")
            
        except Exception as e:
            print(f"[DEBUG] ERROR in _calculate_exploration_reward: {e}")
            import traceback
            traceback.print_exc()
            reward = 0.0
        
        return reward
    
    def _history_reward(self, action, old_pos, old_orient, new_pos, new_orient):
        """Penalize moves that revisit recent position+orientation combinations"""
        print(f"[DEBUG] _history_reward: action={action}, old_pos={old_pos}, new_pos={new_pos}")
        print(f"[DEBUG] Position history: {list(self.position_orientation_history)}")
        
        # Check if this position+orientation combination was visited recently
        current_state = (new_pos, new_orient)
        if current_state in self.position_orientation_history:
            print(f"[DEBUG] Found recent visit to {current_state} -> penalty -0.5")
            return -0.5  # Penalty for flip-flopping or spinning
            
        print(f"[DEBUG] No recent visit to {current_state} -> reward 0.0")
        return 0.0
    
    def _distance_reward(self, old_pos, new_pos, knowledge_map):
        """Reward moves that get closer to objectives using BFS distances"""
        print(f"[DEBUG] _distance_reward: old_pos={old_pos}, new_pos={new_pos}")
        
        if old_pos == new_pos:  # No movement (rotation only)
            print(f"[DEBUG] No movement -> reward 0.0")
            return 0.0
        
        try:
            # Calculate BFS distances from both positions
            print(f"[DEBUG] Calculating BFS distances...")
            old_distances = self._bfs_distances(old_pos, knowledge_map)
            new_distances = self._bfs_distances(new_pos, knowledge_map)
            
            print(f"[DEBUG] Old distances calculated: {len(old_distances)} reachable cells")
            print(f"[DEBUG] New distances calculated: {len(new_distances)} reachable cells")
            
            reward = 0.0
            
            # Check if return target is present (all dirt cleaned)
            return_target_pos = self._find_return_target(knowledge_map)
            print(f"[DEBUG] Return target position: {return_target_pos}")
            
            if return_target_pos is not None:
                # Heavily reward moving toward return target
                old_dist = old_distances.get(return_target_pos, float('inf'))
                new_dist = new_distances.get(return_target_pos, float('inf'))
                print(f"[DEBUG] Return target distances: old={old_dist}, new={new_dist}")
                
                if new_dist < old_dist:
                    reward += 20.0  # Large reward for approaching home
                    print(f"[DEBUG] Approaching home -> +20.0")
                else:
                    reward -= 20.0  # Large penalty for moving away from home
                    print(f"[DEBUG] Moving away from home -> -20.0")
            else:
                # Reward moving toward closest dirty cell
                print(f"[DEBUG] Looking for dirty cells...")
                dirty_improvement = self._closest_objective_improvement(
                    old_distances, new_distances, knowledge_map, DIRTY
                )
                print(f"[DEBUG] Dirty improvement: {dirty_improvement}")
                if dirty_improvement > 0:
                    dirty_reward = 0.6 * dirty_improvement
                    reward += dirty_reward
                    print(f"[DEBUG] Dirty approach -> +{dirty_reward}")
            
            # Reward moving toward unknown cells (exploration)
            print(f"[DEBUG] Looking for unknown cells...")
            unknown_improvement = self._closest_objective_improvement(
                old_distances, new_distances, knowledge_map, UNKNOWN
            )
            print(f"[DEBUG] Unknown improvement: {unknown_improvement}")
            if unknown_improvement > 0:
                unknown_reward = 0.3 * unknown_improvement
                reward += unknown_reward
                print(f"[DEBUG] Unknown exploration -> +{unknown_reward}")
                
            print(f"[DEBUG] Total distance reward: {reward}")
            return reward
            
        except Exception as e:
            print(f"[DEBUG] ERROR in _distance_reward: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _tweaked_base_rewards(self, action, old_pos, new_pos, knowledge_map, base_reward):
        """Counteract harsh base penalties and add appropriate ones"""
        print(f"[DEBUG] _tweaked_base_rewards: action={action}, moved={old_pos != new_pos}, base_reward={base_reward}")
        
        reward = 0.0
        
        try:
            # If this was an invalid move (position didn't change on forward action)
            if action == 0 and old_pos == new_pos:
                # Counteract the base -1.0 invalid move penalty
                reward += 1.0
                print(f"[DEBUG] Invalid move counteraction -> +1.0")
                
                # Add appropriate penalty based on reason
                dx, dy = self.orientations[self.prev_orient]
                target_x, target_y = old_pos[0] + dx, old_pos[1] + dy
                
                # Out of bounds or KNOWN obstacle penalty
                if (target_x < 0 or target_x >= self.grid_size[0] or 
                    target_y < 0 or target_y >= self.grid_size[1] or
                    knowledge_map[target_x, target_y] == OBSTACLE):
                    reward -= 1.0
                    print(f"[DEBUG] Out of bounds/obstacle penalty -> -1.0")
            
            # Counteract revisit penalty (base environment penalizes revisiting)
            if action == 0 and old_pos != new_pos and knowledge_map[new_pos] == CLEAN:
                reward += 0.3  # Small reward to counteract base revisit penalty
                print(f"[DEBUG] Revisit counteraction -> +0.3")
            
            print(f"[DEBUG] Total base tweaks: {reward}")
            return reward
            
        except Exception as e:
            print(f"[DEBUG] ERROR in _tweaked_base_rewards: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _rotation_reward(self, action, old_pos, old_orient, new_orient, knowledge_map):
        """Reward smart rotation behavior"""
        print(f"[DEBUG] _rotation_reward: action={action}, old_orient={old_orient}, new_orient={new_orient}")
        
        if action == 0:  # Not a rotation
            print(f"[DEBUG] Not a rotation -> reward 0.0")
            return 0.0
            
        try:
            reward = 0.0
            
            # Get the direction we're now facing after rotation
            dx, dy = self.orientations[new_orient]
            target_x, target_y = old_pos[0] + dx, old_pos[1] + dy
            
            print(f"[DEBUG] Now facing direction {new_orient} -> target cell ({target_x}, {target_y})")
            
            # Reward rotating when facing boundary or known obstacle
            if (target_x < 0 or target_x >= self.grid_size[0] or 
                target_y < 0 or target_y >= self.grid_size[1] or
                knowledge_map[target_x, target_y] == OBSTACLE):
                reward += 0.5  # Good to turn away from known obstacle or boundary
                print(f"[DEBUG] Turning away from obstacle/boundary -> +0.5")
                
            # Reward rotating toward dirty cells or unknown cells
            if (0 <= target_x < self.grid_size[0] and 0 <= target_y < self.grid_size[1]):
                target_value = knowledge_map[target_x, target_y]
                print(f"[DEBUG] Target cell value: {target_value}")
                
                if target_value == DIRTY:
                    reward += 0.1  # Small reward for facing dirt
                    print(f"[DEBUG] Facing dirt -> +0.1")
                elif target_value == UNKNOWN:
                    reward += 0.05  # Small reward for facing unknown
                    print(f"[DEBUG] Facing unknown -> +0.05")
                elif target_value == RETURN_TARGET:
                    reward += 0.3  # Larger reward for facing home when needed
                    print(f"[DEBUG] Facing home -> +0.3")
                    
            print(f"[DEBUG] Total rotation reward: {reward}")
            return reward
            
        except Exception as e:
            print(f"[DEBUG] ERROR in _rotation_reward: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _bfs_distances(self, start_pos, knowledge_map):
        """Calculate BFS distances from start_pos to all reachable cells"""
        print(f"[DEBUG] _bfs_distances from {start_pos}")
        
        try:
            distances = {}
            queue = deque([(start_pos, 0)])
            visited = {start_pos}
            
            while queue:
                pos, dist = queue.popleft()
                distances[pos] = dist
                
                # Explore neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = pos[0] + dx, pos[1] + dy
                    
                    # Check bounds
                    if (nx < 0 or nx >= self.grid_size[0] or 
                        ny < 0 or ny >= self.grid_size[1]):
                        continue
                        
                    if (nx, ny) in visited:
                        continue
                        
                    cell_value = knowledge_map[nx, ny]
                    
                    # Cannot pass through obstacles
                    if cell_value == OBSTACLE:
                        continue
                        
                    visited.add((nx, ny))
                    
                    # Can reach unknown cells but not go beyond them
                    if cell_value == UNKNOWN:
                        distances[(nx, ny)] = dist + 1
                        # Don't add to queue - can't go beyond unknown cells
                    else:
                        # Can pass through all other cell types
                        queue.append(((nx, ny), dist + 1))
                        
            print(f"[DEBUG] BFS found {len(distances)} reachable cells")
            return distances
            
        except Exception as e:
            print(f"[DEBUG] ERROR in _bfs_distances: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _find_return_target(self, knowledge_map):
        """Find the return target position if it exists"""
        try:
            target_positions = np.where(knowledge_map == RETURN_TARGET)
            if len(target_positions[0]) > 0:
                pos = (target_positions[0][0], target_positions[1][0])
                print(f"[DEBUG] Found return target at {pos}")
                return pos
            print(f"[DEBUG] No return target found")
            return None
        except Exception as e:
            print(f"[DEBUG] ERROR in _find_return_target: {e}")
            return None
    
    def _closest_objective_improvement(self, old_distances, new_distances, knowledge_map, objective_value):
        """Calculate improvement in distance to closest objective of given type"""
        try:
            # Find all positions with the objective value
            objective_positions = np.where(knowledge_map == objective_value)
            if len(objective_positions[0]) == 0:
                print(f"[DEBUG] No objectives found with value {objective_value}")
                return 0.0
                
            print(f"[DEBUG] Found {len(objective_positions[0])} objectives with value {objective_value}")
            
            # Find closest objective in both distance maps
            old_min_dist = float('inf')
            new_min_dist = float('inf')
            
            for i in range(len(objective_positions[0])):
                pos = (objective_positions[0][i], objective_positions[1][i])
                old_dist = old_distances.get(pos, float('inf'))
                new_dist = new_distances.get(pos, float('inf'))
                old_min_dist = min(old_min_dist, old_dist)
                new_min_dist = min(new_min_dist, new_dist)
                
            print(f"[DEBUG] Closest objective distances: old={old_min_dist}, new={new_min_dist}")
            
            # Return improvement (positive if we got closer)
            if old_min_dist == float('inf') and new_min_dist == float('inf'):
                improvement = 0.0
            elif old_min_dist == float('inf'):
                improvement = 1.0  # Found a path to objective
            elif new_min_dist == float('inf'):
                improvement = -1.0  # Lost path to objective
            else:
                improvement = old_min_dist - new_min_dist
                
            print(f"[DEBUG] Objective improvement: {improvement}")
            return improvement
            
        except Exception as e:
            print(f"[DEBUG] ERROR in _closest_objective_improvement: {e}")
            import traceback
            traceback.print_exc()
            return 0.0 

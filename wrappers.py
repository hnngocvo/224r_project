import gymnasium as gym
import numpy as np
from collections import deque

# Import semantic values from constants.py
from constants import OBSTACLE, CLEAN, ROBOT, UNKNOWN, DIRTY, RETURN_TARGET

class ExplorationBonusWrapper(gym.Wrapper):
    """
    Gives a bonus reward for visiting new tiles.
    Bonus decays over time to prioritize early exploration.
    """
    def __init__(self, env, bonus=0.2, decay=0.995):
        super().__init__(env)
        # Use unwrapped env to access custom attributes
        self.grid_size = self.env.unwrapped.grid_size
        self.bonus = bonus
        self.decay = decay
        self.visit_map = np.zeros(self.grid_size, dtype=np.float32)

    def reset(self, **kwargs):
        self.visit_map *= self.decay
        obs, info = self.env.reset(**kwargs)
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visit_map[pos] += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = tuple(self.env.unwrapped.agent_pos)

        if self.visit_map[pos] < 1e-3:
            reward += self.bonus

        reward += max(0, self.bonus * (0.3 - self.visit_map[pos]))

        self.visit_map[pos] += 1
        return obs, reward, terminated, truncated, info


class ExploitationPenaltyWrapper(gym.Wrapper):
    """
    Penalizes inefficient behavior like:
    - Staying still
    - Taking too long after cleaning
    """
    def __init__(
        self,
        env,
        time_penalty=-0.001,
        stay_penalty=-0.1,
        stay_thresholds=(5, 10, 20),
        penalties=(-5.0, -10.0, -40.0),
        disable_internal_counter=True,
    ):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.stay_penalty = stay_penalty
        self.stay_thresholds = stay_thresholds
        self.penalties = penalties

        self.prev_pos = None
        self.stay_counter = 0

        if disable_internal_counter and hasattr(env.unwrapped, "use_internal_stuck_penalty"):
            env.unwrapped.use_internal_stuck_penalty = False

    def reset(self, **kwargs):
        self.prev_pos = None
        self.stay_counter = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Penalize time usage
        reward += self.time_penalty

        # Track and penalize staying still
        curr_pos = list(self.env.unwrapped.agent_pos)
        if self.prev_pos == curr_pos:
            self.stay_counter += 1
            reward += self.stay_penalty * self.stay_counter  # increase over time
        else:
            self.stay_counter = 0
        self.prev_pos = curr_pos

        for threshold, penalty in zip(self.stay_thresholds, self.penalties):
            if self.stay_counter == threshold:
                reward += penalty

        return obs, reward, terminated, truncated, info


class SmartExplorationWrapper(gym.Wrapper):
    """
    Advanced exploration wrapper with dense rewards for efficient cleaning.
    
    Features:
    1. History rewards: Penalize revisiting recent position+orientation combinations
    2. Distance rewards: BFS-based rewards for moving toward objectives  
    3. Tweaked base rewards: Counteract harsh base penalties
    4. Rotation rewards: Encourage smart turning behavior
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.grid_size
        self.orientations = self.env.unwrapped.orientations
        
        # History tracking (past 10 moves)
        self.action_history = deque(maxlen=10)
        self.position_orientation_history = deque(maxlen=10)
        
        # Previous state for reward calculation
        self.prev_pos = None
        self.prev_orient = None
        self.prev_knowledge_map = None
        
        # Reward tracking for debugging
        self.last_base_reward = 0.0
        self.last_exploration_reward = 0.0
        
    def reset(self, **kwargs):
        self.action_history.clear()
        self.position_orientation_history.clear()
        self.prev_pos = None
        self.prev_orient = None
        self.prev_knowledge_map = None
        self.last_base_reward = 0.0
        self.last_exploration_reward = 0.0
        
        obs, info = self.env.reset(**kwargs)
        
        # Initialize tracking
        self.prev_pos = tuple(self.env.unwrapped.agent_pos)
        self.prev_orient = self.env.unwrapped.agent_orient
        self.prev_knowledge_map = obs['knowledge_map'].copy()
        
        return obs, info
    
    def step(self, action):
        # Store state before action
        old_pos = tuple(self.env.unwrapped.agent_pos)
        old_orient = self.env.unwrapped.agent_orient
        
        # Take the action
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Get new state
        new_pos = tuple(self.env.unwrapped.agent_pos)
        new_orient = self.env.unwrapped.agent_orient
        knowledge_map = obs['knowledge_map']
        
        # Calculate exploration reward
        exploration_reward = self._calculate_exploration_reward(
            action, old_pos, old_orient, new_pos, new_orient, knowledge_map, base_reward
        )
        
        # Store for debugging
        self.last_base_reward = base_reward
        self.last_exploration_reward = exploration_reward
        
        # Add reward breakdown to info for debugging
        info['base_reward'] = base_reward
        info['exploration_reward'] = exploration_reward
        info['total_reward'] = base_reward + exploration_reward
        
        # Update history
        self.action_history.append(action)
        self.position_orientation_history.append((old_pos, old_orient))
        
        # Update tracking variables
        self.prev_pos = new_pos
        self.prev_orient = new_orient
        self.prev_knowledge_map = knowledge_map.copy()
        
        return obs, base_reward + exploration_reward, terminated, truncated, info
    
    def _calculate_exploration_reward(self, action, old_pos, old_orient, new_pos, new_orient, knowledge_map, base_reward):
        """Calculate dense exploration rewards"""
        reward = 0.0
        # Reward forward movement
        if action == 0:  # Forward action
            reward += 0.1

        # 1. History rewards - penalize revisiting recent position+orientation combinations
        reward += self._history_reward(action, old_pos, old_orient, new_pos, new_orient)
        
        # 2. Distance rewards - reward moves toward objectives
        reward += self._distance_reward(old_pos, new_pos, knowledge_map)
        
        # 3. Tweaked base rewards - counteract harsh base penalties
        reward += self._tweaked_base_rewards(action, old_pos, new_pos, knowledge_map, base_reward)
        
        # 4. Rotation rewards - encourage smart turning
        reward += self._rotation_reward(action, old_pos, old_orient, new_orient, knowledge_map)
        
        return reward
    
    def _history_reward(self, action, old_pos, old_orient, new_pos, new_orient):
        """Penalize moves that revisit recent position+orientation combinations"""
        # Check if this position+orientation combination was visited recently
        current_state = (new_pos, new_orient)
        if current_state in self.position_orientation_history:
            return -0.5  # Penalty for flip-flopping or spinning
        return 0.0
    
    def _distance_reward(self, old_pos, new_pos, knowledge_map):
        """Reward moves that get closer to objectives using BFS distances"""
        if old_pos == new_pos:  # No movement (rotation only)
            return 0.0
            
        # Calculate BFS distances from both positions
        old_distances = self._bfs_distances(old_pos, knowledge_map)
        new_distances = self._bfs_distances(new_pos, knowledge_map)
        
        reward = 0.0
        
        # Check if return target is present (all dirt cleaned)
        return_target_pos = self._find_return_target(knowledge_map)
        if return_target_pos is not None:
            # Heavily reward moving toward return target
            old_dist = old_distances.get(return_target_pos, float('inf'))
            new_dist = new_distances.get(return_target_pos, float('inf'))
            if new_dist < old_dist:
                reward += 1.0  # Large reward for approaching home
            else:
                reward -= 1.0  # Large penalty for moving away from home
        else:
            # Reward moving toward closest dirty cell
            dirty_improvement = self._closest_objective_improvement(
                old_distances, new_distances, knowledge_map, DIRTY
            )
            if dirty_improvement > 0:
                reward += 0.3 * dirty_improvement       
            # Reward moving toward unknown cells (exploration)
            unknown_improvement = self._closest_objective_improvement(
                old_distances, new_distances, knowledge_map, UNKNOWN
            )
            if unknown_improvement > 0:
                reward += 0.3 * unknown_improvement
        return reward
    
    def _tweaked_base_rewards(self, action, old_pos, new_pos, knowledge_map, base_reward):
        """Counteract harsh base penalties and add appropriate ones"""
        reward = 0.0
        
        # If this was an invalid move (position didn't change on forward action)
        if action == 0 and old_pos == new_pos:
            # Counteract the base -1.0 invalid move penalty
            reward += 1.0
            
            # Add appropriate penalty based on reason
            dx, dy = self.orientations[self.prev_orient]
            target_x, target_y = old_pos[0] + dx, old_pos[1] + dy
            
            # Out of bounds or KNOWN obstacle penalty
            if (target_x < 0 or target_x >= self.grid_size[0] or 
                target_y < 0 or target_y >= self.grid_size[1] or
                knowledge_map[target_x, target_y] == OBSTACLE):
                reward -= 1.0
        
        # Handle valid forward moves - detect and fix the visit count bug
        elif action == 0 and old_pos != new_pos:
            # Check if this move is to a non-dirt cell (gets revisit penalty)
            # Dirt cells give base_reward around +1.4 (clean bonus + forward penalty)
            # Non-dirt cells give base_reward = forward_penalty + revisit_penalty
            if base_reward < 0:  # Negative reward means no dirt was cleaned
                # Calculate what the revisit penalty was
                # base_reward = penalty_forward + revisit_penalty
                # base_reward = -0.1 + revisit_penalty
                # So: revisit_penalty = base_reward + 0.1
                revisit_penalty = base_reward + 0.1
                
                # Cancel out 95% of the revisit penalty - let distance rewards handle exploration
                revisit_counteract = -revisit_penalty * 0.98
                reward += revisit_counteract
        
        # Counteract delay return penalty - we handle this with distance rewards instead
        if "penalty_delay_return" in str(base_reward):  # Heuristic check
            reward += 0.05  # Counteract the -0.05 penalty
            
        return reward
    
    def _rotation_reward(self, action, old_pos, old_orient, new_orient, knowledge_map):
        """Reward smart rotation behavior"""
        if action == 0:  # Not a rotation
            return 0.0
            
        reward = 0.0
        
        # Penalize excessive turning - check recent action history
        if len(self.action_history) >= 5:
            recent_actions = list(self.action_history)[-5:]  # Last 5 actions
            turn_count = sum(1 for a in recent_actions if a in [1, 2])  # Count turns
            
            if turn_count >= 5:  # 5 turns in last 5 actions = spinning
                reward -= 0.5  # Gentle penalty for spinning
            elif turn_count >= 4:  # 4 turns in last 5 actions = excessive turning
                reward -= 0.1  # Very gentle penalty for excessive turning
                # Don't log excessive turning to reduce clutter
        
        # Get the direction we're now facing after rotation
        dx, dy = self.orientations[new_orient]
        target_x, target_y = old_pos[0] + dx, old_pos[1] + dy
        
        # Reward rotating when facing boundary or known obstacle
        if (target_x < 0 or target_x >= self.grid_size[0] or 
            target_y < 0 or target_y >= self.grid_size[1] or
            knowledge_map[target_x, target_y] == OBSTACLE):
            reward += 0.5  # Good to turn away from known obstacle or boundary
            
        # Reward rotating toward dirty cells or unknown cells
        if (0 <= target_x < self.grid_size[0] and 0 <= target_y < self.grid_size[1]):
            target_value = knowledge_map[target_x, target_y]
            if target_value == DIRTY:
                reward += 0.1  # Small reward for facing dirt
            elif target_value == UNKNOWN:
                reward += 0.05  # Small reward for facing unknown
            elif target_value == RETURN_TARGET:
                reward += 0.3  # Larger reward for facing home when needed
                
        return reward
    
    def _bfs_distances(self, start_pos, knowledge_map):
        """Calculate BFS distances from start_pos to all reachable cells"""
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
                    
        return distances
    
    def _find_return_target(self, knowledge_map):
        """Find the return target position if it exists"""
        target_positions = np.where(knowledge_map == RETURN_TARGET)
        if len(target_positions[0]) > 0:
            return (target_positions[0][0], target_positions[1][0])
        return None
    
    def _closest_objective_improvement(self, old_distances, new_distances, knowledge_map, objective_value):
        """Calculate improvement in distance to closest objective of given type"""
        # Find all positions with the objective value
        objective_positions = np.where(knowledge_map == objective_value)
        if len(objective_positions[0]) == 0:
            return 0.0
            
        # Find closest objective in both distance maps
        old_min_dist = float('inf')
        new_min_dist = float('inf')
        
        for i in range(len(objective_positions[0])):
            pos = (objective_positions[0][i], objective_positions[1][i])
            old_min_dist = min(old_min_dist, old_distances.get(pos, float('inf')))
            new_min_dist = min(new_min_dist, new_distances.get(pos, float('inf')))
            
        # Return improvement (positive if we got closer)
        if old_min_dist == float('inf') and new_min_dist == float('inf'):
            return 0.0
        elif old_min_dist == float('inf'):
            return 1.0  # Found a path to objective
        elif new_min_dist == float('inf'):
            return -1.0  # Lost path to objective
        else:
            return old_min_dist - new_min_dist

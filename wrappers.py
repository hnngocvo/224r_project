import gymnasium as gym
import numpy as np
from collections import deque

# Import semantic values from constants.py
from constants import OBSTACLE, CLEAN, ROBOT, UNKNOWN, DIRTY, RETURN_TARGET

class DQNExplorationWrapper(gym.Wrapper):
    """
    Exploration-enhancing wrapper with:
    - New-tile bonus
    - Obstacle discovery bonus
    - Frontier bonus (non-decaying)
    - Facing direction bonus (toward dirt or unknown)
    - Escape bonus after being blocked
    - Rotation shaping and spin penalties
    - Logs all reward components
    """

    def __init__(self, env, new_tile_bonus=0.3, new_obs_bonus=0.1, frontier_bonus=0.5,
                 facing_bonus_dirty=0.25, facing_bonus_unknown=0.05, step_into_dirty_bonus=0.25,
                 meaningless_turn_penalty=-0.1, stuck_penalty=-0.3,
                 spin_without_move_penalty=-0.3, escape_bonus=0.1, wall_face_penalty=-0.02,
                 decay=0.995):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.grid_size
        self.orientations = self.env.unwrapped.orientations
        self.max_steps = getattr(self.env.unwrapped, "max_steps", 3000)  # Fallback if undefined

        # Reward tuning parameters
        self.new_tile_bonus = new_tile_bonus
        self.new_obs_bonus = new_obs_bonus
        self.frontier_bonus = frontier_bonus
        self.facing_bonus_dirty = facing_bonus_dirty
        self.facing_bonus_unknown = facing_bonus_unknown
        self.step_into_dirty_bonus = step_into_dirty_bonus
        self.meaningless_turn_penalty = meaningless_turn_penalty
        self.stuck_penalty = stuck_penalty
        self.spin_without_move_penalty = spin_without_move_penalty
        self.escape_bonus = escape_bonus
        self.wall_face_penalty = wall_face_penalty
        self.decay = decay  # unused now for frontier bonus

        # For dynamic shaping
        self.initial_stuck_penalty = stuck_penalty
        self.initial_spin_penalty = spin_without_move_penalty
        self.dynamic_penalty_enabled = True

        # Internal state
        self.visit_map = np.zeros(self.grid_size, dtype=np.float32)
        self.known_mask = np.zeros(self.grid_size, dtype=np.uint8)
        self.action_history = deque(maxlen=6)
        self.pos_history = deque(maxlen=10)
        self.was_stuck_or_blocked = False
        self.timestep = 0

    def reset(self, **kwargs):
        self.visit_map.fill(0)
        self.known_mask.fill(0)
        self.action_history.clear()
        self.pos_history.clear()
        self.was_stuck_or_blocked = False
        self.timestep = 0

        obs, info = self.env.reset(**kwargs)
        self._update_known_mask()
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visit_map[pos] += 1
        self.pos_history.append(pos)
        return obs, info

    def step(self, action):
        self.timestep += 1
        known_mask_before = np.copy(self.known_mask)
        prev_pos = tuple(self.env.unwrapped.agent_pos)

        # Dynamic penalty shaping
        if self.dynamic_penalty_enabled:
            progress = self.timestep / self.max_steps
            self.stuck_penalty = -1.0 - (1.0 * (1.0 - progress))  # from -2.0 to -1.0
            self.spin_without_move_penalty = -1.0 - (1.0 * (1.0 - progress))  # from -2.0 to -1.0

        # Capture pre-step dirt status for dirty tile bonus
        pre_dirt_map = np.copy(self.env.unwrapped.dirt_map)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_known_mask()

        pos = tuple(self.env.unwrapped.agent_pos)
        orient = self.env.unwrapped.agent_orient
        dirt_map = self.env.unwrapped.dirt_map
        known_map = self.env.unwrapped.known_obstacle_map

        info['reward_components'] = {
            'base_env_reward': reward,
            'new_tile_bonus': 0.0,
            'new_obs_bonus': 0.0,
            'frontier_bonus': 0.0,
            'facing_bonus': 0.0,
            'step_into_dirty_bonus': 0.0,
            'stuck_penalty': 0.0,
            'escape_bonus': 0.0,
            'wall_face_penalty': 0.0,
            'rotate_when_blocked': 0.0,
            'spin_without_move_penalty': 0.0,
        }

        # 1. New tile bonus: encourages visiting new locations
        if self.visit_map[pos] < 0.1:
            reward += self.new_tile_bonus
            info['reward_components']['new_tile_bonus'] = self.new_tile_bonus
        self.visit_map[pos] += 1

        # 2. Obstacle discovery bonus: for revealing new obstacles
        newly_seen = (known_mask_before == 0) & (self.known_mask == 1)
        obs_bonus = self.new_obs_bonus * np.sum(newly_seen)
        reward += obs_bonus
        info['reward_components']['new_obs_bonus'] = float(obs_bonus)

        # 3. Frontier bonus (non-decaying): encourages staying near dirty or unknown areas
        frontier = self._frontier_bonus(pos, dirt_map)
        reward += frontier
        info['reward_components']['frontier_bonus'] = frontier

        # 4. Rotation shaping: rewards facing useful directions
        face_bonus = self._rotation_shaping(action, pos, orient, dirt_map, known_map, known_mask_before)
        reward += face_bonus
        info['reward_components']['facing_bonus'] = face_bonus

        # 4.5 Step into dirty tile bonus (captured BEFORE step zeroes it)
        if pre_dirt_map[pos] == 1:
            reward += self.step_into_dirty_bonus
            info['reward_components']['step_into_dirty_bonus'] = self.step_into_dirty_bonus

        # 5. Stuck penalty: penalizes staying in same position for too long
        self.pos_history.append(pos)
        if self.pos_history.count(pos) > 3:
            reward += self.stuck_penalty
            info['reward_components']['stuck_penalty'] = self.stuck_penalty
            self.was_stuck_or_blocked = True

        # 6. Escape and wall face penalty: escaping a stuck state or facing walls
        if action == 0:
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            facing_wall_or_oob = not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]) or known_map[tx, ty] == 1
            if facing_wall_or_oob:
                reward += self.wall_face_penalty
                info['reward_components']['wall_face_penalty'] = self.wall_face_penalty
                self.was_stuck_or_blocked = True
            elif self.was_stuck_or_blocked and pos != prev_pos:
                capped = self.escape_bonus / (1 + 0.01 * self.timestep)
                reward += capped
                info['reward_components']['escape_bonus'] = capped
                self.was_stuck_or_blocked = False

        # 7. Rotate when blocked: small reward if turning away from obstacle
        if self._obstacle_ahead() and action in [1, 2]:
            new_orient = (self.env.unwrapped.agent_orient + (1 if action == 2 else -1)) % 8
            dx, dy = self.orientations[new_orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 0:
                    reward += 0.1
                    info['reward_components']['rotate_when_blocked'] = 0.1

        # 8. Penalize meaningless spinning without movement
        self.action_history.append(action)
        if len(self.action_history) == self.action_history.maxlen:
            recent_rotations = sum(1 for a in self.action_history if a in [1, 2])
            if recent_rotations >= 5 and pos == prev_pos:
                reward += self.spin_without_move_penalty
                info['reward_components']['spin_without_move_penalty'] = self.spin_without_move_penalty

        return obs, reward, terminated, truncated, info

    def _update_known_mask(self):
        """Update known_mask with the visible tiles around the agent."""
        x, y = self.env.unwrapped.agent_pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_mask[nx, ny] = 1

    def _frontier_bonus(self, pos, dirt_map):
        """Return frontier bonus if any adjacent tile is dirty or unknown."""
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if dirt_map[nx, ny] == 1 or self.known_mask[nx, ny] == 0:
                        return self.frontier_bonus
        return 0.0

    def _rotation_shaping(self, action, pos, orient, dirt_map, known_map, known_mask_before):
        """Bonus for facing useful tiles after rotating."""
        facing_bonus = 0.0
        if action in [1, 2]:
            facing_bonus += self._facing_bonus(pos, orient, dirt_map)
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1 and known_mask_before[tx, ty] == 0:
                    facing_bonus += 0.3
        return facing_bonus

    def _facing_bonus(self, pos, orient, dirt_map):
        """Bonus for facing dirt or unknown tile."""
        dx, dy = self.orientations[orient]
        tx, ty = pos[0] + dx, pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return 0.0
        if dirt_map[tx, ty] == 1:
            return self.facing_bonus_dirty
        if self.known_mask[tx, ty] == 0:
            return self.facing_bonus_unknown
        return 0.0

    def _obstacle_ahead(self):
        """Check if the tile ahead is a known obstacle or out of bounds."""
        dx, dy = self.env.unwrapped.orientations[self.env.unwrapped.agent_orient]
        tx = self.env.unwrapped.agent_pos[0] + dx
        ty = self.env.unwrapped.agent_pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return True
        return self.env.unwrapped.known_obstacle_map[tx, ty] == 1


class PPOExplorationWrapper(gym.Wrapper):
    """
    PPO Exploration-enhancing wrapper with:
    - Decaying new-tile bonus
    - Frontier bonus with decay
    - Spin penalty when rotating too much
    - Escape bonus with decay
    - Logs reward components
    """

    def __init__(self, env, new_tile_bonus=0.3, new_obs_bonus=0.1, frontier_bonus=0.5,
                 facing_bonus_dirty=0.15, facing_bonus_unknown=0.05,
                 meaningless_turn_penalty=-0.1, stuck_penalty=-0.3,
                 spin_without_move_penalty=-0.3, escape_bonus=0.2, wall_face_penalty=-0.05,
                 decay=0.995):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.grid_size
        self.orientations = self.env.unwrapped.orientations

        self.new_tile_bonus = new_tile_bonus
        self.new_obs_bonus = new_obs_bonus
        self.frontier_bonus = frontier_bonus
        self.facing_bonus_dirty = facing_bonus_dirty
        self.facing_bonus_unknown = facing_bonus_unknown
        self.meaningless_turn_penalty = meaningless_turn_penalty
        self.stuck_penalty = stuck_penalty
        self.spin_without_move_penalty = spin_without_move_penalty
        self.escape_bonus = escape_bonus
        self.wall_face_penalty = wall_face_penalty
        self.decay = decay

        self.visit_map = np.zeros(self.grid_size, dtype=np.float32)
        self.known_mask = np.zeros(self.grid_size, dtype=np.uint8)
        self.action_history = deque(maxlen=6)
        self.pos_history = deque(maxlen=10)
        self.was_stuck_or_blocked = False
        self.timestep = 0

    def reset(self, **kwargs):
        self.visit_map.fill(0)
        self.known_mask.fill(0)
        self.action_history.clear()
        self.pos_history.clear()
        self.was_stuck_or_blocked = False
        self.timestep = 0

        obs, info = self.env.reset(**kwargs)
        self._update_known_mask()
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visit_map[pos] += 1
        self.pos_history.append(pos)
        return obs, info

    def step(self, action):
        self.timestep += 1
        known_before = np.copy(self.known_mask)
        prev_pos = tuple(self.env.unwrapped.agent_pos)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_known_mask()

        pos = tuple(self.env.unwrapped.agent_pos)
        orient = self.env.unwrapped.agent_orient
        dirt_map = self.env.unwrapped.dirt_map
        known_map = self.env.unwrapped.known_obstacle_map  # Still useful for obstacle checks

        info['reward_components'] = {
            'base_env_reward': reward,
            'new_tile_bonus': 0.0,
            'new_obs_bonus': 0.0,
            'frontier_bonus': 0.0,
            'facing_bonus': 0.0,
            'spin_penalty': 0.0,
            'stuck_penalty': 0.0,
            'escape_bonus': 0.0,
            'wall_face_penalty': 0.0,
            'rotate_when_blocked': 0.0,
            'spin_without_move_penalty': 0.0,
        }

        # 1. New tile bonus: different from clean a new tile reward; encourages exploration
        if self.visit_map[pos] < 0.1:
            reward += self.new_tile_bonus
            info['reward_components']['new_tile_bonus'] = self.new_tile_bonus
        self.visit_map[pos] += 1

        # 2. Obstacle discovery (newly seen tiles): encourages discovering previously unknwon obstacles
        newly_seen = (known_before == 0) & (self.known_mask == 1)
        obs_bonus = self.new_obs_bonus * np.sum(newly_seen)
        reward += obs_bonus
        info['reward_components']['new_obs_bonus'] = float(obs_bonus)

        # 3. Frontier bonus with decay: encourages exploring dirty or unexplored borders
        decay_factor = self.decay ** self.visit_map[pos]
        frontier = self._frontier_bonus(pos, dirt_map) * decay_factor
        reward += frontier
        info['reward_components']['frontier_bonus'] = frontier

        # 4. Rotation shaping
        rot_total, face_bonus, spin_penalty = self._rotation_shaping(action, pos, orient, dirt_map, known_map)
        reward += rot_total
        info['reward_components']['facing_bonus'] = face_bonus
        info['reward_components']['spin_penalty'] = spin_penalty

        # 5. Stuck penalty: if stuck for 3 consecutive steps, penalize; encourage use of escape logic
        self.pos_history.append(pos)
        if self.pos_history.count(pos) > 3:
            reward += self.stuck_penalty
            info['reward_components']['stuck_penalty'] = self.stuck_penalty
            self.was_stuck_or_blocked = True

        # 6. Escape or wall face penalty
        if action == 0:
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1:
                    reward += self.wall_face_penalty
                    info['reward_components']['wall_face_penalty'] = self.wall_face_penalty
                    self.was_stuck_or_blocked = True
                elif self.was_stuck_or_blocked and pos != prev_pos:
                    capped = self.escape_bonus / (1 + 0.01 * self.timestep)
                    reward += capped
                    info['reward_components']['escape_bonus'] = capped
                    self.was_stuck_or_blocked = False

        # 7. Rotate when blocked
        if self._obstacle_ahead() and action in [1, 2]:
            reward += 0.1
            info['reward_components']['rotate_when_blocked'] = 0.1

        # 8. Penalize rotation without movement
        self.action_history.append(action)
        if len(self.action_history) == self.action_history.maxlen:
            if all(a in [1, 2] for a in self.action_history) and prev_pos == pos:
                reward += self.spin_without_move_penalty
                info['reward_components']['spin_without_move_penalty'] = self.spin_without_move_penalty

        return obs, reward, terminated, truncated, info

    def _update_known_mask(self):
        x, y = self.env.unwrapped.agent_pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_mask[nx, ny] = 1

    def _frontier_bonus(self, pos, dirt_map):
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if dirt_map[nx, ny] == 1 or self.known_mask[nx, ny] == 0:
                        return self.frontier_bonus
        return 0.0

    def _rotation_shaping(self, action, pos, orient, dirt_map, known_map):
        facing_bonus = 0.0
        spin_penalty = 0.0
        if action in [1, 2]:
            facing_bonus += self._facing_bonus(pos, orient, dirt_map)
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1:
                    facing_bonus += 0.3

        if len(self.action_history) == self.action_history.maxlen:
            if sum(1 for a in self.action_history if a in [1, 2]) >= 4:
                spin_penalty = self.meaningless_turn_penalty

        return facing_bonus + spin_penalty, facing_bonus, spin_penalty

    def _facing_bonus(self, pos, orient, dirt_map):
        dx, dy = self.orientations[orient]
        tx, ty = pos[0] + dx, pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return 0.0
        if dirt_map[tx, ty] == 1:
            return self.facing_bonus_dirty
        if self.known_mask[tx, ty] == 0:
            return self.facing_bonus_unknown
        return 0.0

    def _obstacle_ahead(self):
        dx, dy = self.env.unwrapped.orientations[self.env.unwrapped.agent_orient]
        tx = self.env.unwrapped.agent_pos[0] + dx
        ty = self.env.unwrapped.agent_pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return True
        return self.env.unwrapped.known_obstacle_map[tx, ty] == 1


class DumbWrapper(gym.Wrapper):
    """
    Minimal exploration wrapper that adds only essential rewards:
    1. New tile bonus - encourages exploration
    2. Obstacle discovery bonus - rewards finding walls
    3. Escalating stuck prevention - prevents infinite loops
    """
    
    def __init__(self, env, new_tile_bonus=0.15, obstacle_discovery_bonus=0.05, stuck_penalty=-0.1):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.grid_size
        self.new_tile_bonus = new_tile_bonus  # larger than forward cost (0.05)
        self.obstacle_discovery_bonus = obstacle_discovery_bonus
        self.stuck_penalty = stuck_penalty
        
        # Simple tracking
        self.visit_map = np.zeros(self.grid_size, dtype=np.int32)
        self.known_obstacles = np.zeros(self.grid_size, dtype=np.uint8)
        self.last_pos = None
        self.stuck_counter = 0
        
    def reset(self, **kwargs):
        self.visit_map.fill(0)
        self.known_obstacles.fill(0)
        self.last_pos = None
        self.stuck_counter = 0
        
        obs, info = self.env.reset(**kwargs)
        
        # Mark starting position as visited
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visit_map[pos] = 1
        self.last_pos = pos
        
        return obs, info
    
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        pos = tuple(self.env.unwrapped.agent_pos)
        exploration_reward = 0.0
        
        # 1. New tile bonus - simple exploration incentive
        if self.visit_map[pos] == 0:
            exploration_reward += self.new_tile_bonus
        self.visit_map[pos] += 1
        
        # 2. Obstacle discovery bonus - reward finding new obstacles
        newly_discovered = 0
        known_map = self.env.unwrapped.known_obstacle_map
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if known_map[i, j] == 1 and self.known_obstacles[i, j] == 0:
                    self.known_obstacles[i, j] = 1
                    newly_discovered += 1
        
        if newly_discovered > 0:
            exploration_reward += self.obstacle_discovery_bonus * newly_discovered
        
        # 3. Escalating stuck prevention - penalize staying in same spot
        if pos == self.last_pos:
            self.stuck_counter += 1
            if self.stuck_counter >= 5:
                # Escalate: -0.1, -0.2, -0.3, -0.4, etc.
                exploration_reward += self.stuck_penalty * (self.stuck_counter - 3)
        else:
            self.stuck_counter = 0
        
        self.last_pos = pos
        
        # Add reward breakdown to info for debugging
        info['base_reward'] = base_reward
        info['exploration_reward'] = exploration_reward
        info['total_reward'] = base_reward + exploration_reward
        
        return obs, base_reward + exploration_reward, terminated, truncated, info


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

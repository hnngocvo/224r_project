import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple
from env import VacuumEnv
import matplotlib.pyplot as plt
from matplotlib import animation
from eval import MetricWrapper
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit
from wrappers import DumbWrapper, SmartExplorationWrapper
from datetime import datetime


# Register custom environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

# Create base log directory
BASE_LOG_DIR = "./logs"
os.makedirs(BASE_LOG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Constants
MAX_STEPS = 3000  # Maximum steps per episode
MAX_TIMESTEPS = 1000000  # Maximum total timesteps for training

# Hyperparameters - Optimized based on Optuna study (Trial 12: -104.76 ± 57.07)
GAMMA = 0.9619576978096083  # Optimized discount factor
LR = 0.0004067366522568581  # Optimized learning rate
BATCH_SIZE = 256  # Optimized batch size (256 can cause numerical instability)
BUFFER_SIZE = int(1e5)
START_TRAINING = 200  # Start training earlier with dense rewards
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
UPDATE_TARGET_EVERY = 1000  # More frequent target updates
N_ATOMS = 51  # Standard number of atoms
V_MIN = -10  # Simpler value range for dense rewards
V_MAX = 10  # Simpler value range for dense rewards
N_STEP = 1  # 1-step returns for immediate learning
ALPHA = 0.6  # Standard PER parameters
BETA = 0.4  # Standard PER parameters
BETA_INCREMENT = 0.001  # Standard PER parameters

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'priority'))

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.noise_scale = 1.0  # Add noise scaling factor

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)  # Ensure noise is on same device
        return x.sign().mul_(x.abs().sqrt_())

    def set_noise_scale(self, scale):
        """Set the noise scaling factor (0.0 = no noise, 1.0 = full noise)"""
        self.noise_scale = scale

    def forward(self, x):
        if self.training:  # During training, use noise for exploration
            # Apply noise scaling for gradual annealing
            scaled_weight_noise = self.noise_scale * self.weight_sigma * self.weight_epsilon
            scaled_bias_noise = self.noise_scale * self.bias_sigma * self.bias_epsilon
            return F.linear(x, 
                self.weight_mu + scaled_weight_noise,
                self.bias_mu + scaled_bias_noise)
        else:  # During evaluation, use mean values for deterministic behavior
            return F.linear(x, self.weight_mu, self.bias_mu)

class RainbowDQN(nn.Module):
    def __init__(self, obs_dim: Dict[str, int], n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.atoms = N_ATOMS
        self.support = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(device)
        self.delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

        # Process observation components
        # Use embedding for discrete orientation (8 directions -> 8 features)
        self.agent_orient_embedding = nn.Embedding(8, 8)

        # Knowledge map processing network (flatten 2D map to 1D)
        map_size = obs_dim.get('knowledge_map', 36)  # Default to 6x6=36 if not specified
        
        self.knowledge_map_net = nn.Sequential(
            nn.Linear(map_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Combine all features: 8 + 64 = 72
        combined_size = 8 + 64

        # Advantage and Value streams with NoisyLinear
        self.advantage_hidden = NoisyLinear(combined_size, 128)
        self.advantage = NoisyLinear(128, n_actions * N_ATOMS)
        
        self.value_hidden = NoisyLinear(combined_size, 128)
        self.value = NoisyLinear(128, N_ATOMS)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = x['agent_orient'].shape[0]
        
        # Process agent orientation for embedding - ensure it's integer and right shape
        orient = x['agent_orient']
        if orient.dim() == 2 and orient.shape[1] == 1:
            orient = orient.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        orient_features = self.agent_orient_embedding(orient.long())  # [batch_size, 8]
        
        # Process knowledge map (flatten 2D to 1D)
        knowledge_map = x['knowledge_map'].float().flatten(start_dim=1)  # [batch_size, H*W]
        knowledge_features = self.knowledge_map_net(knowledge_map)  # [batch_size, 64]
        
        # Combine all features
        combined = torch.cat([
            orient_features, knowledge_features
        ], dim=1)  # [batch_size, 72]
        
        # Dueling architecture with noisy layers
        advantage_hidden = F.relu(self.advantage_hidden(combined))
        advantage = self.advantage(advantage_hidden).view(batch_size, self.n_actions, self.atoms)
        
        value_hidden = F.relu(self.value_hidden(combined))
        value = self.value(value_hidden).view(batch_size, 1, self.atoms)
        
        # Combine value and advantage
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Get probabilities with numerical stability
        return F.softmax(q_dist, dim=2) + 1e-8  # Add small epsilon to prevent exact zeros

    def reset_noise(self):
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value_hidden.reset_noise()
        self.value.reset_noise()

    def set_noise_scale(self, scale):
        """Set noise scaling factor for all noisy layers"""
        self.advantage_hidden.set_noise_scale(scale)
        self.advantage.set_noise_scale(scale)
        self.value_hidden.set_noise_scale(scale)
        self.value.set_noise_scale(scale)

    def act(self, state: Dict[str, np.ndarray], epsilon: float = 0.0, temperature: float = 1.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            # Convert numpy arrays to tensors and add batch dimension
            state = {
                'agent_orient': torch.FloatTensor(state['agent_orient']).to(device),
                'knowledge_map': torch.FloatTensor(state['knowledge_map']).unsqueeze(0).to(device)
            }
            
            dist = self(state)
            q_values = (dist * self.support).sum(dim=2)
            
            if self.training:
                # During training: use deterministic argmax (noise provides exploration)
                action = q_values.argmax(1).item()
            else:
                # During evaluation: sample from Q-value distribution
                # Apply temperature scaling for controllable randomness
                q_scaled = q_values[0] / temperature
                action_probs = F.softmax(q_scaled, dim=0)
                action = torch.multinomial(action_probs, 1).item()
            
            # Log Q-values for debugging (only during evaluation)
            # if not self.training:
            #     q_vals = q_values[0].cpu().numpy()
            #     action_names = ["forward", "left", "right"]
            #     action_probs_np = action_probs.cpu().numpy()
            #     print(f"Q-values: [Forward: {q_vals[0]:.3f}, Left: {q_vals[1]:.3f}, Right: {q_vals[2]:.3f}]")
            #     print(f"Action probs: [Forward: {action_probs_np[0]:.3f}, Left: {action_probs_np[1]:.3f}, Right: {action_probs_np[2]:.3f}] -> Action: {action_names[action]}")
            
            return action

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.n_step_buffer = deque(maxlen=N_STEP)
        self.gamma = GAMMA

    def _get_n_step_info(self) -> Tuple[float, Dict, bool]:
        reward = 0
        next_state = None
        done = False

        for idx, (_, _, rew, next_s, done) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * rew
            if done:
                next_state = next_s
                break
            
        if not done:
            next_state = self.n_step_buffer[-1][3]
            
        return reward, next_state, done

    def push(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < N_STEP:
            return
            
        reward, next_state, done = self._get_n_step_info()
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Transition(state, action, reward, next_state, done, max_priority)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        if len(self.buffer) < batch_size:
            return None

        # Get priorities and handle NaN/inf values
        priorities = self.priorities[:len(self.buffer)]
        
        # Replace NaN and inf values with small positive number
        priorities = np.where(np.isfinite(priorities), priorities, 1e-8)
        priorities = np.maximum(priorities, 1e-8)  # Ensure all priorities are positive
        
        probs = priorities ** ALPHA
        
        # Normalize probabilities and handle edge cases
        prob_sum = probs.sum()
        if prob_sum <= 0 or not np.isfinite(prob_sum):
            # Fallback to uniform sampling if probabilities are invalid
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= prob_sum
        
        # Final safety check
        probs = np.where(np.isfinite(probs), probs, 1.0 / len(probs))
        probs /= probs.sum()  # Renormalize

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights with safety checks
        selected_probs = probs[indices]
        weights = (len(self.buffer) * selected_probs) ** (-beta)
        
        # Handle potential NaN/inf in weights
        weights = np.where(np.isfinite(weights), weights, 1.0)
        max_weight = weights.max()
        if max_weight <= 0 or not np.isfinite(max_weight):
            weights = np.ones_like(weights)
        else:
            weights /= max_weight
            
        weights = torch.FloatTensor(weights).to(device)

        # Convert dictionary observations to tensors with consistent shapes
        states = {
            'agent_orient': torch.FloatTensor(np.array([s.state['agent_orient'] for s in samples])).to(device),  # [batch_size, 1]
            'knowledge_map': torch.FloatTensor(np.stack([s.state['knowledge_map'] for s in samples])).to(device)  # [batch_size, H, W]
        }
        actions = torch.LongTensor([s.action for s in samples]).to(device)
        rewards = torch.FloatTensor([s.reward for s in samples]).to(device)
        next_states = {
            'agent_orient': torch.FloatTensor(np.array([s.next_state['agent_orient'] for s in samples])).to(device),  # [batch_size, 1]
            'knowledge_map': torch.FloatTensor(np.stack([s.next_state['knowledge_map'] for s in samples])).to(device)  # [batch_size, H, W]
        }
        dones = torch.FloatTensor([s.done for s in samples]).to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            # Ensure priority is finite and positive
            if np.isfinite(priority) and priority > 0:
                self.priorities[idx] = priority
            else:
                # Use small positive value as fallback
                self.priorities[idx] = 1e-8

    def __len__(self) -> int:
        return len(self.buffer)

def project_distribution(next_dist, rewards, dones, support, delta_z, gamma):
    batch_size = len(rewards)
    atoms = len(support)
    
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    support = support.unsqueeze(0)
    
    Tz = rewards + (1 - dones) * gamma * support
    Tz = Tz.clamp(min=V_MIN, max=V_MAX)
    b = (Tz - V_MIN) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    
    # Clamp indices to valid range [0, atoms-1] - essential for GPU/CPU consistency
    l = l.clamp(0, atoms - 1)
    u = u.clamp(0, atoms - 1)
    
    proj_dist = torch.zeros_like(next_dist)
    
    # Vectorized version - much faster than nested loops
    # Create index tensors for scatter operations
    batch_indices = torch.arange(batch_size, device=next_dist.device).unsqueeze(1).expand(-1, atoms)
    
    # Flatten for scatter operations
    batch_flat = batch_indices.flatten()
    l_flat = l.flatten()
    u_flat = u.flatten()
    next_dist_flat = next_dist.flatten()
    b_flat = b.flatten()
    
    # Calculate weights
    # Original buggy weights:
    # l_weight = (u_flat - b_flat) * next_dist_flat
    # u_weight = (b_flat - l_flat) * next_dist_flat

    # Corrected weights to handle integer b_flat cases
    is_exact_match = (l_flat == u_flat)
    
    # Coefficient for the lower bin l
    l_coeff = u_flat.float() - b_flat # Use .float() for b_flat to ensure float arithmetic
    # Coefficient for the upper bin u
    u_coeff = b_flat - l_flat.float() # Use .float() for l_flat

    l_weight = torch.where(is_exact_match, torch.ones_like(l_coeff), l_coeff) * next_dist_flat
    u_weight = torch.where(is_exact_match, torch.zeros_like(u_coeff), u_coeff) * next_dist_flat
    
    # Use scatter_add for vectorized accumulation
    proj_dist.view(-1).scatter_add_(0, batch_flat * atoms + l_flat, l_weight)
    proj_dist.view(-1).scatter_add_(0, batch_flat * atoms + u_flat, u_weight)
            
    return proj_dist

def plot_training_metrics(metrics: Dict[str, List[float]], log_dir: str):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Rainbow DQN Training Metrics")

    # Plot episode rewards
    axes[0, 0].plot(metrics["episode_rewards"])
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")

    # Plot episode lengths
    axes[0, 1].plot(metrics["episode_lengths"])
    axes[0, 1].set_title("Episode Lengths")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")

    # Plot coverage ratio
    if metrics["coverage_ratio"]:
        axes[0, 2].plot(metrics["coverage_ratio"])
        axes[0, 2].set_title("Coverage Ratio")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Ratio")

        # Plot path efficiency
        axes[1, 0].plot(metrics["path_efficiency"])
        axes[1, 0].set_title("Path Efficiency")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Efficiency")

        # Plot revisit ratio
        axes[1, 1].plot(metrics["revisit_ratio"])
        axes[1, 1].set_title("Revisit Ratio")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Ratio")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_metrics.png"))
    plt.close()

def print_section_header(title):
    print("\n" + "="*50)
    print(f"{title:^50}")
    print("="*50)

def print_metrics(metrics, prefix=""):
    print(f"{prefix}Metrics:")
    print("-"*30)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{key:20}: {value:.4f}")
        else:
            print(f"{prefix}{key:20}: {value}")

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    print(f' {prefix} {percent}% {suffix}', end=' ')
    if iteration == total:
        print()

def save_model(model, full_save_path):
    """Save model state dict to file"""
    # Save model state and training info
    save_dict = {
        'state_dict': model.state_dict(),
        'device': str(next(model.parameters()).device),
        'training': model.training
    }
    torch.save(save_dict, full_save_path)
    print(f"\nModel saved to: {full_save_path} (device: {save_dict['device']}, training: {save_dict['training']})")

def load_model(model, full_load_path):
    """Load model state dict from file"""
    if os.path.exists(full_load_path):
        # Load with device mapping
        checkpoint = torch.load(full_load_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            saved_device = checkpoint.get('device', 'unknown')
            saved_training = checkpoint.get('training', 'unknown')
            print(f"\nModel loaded from: {full_load_path}")
            print(f"Original device: {saved_device}, Current device: {device}")
            print(f"Original training mode: {saved_training}, Current training mode: {model.training}")
        else:
            # Handle old format
            model.load_state_dict(checkpoint)
            print(f"\nModel loaded from: {full_load_path} (old format)")
            print(f"Current device: {device}, Current training mode: {model.training}")
        
        return model
    else:
        print(f"\nNo saved model found at: {full_load_path}")
        return None

def get_log_dir(grid_size, dirt_num=5, wall_mode="none", custom_dir=None):
    """Create and return log directory with descriptive naming convention.
    
    Args:
        grid_size (tuple): Grid dimensions
        dirt_num (int): Number of dirt clusters
        wall_mode (str): Wall configuration ("none", "random", "hardcoded")
        custom_dir (str, optional): Custom directory path to use instead of default
    """
    if custom_dir is not None:
        log_dir = custom_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(BASE_LOG_DIR, f"rainbow_dqn_{grid_size[0]}x{grid_size[1]}_dirt_{dirt_num}_{wall_mode}_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def train(env_id: str = "Vacuum-v0", grid_size: tuple = (6, 6), total_timesteps: int = MAX_TIMESTEPS, save_freq: int = 10000, walls=None, env=None, seed=None, output_dir: str = None, dirt_num: int = 5, wrapper: str = 'smart'):
    if output_dir is None:
        # Fallback if no output_dir is provided, though __main__ should provide it.
        wall_mode_name = "hardcoded" if walls is not None and walls != "" else ("random" if walls == "" else "none")
        output_dir = get_log_dir(grid_size, dirt_num, wall_mode_name)
    else:
        os.makedirs(output_dir, exist_ok=True) # Ensure it exists if provided
    
    # Set random seeds if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    print_section_header("Starting Training")
    print(f"Environment: {env_id}")
    print(f"Grid Size: {grid_size[0]}x{grid_size[1]}")
    print(f"Wall Mode: {'hardcoded' if walls is not None else 'random'}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Saving model every {save_freq} steps")
    print(f"Random seed: {seed}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    print(f"TensorBoard logs will be saved to: {os.path.join(output_dir, 'tensorboard')}")
    
    # Save metadata file with training command and evaluation command
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, "w") as f:
        f.write("Training Metadata\n")
        f.write("=================\n\n")
        
        # Original training command
        import sys
        original_command = " ".join(sys.argv)
        f.write(f"Training Command:\n{original_command}\n\n")
        
        # Evaluation commands for different models
        eval_best = f"python rainbow_dqn.py --mode eval --model_path \"{os.path.join(output_dir, 'best_model.pth')}\" --grid_size {grid_size[0]} {grid_size[1]} --dirt_num {dirt_num}"
        eval_final = f"python rainbow_dqn.py --mode eval --model_path \"{os.path.join(output_dir, 'final_model.pth')}\" --grid_size {grid_size[0]} {grid_size[1]} --dirt_num {dirt_num}"
        
        # Add wall mode if applicable
        if walls is not None and walls != "":
            wall_mode_name = "hardcoded" if walls != "" else "random"
            eval_best += f" --wall_mode {wall_mode_name}"
            eval_final += f" --wall_mode {wall_mode_name}"
        elif walls == "":
            eval_best += " --wall_mode random"
            eval_final += " --wall_mode random"
        
        # Add evaluation episodes and temperature (common settings)
        eval_best += " --eval_episodes 5 --temperature 1.0"
        eval_final += " --eval_episodes 5 --temperature 1.0"
        
        f.write("Evaluation Commands:\n")
        f.write("-------------------\n")
        f.write(f"Best Model:\n{eval_best}\n\n")
        f.write(f"Final Model:\n{eval_final}\n\n")
        
        # Training configuration summary
        f.write("Configuration Summary:\n")
        f.write("---------------------\n")
        f.write(f"Grid Size: {grid_size[0]}x{grid_size[1]}\n")
        f.write(f"Dirt Clusters: {dirt_num}\n")
        wall_mode_display = "None" if walls is None else ("Random" if walls == "" else "Hardcoded")
        f.write(f"Wall Mode: {wall_mode_display}\n")
        f.write(f"Total Timesteps: {total_timesteps}\n")
        f.write(f"Max Steps per Episode: {MAX_STEPS}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Training Device: {device}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Buffer Size: {BUFFER_SIZE}\n")
        
    print(f"Metadata saved to: {metadata_file}")
    
    # Create environment if not provided
    if env is None:
        # Create and wrap environment with metrics and additional wrappers
        env = gym.make(env_id, grid_size=grid_size, dirt_num=dirt_num)
        env = TimeLimit(env, max_episode_steps=MAX_STEPS)
        
        # Apply chosen wrapper
        if wrapper == 'smart':
            env = SmartExplorationWrapper(env)  # Advanced dense rewards
        else:
            env = DumbWrapper(env)  # Simple reward shaping
            
        env = MetricWrapper(env)
    
    print("\nEnvironment Wrappers:")
    print(f"- TimeLimit: {MAX_STEPS} steps")
    if wrapper == 'smart':
        print("- SmartExplorationWrapper: Advanced dense rewards enabled")
    else:
        print("- DumbWrapper: Simple reward shaping enabled")
    print("- MetricWrapper: enabled")
    
    # Initialize metrics dictionary
    training_metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "coverage_ratio": [],
        "path_efficiency": [],
        "revisit_ratio": [],
        "cleaning_time": [],
        "losses": []
    }
    
    print_section_header("Network Architecture")
    obs_dim = {
        'agent_orient': 1,
        'knowledge_map': grid_size[0] * grid_size[1] # Flattened 2D map size
    }
    n_actions = env.action_space.n
    print(f"Observation dimensions: {obs_dim}")
    print(f"Number of actions: {n_actions}")
    print(f"Number of atoms: {N_ATOMS}")
    print(f"Knowledge map: {grid_size[0]}x{grid_size[1]} = {grid_size[0] * grid_size[1]} values")
    print("Network includes: agent_orient, knowledge_map")

    # Initialize networks and optimizer
    policy_net = RainbowDQN(obs_dim, n_actions).to(device)
    target_net = RainbowDQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    
    # Try to load latest checkpoint if exists
    latest_model_path = os.path.join(output_dir, "latest_model.pth")
    if os.path.exists(latest_model_path):
        print(f"Attempting to load existing model from {latest_model_path}")
        latest_model = load_model(policy_net, latest_model_path)
        if latest_model is not None:
            policy_net = latest_model
            target_net.load_state_dict(policy_net.state_dict())
            print("Successfully loaded existing model.")
        else:
            print("Failed to load existing model, starting from scratch.")
    else:
        print("No existing model found at specified output_dir, starting from scratch.")
    
    # Initialize replay buffer
    buffer = PrioritizedReplayBuffer(BUFFER_SIZE)

    print_section_header("Training Parameters")
    print(f"Learning rate: {LR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Buffer size: {BUFFER_SIZE}")
    print(f"Training device: {device}")
    print(f"Gamma: {GAMMA}")
    print(f"Start training at: {START_TRAINING} steps")
    print(f"Target network update frequency: {UPDATE_TARGET_EVERY}")
    print(f"Noise annealing: 1.0 -> 0.1 over {total_timesteps} steps")
    print("*** OPTIMIZED FOR DENSE REWARD SHAPING ***")
    print("\nStarting training loop...")

    obs, _ = env.reset(options={"walls": walls})
    episode_reward = 0
    episode_base_reward = 0  # Track base rewards separately
    episode_exploration_reward = 0  # Track exploration rewards separately
    episode_count = 0
    episode_steps = 0
    total_loss = 0
    num_updates = 0
    beta = BETA
    noise_scale = 1.0  # Initialize noise scale
    best_reward = float('-inf')
    action_counts = [0, 0, 0]  # Track action distribution [forward, left, right]

    for step in range(total_timesteps):
        if step % 100 == 0:
            print_progress_bar(step, total_timesteps, prefix='Progress:', suffix='Complete', length=40)
            
        episode_steps += 1
        
        # Implement noise annealing: start at 1.0, end at 0.1
        noise_scale = max(0.1, 1.0 - 0.9 * (step / total_timesteps))
        policy_net.set_noise_scale(noise_scale)
        policy_net.reset_noise()
        
        action = policy_net.act(obs)
        action_counts[action] += 1  # Track action distribution
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track reward components if available
        base_reward = info.get('base_reward', reward)  # Fallback to total reward if not available
        exploration_reward = info.get('exploration_reward', 0.0)
        
        # Debug reward breakdown for first 5 steps of first 2 episodes
        if episode_count < 2 and episode_steps <= 5:
            action_names = ["Forward", "Left", "Right"]
            print(f"Step {episode_steps}: {action_names[action]} | Base: {base_reward:.3f}, Exploration: {exploration_reward:.3f}, Total: {reward:.3f}")
        
        buffer.push(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward  # Track total reward for logging
        episode_base_reward += base_reward  # Track base reward separately
        episode_exploration_reward += exploration_reward  # Track exploration reward separately

        if done:
            episode_count += 1
            avg_loss = total_loss/max(1, num_updates)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Train/Episode_Reward', episode_reward, episode_count)
            writer.add_scalar('Train/Episode_Base_Reward', episode_base_reward, episode_count)
            writer.add_scalar('Train/Episode_Exploration_Reward', episode_exploration_reward, episode_count)
            writer.add_scalar('Train/Episode_Length', episode_steps, episode_count)
            writer.add_scalar('Train/Average_Loss', avg_loss, episode_count)
            writer.add_scalar('Train/Noise_Scale', noise_scale, episode_count)
            
            if "coverage_ratio" in info:
                writer.add_scalar('Train/Coverage_Ratio', info["coverage_ratio"], episode_count)
                writer.add_scalar('Train/Path_Efficiency', info["path_efficiency"], episode_count)
                writer.add_scalar('Train/Revisit_Ratio', info["revisit_ratio"], episode_count)
            
            # Save best model if this is the best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                save_model(policy_net, os.path.join(output_dir, "best_model.pth"))
                print(f"\nNew best reward: {best_reward:.2f}")
            
            # Log metrics to dictionary
            training_metrics["episode_rewards"].append(episode_reward)
            training_metrics["episode_lengths"].append(episode_steps)
            if "coverage_ratio" in info:
                training_metrics["coverage_ratio"].append(info["coverage_ratio"])
                training_metrics["path_efficiency"].append(info["path_efficiency"])
                training_metrics["revisit_ratio"].append(info["revisit_ratio"])
            
            print_section_header(f"Episode {episode_count} Complete")
            print(f"Steps: {episode_steps}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"  Base Reward: {episode_base_reward:.2f}")
            print(f"  Exploration Reward: {episode_exploration_reward:.2f}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Noise Scale: {noise_scale:.3f}")
            total_actions = sum(action_counts)
            if total_actions > 0:
                action_dist = [count/total_actions for count in action_counts]
                print(f"Action Distribution - Forward: {action_dist[0]:.2%}, Left: {action_dist[1]:.2%}, Right: {action_dist[2]:.2%}")
            if "coverage_ratio" in info:
                print("\nCleaning Metrics:")
                print(f"Coverage: {info['coverage_ratio']:.2%}")
                print(f"Path Efficiency: {info['path_efficiency']:.2f}")
                print(f"Revisit Ratio: {info['revisit_ratio']:.2f}")
            
            obs, _ = env.reset(options={"walls": walls})
            episode_reward = 0
            episode_base_reward = 0
            episode_exploration_reward = 0
            episode_steps = 0
            total_loss = 0
            num_updates = 0
            action_counts = [0, 0, 0]  # Reset action distribution

        # Training
        if len(buffer) >= START_TRAINING:
            beta = min(1.0, beta + BETA_INCREMENT)
            batch = buffer.sample(BATCH_SIZE, beta)
            if batch is not None:
                states, actions, rewards, next_states, dones, indices, weights = batch

                # Current Q-distribution
                current_q_dist = policy_net(states)
                current_q_dist = current_q_dist[range(BATCH_SIZE), actions]

                # Next Q-distribution (from target network)
                with torch.no_grad():
                    next_actions = (policy_net(next_states) * policy_net.support).sum(2).argmax(1)
                    next_q_dist = target_net(next_states)
                    next_q_dist = next_q_dist[range(BATCH_SIZE), next_actions]

                    target_q_dist = project_distribution(
                        next_q_dist, rewards, dones,
                        policy_net.support,
                        policy_net.delta_z,
                        GAMMA
                    )

                # Add epsilon to prevent log(0) which causes NaN
                eps = 1e-8
                log_current_q_dist = torch.clamp(current_q_dist, min=eps).log()
                loss = -(target_q_dist * log_current_q_dist).sum(1)
                
                # Check for NaN in loss and handle it
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("Warning: NaN or Inf detected in loss, skipping this batch")
                    continue
                
                # Convert to priorities with safety checks for NaN/inf values
                priorities = loss.detach().cpu().numpy()
                priorities = np.where(np.isfinite(priorities), priorities, 1e-8)
                priorities = np.maximum(priorities, 1e-8)  # Ensure all priorities are positive
                
                weighted_loss = loss * weights
                if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
                    print("Warning: NaN or Inf detected in weighted loss, skipping this batch")
                    continue
                    
                loss = weighted_loss.mean()
                
                # Log training loss to TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), step)
                
                total_loss += loss.item()
                num_updates += 1

                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients before stepping
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print("Warning: NaN or Inf detected in gradients, skipping optimizer step")
                    continue
                    
                optimizer.step()
                
                # Validate network parameters after update
                for name, param in policy_net.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"Warning: NaN or Inf detected in parameter {name}, stopping training")
                        print("Network has become unstable. Consider reducing learning rate or batch size.")
                        return policy_net

                buffer.update_priorities(indices, priorities)

            if step % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"\nUpdated target network at step {step}")

        # Save model periodically
        if step > 0 and step % save_freq == 0:
            save_model(policy_net, os.path.join(output_dir, f"model_step_{step}.pth"))
            save_model(policy_net, os.path.join(output_dir, "latest_model.pth"))  # Always keep the latest model

    print_section_header("Training Complete")
    print(f"Total episodes: {episode_count}")
    print("\nSaving metrics and plots...")
    
    # Save final model
    save_model(policy_net, os.path.join(output_dir, "final_model.pth"))
    
    # Save training metrics
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(training_metrics, f)

    # Plot training metrics
    plot_training_metrics(training_metrics, output_dir)
    print(f"Metrics saved to: {metrics_file}")
    print(f"Plots saved to: {output_dir}")
    
    # Close TensorBoard writer
    writer.close()
    env.close()
    return policy_net

def evaluate(policy_net, env_id="Vacuum-v0", grid_size=(6, 6), episodes=1, render=True, walls=None, env=None, output_dir: str = None, dirt_num: int = 5, temperature: float = 1.0, wrapper: str = 'smart'):
    if output_dir is None:
        # Fallback if no output_dir is provided, though __main__ should provide it.
        wall_mode_name = "hardcoded" if walls is not None and walls != "" else ("random" if walls == "" else "none")
        output_dir = get_log_dir(grid_size, dirt_num, wall_mode_name)
    else:
        os.makedirs(output_dir, exist_ok=True) # Ensure it exists if provided

    print_section_header("Starting Evaluation")
    print(f"Environment: {env_id}")
    print(f"Grid Size: {grid_size[0]}x{grid_size[1]}")
    print(f"Wall Mode: {'hardcoded' if walls is not None else 'random'}")
    print(f"Number of episodes: {episodes}")
    print(f"Rendering: {'enabled' if render else 'disabled'}")
    print(f"Device: {next(policy_net.parameters()).device}")
    print(f"Training mode before eval(): {policy_net.training}")
    print(f"Action selection: Sampling from Q-value distribution (temperature={temperature})")
    
    # Create environment if not provided
    if env is None:
        # Create and wrap environment with same wrappers as training
        env = gym.make(env_id, grid_size=grid_size, dirt_num=dirt_num)
        env = TimeLimit(env, max_episode_steps=MAX_STEPS)
        
        # Apply chosen wrapper
        if wrapper == 'smart':
            env = SmartExplorationWrapper(env)  # Advanced dense rewards
        else:
            env = DumbWrapper(env)  # Simple reward shaping
            
        # env = ExploitationPenaltyWrapper(env, time_penalty=-0.0020989802390739463, stay_penalty=-0.05073560696895504)  # From Optuna study
        env = MetricWrapper(env)
    
    print("\nEnvironment Wrappers:")
    print(f"- TimeLimit: {MAX_STEPS} steps")
    if wrapper == 'smart':
        print("- SmartExplorationWrapper: Advanced dense rewards enabled")
    else:
        print("- DumbWrapper: Simple reward shaping enabled")
    print("- MetricWrapper: enabled")
    print("- Internal stuck counter: disabled")
    
    # Ensure model is in evaluation mode for deterministic behavior
    policy_net.eval()
    print(f"Training mode after eval(): {policy_net.training}")
    
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "coverage_ratio": [],
        "path_efficiency": [],
        "revisit_ratio": [],
        "cleaning_time": [],
        "action_distributions": []  # Store action distribution for each episode
    }

    # Evaluate episodes in normal order
    for ep in range(episodes):
        print_section_header(f"Evaluation Episode {ep + 1}")
        
        # Memory debug info
        import gc
        print(f"Memory before episode: {len(gc.get_objects())} objects in memory")
        
        obs, _ = env.reset(options={"walls": walls})
        done = False
        total_reward = 0
        steps = 0
        eval_action_counts = [0, 0, 0]  # Track actions in this episode
        
        # Initialize rendering - separate video recording from final frame capture
        frames = [] if render else None
        final_frame = None
        video_recording = render  # Track video recording separately from final frame
        should_capture_final = render  # Always try to capture final frame

        while not done:
            # Record frame for video (with limits)
            if video_recording:
                try:
                    frame_data = env.unwrapped.render_frame()
                    frames.append(frame_data)
                    
                    # Limit frames to prevent excessive memory usage for video
                    if len(frames) > 5000:  # Increased limit
                        print(f"Warning: Episode too long ({len(frames)} frames), stopping video recording but continuing final frame capture")
                        video_recording = False
                except Exception as e:
                    print(f"Warning: Failed to render frame for video at step {steps}: {e}")
                    video_recording = False
            
            # Always try to capture final frame (separate from video)
            if should_capture_final and steps > 0 and steps % 50 == 0:  # Sample every 50 steps for final frame
                try:
                    final_frame = env.unwrapped.render_frame()
                except Exception as e:
                    if steps == 1:  # Only warn on first attempt
                        print(f"Warning: Failed to render final frame at step {steps}: {e}")
                        should_capture_final = False
            
            # Use deterministic actions during evaluation (no noise, no epsilon)
            with torch.no_grad():  # Ensure no gradients during evaluation
                action = policy_net.act(obs, epsilon=0.0, temperature=temperature)
                eval_action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1

        # Ensure we have a final frame (capture at end if not already captured)
        if should_capture_final and final_frame is None:
            try:
                final_frame = env.unwrapped.render_frame()
                print(f"Captured final frame at episode end (step {steps})")
            except Exception as e:
                print(f"Warning: Failed to capture final frame at episode end: {e}")

        # Calculate action distribution
        total_eval_actions = sum(eval_action_counts)
        if total_eval_actions > 0:
            eval_action_dist = [count/total_eval_actions for count in eval_action_counts]
        else:
            eval_action_dist = [0.0, 0.0, 0.0]  # Default if no actions taken

        # Log metrics to dictionary
        episode_metrics = {
            "episode": ep + 1,
            "steps": steps,
            "total_reward": total_reward,
            "action_distribution": {
                "forward": eval_action_dist[0],
                "left": eval_action_dist[1], 
                "right": eval_action_dist[2]
            },
            "coverage_ratio": info.get('coverage_ratio', 0),
            "path_efficiency": info.get('path_efficiency', 0),
            "revisit_ratio": info.get('revisit_ratio', 0)
        }
        
        # Store in main metrics lists
        metrics["episode_rewards"].append(total_reward)
        metrics["episode_lengths"].append(steps)
        metrics["action_distributions"].append(eval_action_dist)
        if "coverage_ratio" in info:
            metrics["coverage_ratio"].append(info["coverage_ratio"])
            metrics["path_efficiency"].append(info["path_efficiency"])
            metrics["revisit_ratio"].append(info["revisit_ratio"])

        print("\nEpisode Results:")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        if total_eval_actions > 0:
            print(f"Action Distribution - Forward: {eval_action_dist[0]:.2%}, Left: {eval_action_dist[1]:.2%}, Right: {eval_action_dist[2]:.2%}")
        print("\nCleaning Metrics:")
        print(f"Coverage: {info.get('coverage_ratio', 0):.2%}")
        print(f"Path Efficiency: {info.get('path_efficiency', 0):.2f}")
        print(f"Revisit Ratio: {info.get('revisit_ratio', 0):.2f}")
        
        # Save individual episode metrics to separate file
        episode_metrics_file = os.path.join(output_dir, f"episode_{ep + 1}_metrics.json")
        with open(episode_metrics_file, "w") as f:
            json.dump(episode_metrics, f, indent=2)
        print(f"Episode metrics saved to: {episode_metrics_file}")
        
        # Save video from collected frames (even single frame episodes)
        if frames and len(frames) > 0:
            video_path = os.path.join(output_dir, f"eval_ep_{ep + 1}.mp4")
            print(f"\nCreating video from {len(frames)} frames...")
            
            try:
                fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
                im = ax.imshow(frames[0])
                ax.axis("off")
                ax.set_title(f"Episode {ep + 1} (Steps: {steps}, Reward: {total_reward:.1f})")

                def update(i):
                    im.set_array(frames[i])
                    return [im]

                ani = animation.FuncAnimation(
                    fig, update, frames=len(frames), interval=100, blit=True
                )

                # Save with optimized settings
                ani.save(video_path, writer="ffmpeg", fps=10, bitrate=1000)
                plt.close(fig)
                print(f"Video saved to: {video_path}")
                
            except Exception as e:
                print(f"Warning: Failed to create video: {e}")
                plt.close('all')
        else:
            print(f"\nNo frames collected for episode {ep + 1} - skipping video")
        
        # Save final frame as static image
        if final_frame is not None:
            image_path = os.path.join(output_dir, f"eval_ep_{ep + 1}_final.png")
            print(f"Saving final frame...")
            
            try:
                plt.figure(figsize=(8, 8))
                plt.imshow(final_frame)
                plt.axis("off")
                plt.title(f"Episode {ep + 1} Final State (Steps: {steps}, Reward: {total_reward:.1f})")
                plt.savefig(image_path, bbox_inches="tight", pad_inches=0, dpi=150)
                plt.close()
                print(f"Final frame saved to: {image_path}")
                
            except Exception as e:
                print(f"Warning: Failed to save final frame: {e}")
                plt.close('all')
        
        # Aggressive memory cleanup
        if frames:
            # Clear references to frame data
            for i in range(len(frames)):
                frames[i] = None
            frames.clear()
            del frames
        
        # Clear final frame reference
        if final_frame is not None:
            final_frame = None
            del final_frame
        
        # Force matplotlib cleanup
        plt.close('all')
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Memory debug info
        print(f"Memory after episode cleanup: {len(gc.get_objects())} objects in memory")

    # Calculate and print summary statistics
    print_section_header("Evaluation Summary")
    summary_metrics = {
        "Average Reward": sum(metrics["episode_rewards"]) / episodes,
        "Average Steps": sum(metrics["episode_lengths"]) / episodes,
        "Average Coverage": sum(metrics["coverage_ratio"]) / episodes if metrics["coverage_ratio"] else 0,
        "Average Path Efficiency": sum(metrics["path_efficiency"]) / episodes if metrics["path_efficiency"] else 0,
        "Average Revisit Ratio": sum(metrics["revisit_ratio"]) / episodes if metrics["revisit_ratio"] else 0,
        "Best Episode Reward": max(metrics["episode_rewards"]),
        "Worst Episode Reward": min(metrics["episode_rewards"])
    }
    
    print_metrics(summary_metrics)

    # Save evaluation metrics with detailed breakdown
    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
    
    # Add detailed episode-by-episode breakdown to main metrics
    detailed_metrics = metrics.copy()
    detailed_metrics["episodes_detail"] = []
    
    for i in range(episodes):
        episode_detail = {
            "episode": i + 1,
            "steps": metrics["episode_lengths"][i],
            "total_reward": metrics["episode_rewards"][i],
            "action_distribution": {
                "forward": metrics["action_distributions"][i][0],
                "left": metrics["action_distributions"][i][1],
                "right": metrics["action_distributions"][i][2]
            }
        }
        
        # Add cleaning metrics if available
        if i < len(metrics["coverage_ratio"]):
            episode_detail["coverage_ratio"] = metrics["coverage_ratio"][i]
            episode_detail["path_efficiency"] = metrics["path_efficiency"][i]
            episode_detail["revisit_ratio"] = metrics["revisit_ratio"][i]
        
        detailed_metrics["episodes_detail"].append(episode_detail)
    
    with open(metrics_file, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"\nDetailed evaluation metrics saved to: {metrics_file}")
    print(f"Individual episode files: episode_1_metrics.json, episode_2_metrics.json, ...")

    env.close()
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or evaluate Rainbow DQN')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                      help='Whether to train a new model or evaluate an existing one')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                      help='Path to model file for evaluation (e.g., logs/run_folder/best_model.pth or an absolute path)')
    parser.add_argument('--timesteps', type=int, default=MAX_TIMESTEPS,
                      help='Number of timesteps to train for')
    parser.add_argument('--eval_episodes', type=int, default=5,
                      help='Number of episodes to evaluate for, at most')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[6, 6],
                      help='Grid size as two integers (e.g., 6 6 for 6x6 grid)')
    parser.add_argument('--wall_mode', choices=['none', 'random', 'hardcoded'], default='none',
                      help='Wall layout: "none" (no walls), "random" (generate random rooms), or "hardcoded" (only applies to 40x30 grid)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--dirt_num', type=int, default=5,
                      help='Number of dirt clusters to generate (0 means all non-obstacle tiles are dirt)')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Temperature for action sampling during evaluation (1.0=normal, >1.0=more random, <1.0=more deterministic)')
    parser.add_argument('--wrapper', choices=['dumb', 'smart'], default='smart',
                      help='Exploration wrapper to use: "dumb" (minimal rewards) or "smart" (advanced dense rewards)')
    args = parser.parse_args()

    grid_size = tuple(args.grid_size)

    # Set up wall configuration based on mode
    walls = None
    if args.wall_mode == 'hardcoded' and grid_size == (40, 30):
        from run import generate_1b1b_layout_grid
        walls = generate_1b1b_layout_grid()
    elif args.wall_mode == 'random':
        # Pass empty string to trigger random room generation in env.reset()
        # This will cause the else branch in env.py reset() to execute generate_random_rooms()
        walls = ""
    elif args.wall_mode == 'none':
        # Keep walls = None to skip wall generation entirely
        walls = None

    if args.mode == 'train':
        # Training mode: train a new model then evaluate it
        wall_mode_name = "hardcoded" if args.wall_mode == 'hardcoded' else ("random" if args.wall_mode == 'random' else "none")
        run_log_dir = get_log_dir(grid_size, args.dirt_num, wall_mode_name)
        print(f"All outputs for this run will be saved to: {run_log_dir}")

        # Train a new model
        model = train(total_timesteps=args.timesteps, grid_size=grid_size, walls=walls, 
                        seed=args.seed, output_dir=run_log_dir, dirt_num=args.dirt_num, wrapper=args.wrapper)
        # Evaluate the trained model
        evaluate(model, grid_size=grid_size, episodes=args.eval_episodes, walls=walls, 
                output_dir=run_log_dir, dirt_num=args.dirt_num, temperature=args.temperature, wrapper=args.wrapper)
    
    elif args.mode == 'eval':
        # Evaluation mode: load existing model and evaluate only
        print_section_header("Evaluation Mode - Loading Pre-trained Model")
        
        # Determine model path
        if os.path.isabs(args.model_path):
            model_path = args.model_path
            eval_log_dir = os.path.dirname(model_path)
        else:
            # Relative path - look in logs directory or current directory
            if os.path.exists(args.model_path):
                model_path = args.model_path
                eval_log_dir = os.path.dirname(os.path.abspath(model_path))
            else:
                # Try in logs directory
                model_path = os.path.join("logs", args.model_path)
                if not os.path.exists(model_path):
                    print(f"Error: Model file not found at {args.model_path} or {model_path}")
                    exit(1)
                eval_log_dir = os.path.dirname(model_path)
        
        print(f"Loading model from: {model_path}")
        print(f"Evaluation outputs will be saved to: {eval_log_dir}")
        
        # Create model architecture
        obs_dim = {
            'agent_orient': 1,
            'knowledge_map': grid_size[0] * grid_size[1]
        }
        # We need to create a dummy environment to get the number of actions
        dummy_env = gym.make("Vacuum-v0", grid_size=grid_size, dirt_num=args.dirt_num)
        n_actions = dummy_env.action_space.n
        dummy_env.close()
        
        model = RainbowDQN(obs_dim, n_actions).to(device)
        
        # Load the trained model
        loaded_model = load_model(model, model_path)
        if loaded_model is None:
            print(f"Error: Failed to load model from {model_path}")
            exit(1)
        
        model = loaded_model
        
        # Evaluate the loaded model
        evaluate(model, grid_size=grid_size, episodes=args.eval_episodes, walls=walls, 
                output_dir=eval_log_dir, dirt_num=args.dirt_num, temperature=args.temperature, wrapper=args.wrapper)


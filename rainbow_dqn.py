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
from eval import MetricWrapper, compute_coverage_ratio, compute_redundancy_rate, compute_revisit_ratio, plot_comparison_metrics
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit
from wrappers import ExplorationBonusWrapper, ExploitationPenaltyWrapper
from datetime import datetime

# Register custom environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

# Create base log directory
BASE_LOG_DIR = "./logs"
os.makedirs(BASE_LOG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Constants
MAX_STEPS = 3000  # Maximum steps per episode
MAX_TIMESTEPS = 500000  # Maximum total timesteps for training

# Hyperparameters
GAMMA = 0.9890489964807936  # From Optuna study
LR = 1.0161969492721541e-05  # From Optuna study
BATCH_SIZE = 80  # From Optuna study
BUFFER_SIZE = int(5e5)
START_TRAINING = 1000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
UPDATE_TARGET_EVERY = 2000
N_ATOMS = 41  # From Optuna study
V_MIN = -19.97256170921028  # From Optuna study
V_MAX = 19.744371778269656  # From Optuna study
N_STEP = 5  # From Optuna study
ALPHA = 0.6843270904544203  # PER parameters from Optuna study
BETA = 0.560936297194832  # From Optuna study
BETA_INCREMENT = 0.0019984316362548897  # From Optuna study

# Reward normalization parameters
REWARD_SCALING = 1.0
REWARD_CLIP = 10.0

class RunningMeanStd:
    """Tracks running mean and standard deviation of rewards for normalization"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """Normalizes rewards using running statistics"""
        std = np.sqrt(self.var)
        return np.clip((x - self.mean) / (std + 1e-8), -REWARD_CLIP, REWARD_CLIP)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'priority'))

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

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

    def forward(self, x):
        if self.training:  # During training, use noise for exploration
            return F.linear(x, 
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:  # During evaluation, use mean values for deterministic behavior
            return F.linear(x, self.weight_mu, self.bias_mu)

class RainbowDQN(nn.Module):
    def __init__(self, obs_dim: Dict[str, int], n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.atoms = N_ATOMS
        self.support = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(device)
        self.delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

        # Process different observation components
        self.agent_pos_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )
        
        self.agent_orient_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
        
        self.local_view_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )

        # Combine all features
        combined_size = 64 + 32 + 32

        # Advantage and Value streams with NoisyLinear
        self.advantage_hidden = NoisyLinear(combined_size, 128)
        self.advantage = NoisyLinear(128, n_actions * N_ATOMS)
        
        self.value_hidden = NoisyLinear(combined_size, 128)
        self.value = NoisyLinear(128, N_ATOMS)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = x['agent_pos'].shape[0]
        
        # Process each input component
        pos_features = self.agent_pos_net(x['agent_pos'].float())  # [batch_size, 64]
        
        # Handle orientation - ensure it's 2D
        orient = x['agent_orient'].float()
        if orient.dim() == 1:
            orient = orient.unsqueeze(-1)  # Add feature dimension
        elif orient.dim() == 3:
            orient = orient.squeeze(1)  # Remove extra dimension if present
        orient_features = self.agent_orient_net(orient)  # [batch_size, 32]
        
        # Handle local view
        view = x['local_view'].float()
        if view.dim() == 3:
            view = view.squeeze(1)  # Remove extra dimension if present
        view_features = self.local_view_net(view)  # [batch_size, 32]
        
        # Ensure all features have the same number of dimensions
        if pos_features.dim() != orient_features.dim():
            pos_features = pos_features.unsqueeze(1) if pos_features.dim() < orient_features.dim() else pos_features
            orient_features = orient_features.unsqueeze(1) if orient_features.dim() < pos_features.dim() else orient_features
            view_features = view_features.unsqueeze(1) if view_features.dim() < pos_features.dim() else view_features
        
        # Combine features
        combined = torch.cat([pos_features, orient_features, view_features], dim=1)
        
        # Dueling architecture with noisy layers
        advantage_hidden = F.relu(self.advantage_hidden(combined))
        advantage = self.advantage(advantage_hidden).view(batch_size, self.n_actions, self.atoms)
        
        value_hidden = F.relu(self.value_hidden(combined))
        value = self.value(value_hidden).view(batch_size, 1, self.atoms)
        
        # Combine value and advantage
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Get probabilities
        return F.softmax(q_dist, dim=2)

    def reset_noise(self):
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value_hidden.reset_noise()
        self.value.reset_noise()

    def act(self, state: Dict[str, np.ndarray], epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            # Convert numpy arrays to tensors and add batch dimension
            state = {
                'agent_pos': torch.FloatTensor(state['agent_pos']).unsqueeze(0).to(device),
                'agent_orient': torch.FloatTensor([state['agent_orient']]).to(device),
                'local_view': torch.FloatTensor(state['local_view']).unsqueeze(0).to(device)
            }
            
            dist = self(state)
            q_values = (dist * self.support).sum(dim=2)
            return q_values.argmax(1).item()

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
            'agent_pos': torch.FloatTensor(np.vstack([s.state['agent_pos'] for s in samples])).to(device),  # [batch_size, 2]
            'agent_orient': torch.FloatTensor([s.state['agent_orient'] for s in samples]).reshape(-1).to(device),  # [batch_size]
            'local_view': torch.FloatTensor(np.vstack([s.state['local_view'] for s in samples])).to(device)  # [batch_size, 3]
        }
        actions = torch.LongTensor([s.action for s in samples]).to(device)
        rewards = torch.FloatTensor([s.reward for s in samples]).to(device)
        next_states = {
            'agent_pos': torch.FloatTensor(np.vstack([s.next_state['agent_pos'] for s in samples])).to(device),  # [batch_size, 2]
            'agent_orient': torch.FloatTensor([s.next_state['agent_orient'] for s in samples]).reshape(-1).to(device),  # [batch_size]
            'local_view': torch.FloatTensor(np.vstack([s.next_state['local_view'] for s in samples])).to(device)  # [batch_size, 3]
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

def rollout_and_record(env, policy_net, filename="rainbow_run.mp4", max_steps=MAX_STEPS, walls=None):
    """Record a video of the agent's performance"""
    print(f"\nRecording video for {max_steps} steps...")
    obs, _ = env.reset(options={"walls": walls})
    frames = []

    for step in range(max_steps):
        # Render and save frame
        fig = env.render_frame()
        frames.append(fig)

        # Get action from policy
        action = policy_net.act(obs, epsilon=0.0)  # No exploration during recording
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    print(f"Recorded {len(frames)} frames")
    
    # Create animation
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=100, blit=True
    )

    # Save animation
    ani.save(filename, writer="ffmpeg")
    plt.close(fig)
    print(f"Video saved to: {filename}")

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

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
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

def get_log_dir(grid_size, custom_dir=None):
    """Create and return log directory based on grid size or custom directory.
    
    Args:
        grid_size (tuple): Grid dimensions
        custom_dir (str, optional): Custom directory path to use instead of default
    """
    if custom_dir is not None:
        log_dir = custom_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(BASE_LOG_DIR, f"rainbow_dqn_{grid_size[0]}x{grid_size[1]}_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def train(env_id: str = "Vacuum-v0", grid_size: tuple = (6, 6), total_timesteps: int = MAX_TIMESTEPS, save_freq: int = 10000, walls=None, env=None, seed=None, output_dir: str = None, dirt_num: int = 5):
    if output_dir is None:
        # Fallback if no output_dir is provided, though __main__ should provide it.
        output_dir = get_log_dir(grid_size)
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
    
    # Initialize reward normalizer
    reward_normalizer = RunningMeanStd(shape=())
    
    # Create environment if not provided
    if env is None:
        # Create and wrap environment with metrics and additional wrappers
        env = gym.make(env_id, grid_size=grid_size, use_counter=False, dirt_num=dirt_num)  # Add grid_size parameter
        env = TimeLimit(env, max_episode_steps=MAX_STEPS)
        env = ExplorationBonusWrapper(env, bonus=0.10525612628712341)  # From Optuna study
        env = ExploitationPenaltyWrapper(env, time_penalty=-0.0020989802390739463, stay_penalty=-0.05073560696895504)  # From Optuna study
        env = MetricWrapper(env)
    
    print("\nEnvironment Wrappers:")
    print(f"- TimeLimit: {MAX_STEPS} steps")
    print(f"- ExplorationBonus: +{0.10525612628712341}")  # From Optuna study
    print(f"- ExploitationPenalty: time={-0.0020989802390739463}, stay={-0.05073560696895504}")  # From Optuna study
    print("- MetricWrapper: enabled")
    print("- Internal stuck counter: disabled")
    
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
        'agent_pos': env.observation_space['agent_pos'].shape[0],
        'agent_orient': 1,
        'local_view': env.observation_space['local_view'].shape[0]
    }
    n_actions = env.action_space.n
    print(f"Observation dimensions: {obs_dim}")
    print(f"Number of actions: {n_actions}")
    print(f"Number of atoms: {N_ATOMS}")

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
    print("\nStarting training loop...")

    obs, _ = env.reset(options={"walls": walls})
    episode_reward = 0
    episode_count = 0
    episode_steps = 0
    total_loss = 0
    num_updates = 0
    beta = BETA

    best_reward = float('-inf')

    for step in range(total_timesteps):
        if step % 100 == 0:
            print_progress_bar(step, total_timesteps, prefix='Progress:', suffix='Complete', length=40)
            
        episode_steps += 1
        policy_net.reset_noise()
        
        action = policy_net.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Normalize reward
        reward_normalizer.update(np.array([reward]))
        normalized_reward = float(reward_normalizer.normalize(np.array([reward]))[0])
        normalized_reward *= REWARD_SCALING
        
        buffer.push(obs, action, normalized_reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward  # Track original reward for logging

        if done:
            episode_count += 1
            avg_loss = total_loss/max(1, num_updates)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Train/Episode_Reward', episode_reward, episode_count)
            writer.add_scalar('Train/Episode_Length', episode_steps, episode_count)
            writer.add_scalar('Train/Average_Loss', avg_loss, episode_count)
            
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
            print(f"Average Loss: {avg_loss:.4f}")
            if "coverage_ratio" in info:
                print("\nCleaning Metrics:")
                print(f"Coverage: {info['coverage_ratio']:.2%}")
                print(f"Path Efficiency: {info['path_efficiency']:.2f}")
                print(f"Revisit Ratio: {info['revisit_ratio']:.2f}")
            
            obs, _ = env.reset(options={"walls": walls})
            episode_reward = 0
            episode_steps = 0
            total_loss = 0
            num_updates = 0

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

                loss = -(target_q_dist * current_q_dist.log()).sum(1)
                
                # Convert to priorities with safety checks for NaN/inf values
                priorities = loss.detach().cpu().numpy()
                priorities = np.where(np.isfinite(priorities), priorities, 1e-8)
                priorities = np.maximum(priorities, 1e-8)  # Ensure all priorities are positive
                
                loss = (loss * weights).mean()
                
                # Log training loss to TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), step)
                
                total_loss += loss.item()
                num_updates += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()

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

def evaluate(policy_net, env_id="Vacuum-v0", grid_size=(6, 6), episodes=5, render=True, walls=None, env=None, output_dir: str = None, dirt_num: int = 5):
    if output_dir is None:
        # Fallback if no output_dir is provided, though __main__ should provide it.
        output_dir = get_log_dir(grid_size)
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
    
    # Create environment if not provided
    if env is None:
        # Create and wrap environment with same wrappers as training
        env = gym.make(env_id, grid_size=grid_size, render_mode="plot", use_counter=False, dirt_num=dirt_num)
        env = TimeLimit(env, max_episode_steps=MAX_STEPS)
        env = ExplorationBonusWrapper(env, bonus=0.10525612628712341)  # From Optuna study
        env = ExploitationPenaltyWrapper(env, time_penalty=-0.0020989802390739463, stay_penalty=-0.05073560696895504)  # From Optuna study
        env = MetricWrapper(env)
    
    print("\nEnvironment Wrappers:")
    print(f"- TimeLimit: {MAX_STEPS} steps")
    print(f"- ExplorationBonus: +{0.10525612628712341}")  # From Optuna study
    print(f"- ExploitationPenalty: time={-0.0020989802390739463}, stay={-0.05073560696895504}")  # From Optuna study
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
        "cleaning_time": []
    }

    # Evaluate the last few episodes in reverse order
    for ep in range(episodes - 1, -1, -1):
        print_section_header(f"Evaluation Episode {episodes - ep}")
        obs, _ = env.reset(options={"walls": walls})
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Use deterministic actions during evaluation (no noise, no epsilon)
            with torch.no_grad():  # Ensure no gradients during evaluation
                action = policy_net.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        # Log metrics
        metrics["episode_rewards"].append(total_reward)
        metrics["episode_lengths"].append(steps)
        if "coverage_ratio" in info:
            metrics["coverage_ratio"].append(info["coverage_ratio"])
            metrics["path_efficiency"].append(info["path_efficiency"])
            metrics["revisit_ratio"].append(info["revisit_ratio"])

        print("\nEpisode Results:")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print("\nCleaning Metrics:")
        print(f"Coverage: {info.get('coverage_ratio', 0):.2%}")
        print(f"Path Efficiency: {info.get('path_efficiency', 0):.2f}")
        print(f"Revisit Ratio: {info.get('revisit_ratio', 0):.2f}")
        
        if render:
            # ---- DEBUG PRINT ----
            if hasattr(env, 'unwrapped'):
                print(f"[evaluate DEBUG] id(env.unwrapped) before rollout_and_record: {id(env.unwrapped)}")
            else:
                print(f"[evaluate DEBUG] env has no unwrapped attribute before rollout_and_record")
            # ---- END DEBUG PRINT ----
            video_path = os.path.join(output_dir, f"eval_ep_{episodes - ep}.mp4")
            rollout_and_record(env.unwrapped, policy_net, filename=video_path, walls=walls)
            print(f"\nSaved video: {video_path}")

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

    # Save evaluation metrics
    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
    print(f"\nMetrics saved to: {metrics_file}")

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
                      help='Number of episodes to evaluate for')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[6, 6],
                      help='Grid size as two integers (e.g., 6 6 for 6x6 grid)')
    parser.add_argument('--wall_mode', choices=['random', 'hardcoded'], default='random',
                      help='Wall layout: "random" or "hardcoded" (only applies to 40x30 grid)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--dirt_num', type=int, default=5,
                      help='Number of dirt clusters to generate (0 means all non-obstacle tiles are dirt)')
    args = parser.parse_args()

    grid_size = tuple(args.grid_size)
    # LOG_DIR = get_log_dir(grid_size) # Removed: LOG_DIR will be determined per mode

    # Import hardcoded wall layout if needed
    walls = None
    if args.wall_mode == 'hardcoded' and grid_size == (40, 30):
        from run import generate_1b1b_layout_grid
        walls = generate_1b1b_layout_grid()

    if args.mode == 'train':
        # Determine log directory for this run
        run_log_dir = get_log_dir(grid_size)
        print(f"All outputs for this run will be saved to: {run_log_dir}")

        # Train a new model
        model = train(total_timesteps=args.timesteps, grid_size=grid_size, walls=walls, 
                     seed=args.seed, output_dir=run_log_dir, dirt_num=args.dirt_num)
        # Evaluate the trained model, saving results in the same directory
        evaluate(model, grid_size=grid_size, episodes=args.eval_episodes, walls=walls, output_dir=run_log_dir, dirt_num=args.dirt_num)
    else: # args.mode == 'eval'
        # args.model_path is expected to be a path to the model file.
        # The directory of this file will be used for saving evaluation outputs.
        model_load_path = os.path.abspath(args.model_path)
        eval_output_dir = os.path.dirname(model_load_path)
        os.makedirs(eval_output_dir, exist_ok=True) # Ensure the directory exists

        print(f"Loading model from: {model_load_path}")
        print(f"Evaluation outputs will be saved to: {eval_output_dir}")

        # Create a new model instance
        env = gym.make("Vacuum-v0", grid_size=grid_size)
        obs_dim = {
            'agent_pos': env.observation_space['agent_pos'].shape[0],
            'agent_orient': 1,
            'local_view': env.observation_space['local_view'].shape[0]
        }
        n_actions = env.action_space.n
        model = RainbowDQN(obs_dim, n_actions).to(device)
        env.close()

        # Load and evaluate the model
        loaded_model = load_model(model, model_load_path) # Pass full path
        if loaded_model is not None:
            # Ensure model is in evaluation mode
            loaded_model.eval()
            evaluate(loaded_model, grid_size=grid_size, episodes=args.eval_episodes, walls=walls, output_dir=eval_output_dir, dirt_num=args.dirt_num)
        else:
            print(f"Could not find model at {model_load_path}")

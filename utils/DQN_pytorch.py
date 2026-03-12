"""DQN utilities for policy network training"""
from collections import deque, namedtuple
import random
import torch
import math


Transition = namedtuple('Transition', ('state', 'action', 'reward'))


class ReplayMemory:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, state, action, reward):
        """Save a transition"""
        self.memory.append(Transition(state, action, reward))
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class EpsilonGreedy:
    """Epsilon-greedy exploration strategy"""
    
    def __init__(self, eps_start=0.9, eps_end=0.01, eps_decay=2500):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
    
    def get_epsilon(self):
        """Compute current epsilon (decays over time)"""
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-1.0 * self.steps_done / self.eps_decay)
        return eps
    
    def select_action(self, q_values, device):
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            q_values: Tensor [batch_size, num_patches]
            device: torch.device
            
        Returns:
            actions: Tensor [batch_size, num_patches] of 0s and 1s
        """
        sample = random.random()
        batch_size, num_patches = q_values.shape
        
        eps_threshold = self.get_epsilon()
        self.steps_done += 1
        
        if sample > eps_threshold:
            # EXPLOIT: Use Q-values to decide (Q > 0 → use HR)
            with torch.no_grad():
                return (q_values > 0).float()
        else:
            # EXPLORE: Random actions
            return torch.randint(0, 2, (batch_size, num_patches),
                               device=device, dtype=torch.float)
import numpy as np
import random
from collections import namedtuple, deque
import torch
from observation import Observation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trajectory:
    """Variable-size buffer to store trajectory steps."""

    def __init__(self, seed=0):
        """Initialize a Trajectory object.

        Params
        ======
            seed (int): random seed
        """
        self.memory = []
        self.trajectory = namedtuple("Trajectory", field_names=["observation",
                                                                "advantage",
                                                                "target_value"])
        self.seed = random.seed(seed)

    def add(self, observation, advantage, target_value):
        """Add a new trajectory step to memory."""
        t = self.trajectory(observation, advantage, target_value)
        self.memory.append(t)
    
    def sample(self, batch_size):
        """Randomly sample a batch of trajectory steps from memory."""
        trajectory = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([t.observation.state for t in trajectory if t is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.observation.action for t in trajectory if t is not None])).float().to(device)
        log_probs = torch.from_numpy(np.vstack([t.observation.log_prob for t in trajectory if t is not None])).float().to(device)
        advantages = torch.from_numpy(np.vstack([t.advantage for t in trajectory if t is not None])).float().to(device)
        target_values = torch.from_numpy(np.vstack([t.target_value for t in trajectory if t is not None])).float().to(device)

        return (states, actions, log_probs, advantages, target_values)

    def clear(self):
        self.memory = []

    def __len__(self):
        """Return the current size of internal memory buffer."""
        return len(self.memory)

    def __iter__(self):
        """Returns an iterator of internal memory buffer."""
        return iter(self.memory)
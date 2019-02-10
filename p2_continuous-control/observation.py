import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Observation(object):
    """Variable-size buffer to store observations."""

    def __init__(self, seed=0):
        """Initialize a Observation object.

        Params
        ======
            seed (int): random seed
        """
        self.memory = []
        self.observation = namedtuple("Observation", field_names=["state",
                                                                  "action",
                                                                  "reward",
                                                                  "done",
                                                                  "log_prob",
                                                                  "value"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, done, log_prob, value):
        """Add a new obsevation to buffer."""
        o = self.observation(state, action, reward, done, log_prob, value)
        self.memory.append(o)
    
    def sample(self, batch_size):
        """Randomly sample a batch of observations from memory."""
        observations = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([o.state for o in observations if o is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([o.action for o in observations if o is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([o.reward for o in observations if o is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([o.done for o in observations if o is not None]).astype(np.uint8)).float().to(device)
        log_probs = torch.from_numpy(np.vstack([o.log_prob for o in observations if o is not None])).float().to(device)
        values = torch.from_numpy(np.vstack([o.value for o in observations if o is not None])).float().to(device)

        return (states, actions, rewards, dones, log_probs, values)

    def clear(self):
        self.memory = []

    def __len__(self):
        """Return the current size of internal memory buffer."""
        return len(self.memory)

    def __iter__(self):
        """Returns an iterator of internal memory buffer."""
        return iter(self.memory)

    def __getitem__(self,key):
        return self.memory[key]
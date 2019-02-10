import numpy as np
import random

from observation import Observation
from trajectory import Trajectory

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TRAJECTORY_LEN = int(1000)   # trajectory size
BATCH_SIZE = 128             # minibatch size
DISCOUNT = 0.99              # discount (gamma) factor
TAU = 0.95                   # tau (lambda) for GEA
LR = 1e-4                    # learning rate
EPOCH = 10                   # number of optimization epochs
EPSILON = 0.1                # clipped surrogate bounding parameter
BETA = 0.01                  # beta
MAX_GRAD = 10.0              # clip gradient

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO():
    """Interacts with and learns from the environment."""

    def __init__(self, policy, state_size, action_size, 
                 trajectory_len=TRAJECTORY_LEN, num_of_epoch=EPOCH, 
                 batch_size=BATCH_SIZE, discount=DISCOUNT, tau=TAU,
                 epsilon=EPSILON, beta=BETA, learning_rate=LR, 
                 max_gradient=MAX_GRAD, seed=0):
        """Initialize a PPO agent.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # PPO parameters
        self.epsilon = epsilon
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.num_of_epoch = num_of_epoch
        self.max_gradient = max_gradient
        self.trajectory_len = trajectory_len
        
        self.num_agent = 20

        # Policy
        self.policy = policy.to(device)

        # Optimizer
        self.optimizer = optim.SGD(self.policy.parameters(), lr=learning_rate)
        #self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Observation storages
        self.observation = Observation(seed)

        self.t_step = int(0)

        self.sigma = nn.Parameter(torch.zeros(action_size))

    def step(self, state, action, reward, done, log_prob, value):
        
        # Save experience in trajectory
        self.observation.add(state, action, reward, done, log_prob, value)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.trajectory_len
        if self.t_step == 0:
            # If enough samples are available in observation, get random subset and learn
            if len(self.observation) > self.batch_size:
                trajectory = self._make_trajectory()
                self._learn(trajectory)
                # clear observation
                self.observation.clear()
                # the clipping parameter reduces as time goes on
                self.epsilon *= 0.999
                # this reduces exploration as time goes on 
                self.beta *= 0.998

    def act(self, state):
        """Returns action, value, log_prob for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            action (array_like): given action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy.eval()
        with torch.no_grad():
            action_est, value = self.policy(state)
            value = value.detach()
            dist = torch.distributions.Normal(action_est, F.softplus(self.sigma).to(device))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob = torch.sum(log_prob, dim=-1).detach()
            action = action.detach()
        self.policy.train()
        return action, value, log_prob

    def _clipped_surrogate(self, states, actions, old_probs, geas, target_values):
        action_est, values = self.policy(states)
        dist = torch.distributions.Normal(action_est, F.softplus(self.sigma).to(device))
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)

        ratio = torch.exp(log_probs - old_probs)
        ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        # clipped surrogate
        L_CLIP = torch.mean(torch.min(ratio*geas, ratio_clipped*geas))
        # entropy bonus
        S = entropy.mean()
        # squared-error value function loss
        # alternatively, L_VF = F.smooth_l1_loss(values, target_values) can be used
        L_VF = 0.5 * (target_values - values).pow(2).mean()
        # total loss
        L = -(L_CLIP - L_VF + beta*S)
        return L

    def _make_trajectory(self):
        def to_tensor(x, dtype=np.float32):
            return torch.from_numpy(np.array(x).astype(dtype)).to(device)
        def concat(v):
            if len(v.shape) == 3:
                return v.reshape([-1, v.shape[-1]])
            return v.reshape([-1])

        obs = self.observation
        obs_size = len(obs)

        ## Create empty buffer
        #GAE = torch.zeros(obs_size,num_agent).float().to(device)
        #returns = torch.zeros(obs_size,self.num_agent).float().to(device)
        
        # Trajectory storage
        trajectory = Trajectory(self.seed)
        
        # Set start values
        GAE_current = torch.zeros(self.num_agent).float().to(device)

        #TAU = 0.95
        #discount = 0.99
        values_next = obs[-1].value.detach()
        returns_current = obs[-1].value.detach()
        for i in reversed(range(obs_size)):
            values_current = to_tensor(obs[i].value)
            rewards_current = to_tensor(obs[i].reward)
            gamma = self.discount * (1. - to_tensor(obs[i].done, dtype=np.uint8).float())
            # Calculate TD Error
            td_error = rewards_current + gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * self.tau * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE and target value in memory
            trajectory.add(concat(obs[i].state),
                           concat(obs[i].action),
                           concat(obs[i].log_prob),
                           concat(GAE_current),
                           concat(returns_current))
            values_next = values_current
        return trajectory

    def _learn(self, trajectory):
        """Update value parameters using given batch of experience tuples.
        """
        # gradient ascent step
        num_sample = len(trajectory)//self.batch_size
        for epoch in range(self.num_of_epoch):
            for b in range(num_sample):
                # minibatch (Tuple[torch.Variable]): (s, a, p, adv, tv) 
                minibatch = trajectory.sample(self.batch_size)
                s, a, p, adv, tv= minibatch
                # normalize advantage
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                L = self._clipped_surrogate(states=s,
                                            actions=a,
                                            old_probs=p,
                                            geas=adv,
                                            target_values=tv)
                self.optimizer.zero_grad()
                # we need to specify retain_graph=True on the backward pass
                # this is because pytorch automatically frees the computational graph after
                # the backward pass to save memory
                # Without the computational graph, the chain of derivative is lost
                L.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.optimizer.step()
                del(L)


import numpy as np
import random

from observation import Observation
from trajectory import Trajectory

import torch
import torch.nn.functional as F
import torch.optim as optim

TRAJECTORY_SIZE = int(1000)  # trajectory size
BATCH_SIZE = 128             # minibatch size
GAMMA = 0.99                 # discount factor
TAU = 0.95                   # tau (lambda) for GEA
LR = 1e-4                    # learning rate
EPOCH = 10                   # number of optimization epochs
EPSILON = 0.1                # clipped surrogate bounding parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO():
    """Interacts with and learns from the environment."""

    def __init__(self, policy, tmax=TRAJECTORY_SIZE, 
                 num_of_epoch=EPOCH, batch_size=BATCH_SIZE,
                 gamma=GAMMA, tau=TAU, epsilon=EPSILON, seed=0):
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

        # Policy
        self.policy = policy.to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        # Observation storages
        self.observation = Observation(seed)

        self.t_step = int(0)

    def step(self, state, action, reward, next_state, done):
        # Save experience in trajectory
        self.observation.add(state, action, reward, done, log_prob, value)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % TRAJECTORY_SIZE
        if self.t_step == 0:
            # If enough samples are available in observation, get random subset and learn
            if len(self.observation) > BATCH_SIZE:
                trajectory = self._make_trajectory()
                self._learn(trajectory)
                # clear observation
                self.observation.clear()
                # the clipping parameter reduces as time goes on
                epsilon*=.999
                # this reduces exploration as time goes on 
                beta*=.998

    def act(self, state, training_mode=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy.eval()
        with torch.no_grad():
            if training_mode:
                action_est, value = self.policy(state)
                value = values.detach()
            else:
                action_est, _ = self.policy(state)
            sigma = nn.Parameter(torch.zeros(g_action_size))
            dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(g_device))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob = torch.sum(log_prob, dim=-1).detach()
            action = action.detach()
        self.policy.train()
        return action, value, log_prob, 

    def _clipped_surrogate(self, states, actions, old_probs, target_values, geas):
        action_est, values = self.policy(states)
        sigma = nn.Parameter(torch.zeros(g_action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(g_device))
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        ############################################
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
        obs = self.observation
        obs_size = len(obs)

        ## Create empty buffer
        #GAE = torch.zeros(obs_size,num_agent).float().to(g_device)
        #returns = torch.zeros(obs_size,self.num_agent).float().to(g_device)
        
        # Trajectory storage
        trajectory = Trajectory(self.seed)
        
        # Set start values
        GAE_current = torch.zeros(self.num_agent).float().to(g_device)

        TAU = 0.95
        discount = 0.99
        values_next = obs[-1].value.detach()
        returns_current = obs[-1].value.detach()
        for i in reversed(range(obs_size)):
            values_current = obs[i].value[i]
            rewards_current = obs[i].reward[i]
            gamma = discount * (1. - obs[i].done.float())
            # Calculate TD Error
            td_error = rewards_current + gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * TAU * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE and target value in memory
            trajectory.add(obs[i], GAE_current, returns_current)
            values_next = values_current
        return trajectory

    def _learn(self, trajectory):
        """Update value parameters using given batch of experience tuples.
        """
        # gradient ascent step
        num_sample = len(trajectory)//BATCH_SIZE
        for epoch in range(EPOCH):
            for b in range(num_sample):
                # minibatch (Tuple[torch.Variable]): (s, a, p, target_v, geas) 
                minibatch = trajectory.sample(BATCH_SIZE)
                states, actions, old_probs, target_values, geas= minibatch
                L = self._clipped_surrogate(self, states, actions, old_probs, target_values, geas)
                self.optimizer.zero_grad()
                # we need to specify retain_graph=True on the backward pass
                # this is because pytorch automatically frees the computational graph after
                # the backward pass to save memory
                # Without the computational graph, the chain of derivative is lost
                L.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
                self.optimizer.step()
                del(L)


import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import numpy as np
from unityagents import UnityEnvironment

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from actor_critic import ActorCritic

g_env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

# get the default brain
g_brain_name = g_env.brain_names[0]
g_brain = g_env.brains[g_brain_name]
g_env_info = g_env.reset(train_mode=False)[g_brain_name]
g_num_agents = len(g_env_info.agents)
g_action_size = g_brain.vector_action_space_size
g_state_size = g_env_info.vector_observations.shape[1]

# reset the environment
g_env_info = g_env.reset(train_mode=True)[g_brain_name]

# number of agents
g_num_agents = len(g_env_info.agents)
print('Number of agents:', g_num_agents)

# size of each action
g_action_size = g_brain.vector_action_space_size
print('Size of each action:', g_action_size)

# examine the state space 
states = g_env_info.vector_observations
g_state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], g_state_size))
print('The state for the first agent looks like:', states[0])

g_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_policy(envs, policy, tmax=1000):
    reward_list=[]
    env_info = envs.reset(train_mode=False)[g_brain_name]
    for t in range(tmax):
        states = torch.from_numpy(env_info.vector_observations).float().to(g_device)
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(g_action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(g_device))
        actions = dist.sample()
        env_actions = actions.cpu().numpy()
        env_info = envs.step(env_actions)[g_brain_name]
        reward = env_info.rewards
        dones = env_info.local_done
        reward_list.append(np.mean(reward))

        # stop if any of the trajectories is done to have retangular lists
        if np.any(dones):
            break
    return reward_list

if __name__ == "__main__":
    episode = 10
    scores_window = deque(maxlen=100)  # last 100 scores

    # policy and saved model
    """
    policy=ActorCritic(state_size=g_state_size,
              action_size=g_action_size,
              shared_layers=[128, 64],
              critic_hidden_layers=[],
              actor_hidden_layers=[],
              init_type='xavier-uniform',
              seed=0).to(g_device)
    saved_model = 'ppo_128x64_a0_c0_470e.pth'
    """
    policy=ActorCritic(state_size=g_state_size,
              action_size=g_action_size,
              shared_layers=[128, 128],
              critic_hidden_layers=[64],
              actor_hidden_layers=[64],
              init_type='xavier-uniform',
              seed=0).to(g_device)
    saved_model = 'ppo_128x128_a64_c64_193e.pth'
    

    # load the model
    policy.load_state_dict(torch.load(saved_model))

    # evaluate the model
    for e in range(episode):
        rewards = eval_policy(envs=g_env, policy=policy, tmax=1000)
        total_rewards = np.sum(rewards,0)
        scores_window.append(total_rewards.mean())
        print("Episode: {0:d}, score: {1}".format(e+1, np.mean(scores_window)), end="\n")

    g_env.close()

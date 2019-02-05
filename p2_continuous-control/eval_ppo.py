from unityagents import UnityEnvironment
import numpy as np

g_env = UnityEnvironment(file_name='./Reacher_Linux_Multi/Reacher.x86_64')

g_brain_name = g_env.brain_names[0]
g_brain = g_env.brains[g_brain_name]
g_env_info = g_env.reset(train_mode=False)[g_brain_name]
g_num_agents = len(g_env_info.agents)
g_action_size = g_brain.vector_action_space_size
g_state_size = g_env_info.vector_observations.shape[1]
print("g_action_size:{}".format(g_action_size))
print("g_state_size:{}".format(g_state_size))

import torch
import torch.nn as nn
import torch.nn.functional as F
from policy import Policy
from collections import deque

g_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_policy(envs, policy, tmax=200):
    reward_list=[]
    env_info = envs.reset(train_mode=False)[g_brain_name]
    for t in range(tmax):
        state = torch.from_numpy(env_info.vector_observations).float().to(g_device)
        action, _, _, value = policy(state=state)
        action = action.cpu().detach().numpy()
        action = np.clip(action,-1.0,1.0)
        env_info = envs.step(action)[g_brain_name]
        reward = env_info.rewards
        dones = env_info.local_done
        reward_list.append(reward)

        # stop if any of the trajectories is done to have retangular lists
        if np.any(dones):
            break
    return reward_list

scores_window = deque(maxlen=100)  # last 100 scores
episode = 100
# run your own policy!
policy=Policy(state_size=g_state_size,
              action_size=g_action_size,
              hidden_layers=[512, 256],
              seed=0).to(g_device)

# load the weights from file
policy.load_state_dict(torch.load('bestcheckpoint.pth'))

for e in range(episode):
    rewards = eval_policy(envs=g_env, policy=policy, tmax=128)
    total_rewards = np.sum(rewards,0)
    scores_window.append(total_rewards.mean())
    print("Episode: {0:d}, score: {1}".format(e+1, np.mean(scores_window)), end="\r")
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
from ppo import PPO

g_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scores_window = deque(maxlen=100)  # last 100 scores

"""
discount = 0.99
epsilon = 0.1
beta = .01
opt_epoch = 10
batch_size = 128
"""
tmax = 1000 #env episode steps
episode = 2000

print_per_n = min(10,episode/10)
counter = 0
start_time = timeit.default_timer()

def train(env, agent):
    env_info = env.reset(train_mode=True)[g_brain_name]
    rewards_list = []
    for e in range(episode):
        for t in range(tmax):
            states = env_info.vector_observations
            actions, values, log_probs = agent.act(states)
            env_actions = actions.squeeze().detach().cpu().numpy()
            env_info = env.step(env_actions)[g_brain_name]
            rewards = env_info.rewards
            rewards_list.append(np.mean(rewards))
            dones = env_info.local_done
            agent.step(state=states,
                     action=actions,
                     reward=rewards,
                     done=dones,
                     log_prob=log_probs,
                     value=values)
            if np.any(dones):
                avg_score_per_episode = np.sum(rewards_list)
                scores_window.append(avg_score_per_episode)
                env_info = env.reset(train_mode=True)[g_brain_name]
                rewards_list = []

        # display some progress every 25 iterations
        if (e+1)%print_per_n ==0 :
            print("Episode: {0:d}, average score: {1:.2f}, beta: {2:.4f}".format(e+1,np.mean(scores_window), agent.beta), end="\n")
        else:
            print("Episode: {0:d}, score: {1:.2f}".format(e+1, avg_score_per_episode), end="\r")
        if np.mean(scores_window)<5.0:
            counter = 0# stop if any of the trajectories is done to have retangular lists
        if e>=25 and np.mean(scores_window)>30.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e+1, np.mean(scores_window)))
            break

        # update progress widget bar
        #timer.update(e+1)

    #timer.finish()

    print('Average Score: {:.2f}'.format(np.mean(scores_window)))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    print("Saving checkpoint!")
    # save your policy!
    torch.save(agent.policy.state_dict(), 'checkpoint.pth')

if __name__ == "__main__":
    g_env = UnityEnvironment(file_name='./Reacher_Linux_Multi/Reacher.x86_64')

    g_brain_name = g_env.brain_names[0]
    g_brain = g_env.brains[g_brain_name]
    g_env_info = g_env.reset(train_mode=True)[g_brain_name]
    g_num_agents = len(g_env_info.agents)
    g_action_size = g_brain.vector_action_space_size
    g_state_size = g_env_info.vector_observations.shape[1]
    
    policy=ActorCritic(state_size=g_state_size,
              action_size=g_action_size,
              shared_layers=[128, 128],
              critic_hidden_layers=[64],
              actor_hidden_layers=[64],
              init_type='xavier-uniform',
              seed=0)
    agent = PPO(policy=policy,
            state_size=g_state_size,
            action_size=g_action_size)
    train(env=g_env,agent=agent)
    g_env.close()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from actor_critic import ActorCritic

g_env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')
#g_env = UnityEnvironment(file_name='/home/peyman/Projects/tmp/expt/bin/Reacher_Linux/Reacher.x86_64')


# In[2]:


g_brain_name = g_env.brain_names[0]
g_brain = g_env.brains[g_brain_name]
g_env_info = g_env.reset(train_mode=True)[g_brain_name]
g_num_agents = len(g_env_info.agents)
g_action_size = g_brain.vector_action_space_size
g_state_size = g_env_info.vector_observations.shape[1]


# In[3]:

g_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_trajectories(envs, policy, tmax=200, nrand=5, train_mode=False):

    def to_tensor(x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(g_device)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]
    value_list=[]
    done_list=[]

    env_info = envs.reset(train_mode=train_mode)[g_brain_name]
    #env_info = envs.reset(train_mode=train_mode,config={'goal_speed':0.0,'goal_size':10.0})[g_brain_name]

    # perform nrand random steps
    for _ in range(nrand):
        action = np.random.randn(g_num_agents, g_action_size)
        action = np.clip(action, -1.0, 1.0)
        env_info = envs.step(action)[g_brain_name]

    for t in range(tmax):
        ##########################################################################
        states = to_tensor(env_info.vector_observations)
        #pred = policy(states)
        #pred = [v.detach() for v in pred]
        #actions, log_probs, _, values = pred
        ############################################
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(g_action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(g_device))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()
        values = values.detach()
        actions = actions.detach()
        ############################################
        actions_np = actions.cpu().numpy()
        env_info = envs.step(actions_np)[g_brain_name]
        rewards = to_tensor(env_info.rewards)
        dones = to_tensor(env_info.local_done, dtype=np.uint8)

        state_list.append(states.unsqueeze(0))
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones.unsqueeze(0))
        if np.any(dones):
            env_info = envs.reset(train_mode=train_mode)[g_brain_name]

    state_list = torch.cat(state_list, dim=0)
    prob_list = torch.cat(prob_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    return prob_list, state_list, action_list, reward_list, value_list, done_list


# In[5]:

def calc_returns(rewards, values, dones):
    n_step = len(rewards)
    n_agent = len(rewards[0])

    # Create empty buffer
    GAE = torch.zeros(n_step,n_agent).float().to(g_device)
    returns = torch.zeros(n_step,n_agent).float().to(g_device)

    # Set start values
    GAE_current = torch.zeros(n_agent).float().to(g_device)

    TAU = 0.95
    discount = 0.99
    values_next = values[-1].detach()
    returns_current = values[-1].detach()
    for irow in reversed(range(n_step)):
        values_current = values[irow]
        rewards_current = rewards[irow]
        gamma = discount * (1. - dones[irow].float())

        # Calculate TD Error
        td_error = rewards_current + gamma * values_next - values_current
        # Update GAE, returns
        GAE_current = td_error + gamma * TAU * GAE_current
        returns_current = rewards_current + gamma * returns_current
        # Set GAE, returns to buffer
        GAE[irow] = GAE_current
        returns[irow] = returns_current

        values_next = values_current

    return GAE, returns

# run your own policy!
#[512, 256]
policy=ActorCritic(state_size=g_state_size,
              action_size=g_action_size,
              shared_layers=[128, 128],
              critic_hidden_layers=[64],
              actor_hidden_layers=[64],
              init_type='xavier-uniform',
              seed=0).to(g_device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
optimizer = optim.Adam(policy.parameters(), lr=2e-4)


# In[7]:

scores_window = deque(maxlen=100)  # last 100 scores
save_scores = []

discount = 0.99
epsilon = 0.1
beta = .01
opt_epoch = 10
episode = 2000
batch_size = 128
tmax = 1000 #env episode steps

print_per_n = min(10,episode/10)
counter = 0
start_time = timeit.default_timer()

for e in range(episode):
    policy.eval()
    old_probs_lst, states_lst, actions_lst, rewards_lst, values_lst, dones_list = collect_trajectories(envs=g_env,
                                                                                                       policy=policy,
                                                                                                       tmax=tmax,
                                                                                                       nrand = 0,
                                                                                                       train_mode=True)

    avg_score = rewards_lst.sum(dim=0).mean().item()
    scores_window.append(avg_score)
    save_scores.append(avg_score)

    gea, target_value = calc_returns(rewards = rewards_lst,
                                     values = values_lst,
                                     dones=dones_list)
    gea = (gea - gea.mean()) / (gea.std() + 1e-8)

    policy.train()

    # cat all agents
    def concat_all(v):
        if len(v.shape) == 3:
            return v.reshape([-1, v.shape[-1]])
        return v.reshape([-1])

    old_probs_lst = concat_all(old_probs_lst)
    states_lst = concat_all(states_lst)
    actions_lst = concat_all(actions_lst)
    rewards_lst = concat_all(rewards_lst)
    values_lst = concat_all(values_lst)
    gea = concat_all(gea)
    target_value = concat_all(target_value)

    # gradient ascent step
    n_sample = len(old_probs_lst)//batch_size
    idx = np.arange(len(old_probs_lst))
    np.random.shuffle(idx)
    for epoch in range(opt_epoch):
        for b in range(n_sample):
            ind = idx[b*batch_size:(b+1)*batch_size]
            g = gea[ind]
            tv = target_value[ind]
            actions = actions_lst[ind]
            old_probs = old_probs_lst[ind]
            ############################################
            action_est, values = policy(states_lst[ind])
            sigma = nn.Parameter(torch.zeros(g_action_size))
            dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(g_device))
            log_probs = dist.log_prob(actions)
            log_probs = torch.sum(log_probs, dim=-1)
            entropy = torch.sum(dist.entropy(), dim=-1)
            ############################################
            ratio = torch.exp(log_probs - old_probs)
            ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            L_CLIP = torch.mean(torch.min(ratio*g, ratio_clipped*g))
            # entropy bonus
            S = entropy.mean()
            # squared-error value function loss
            L_VF = 0.5 * (tv - values).pow(2).mean()
            # clipped surrogate
            L = -(L_CLIP - L_VF + beta*S)
            optimizer.zero_grad()
            # we need to specify retain_graph=True on the backward pass
            # this is because pytorch automatically frees the computational graph after
            # the backward pass to save memory
            # Without the computational graph, the chain of derivative is lost
            L.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            del(L)

    # the clipping parameter reduces as time goes on
    epsilon*=.999
    
    # the regulation term also reduces
    # this reduces exploration in later runs
    beta*=.998
    
    # display some progress every 25 iterations
    if (e+1)%print_per_n ==0 :
        print("Episode: {0:d}, average score: {1:.2f}, beta: {2:.4f}".format(e+1,np.mean(scores_window), beta), end="\n")
    else:
        print("Episode: {0:d}, score: {1:.2f}".format(e+1, avg_score), end="\r")
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
torch.save(policy.state_dict(), 'checkpoint.pth')

# save data
import pickle
pickle.dump(save_scores, open( "saved_scors.p", "wb" ) )

# plot scores
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(save_scores)), save_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores_plot.png')
plt.show()


# In[ ]:


g_env.close()


# In[ ]:





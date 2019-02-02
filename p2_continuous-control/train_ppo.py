#!/usr/bin/env python
# coding: utf-8

# In[1]:


from unityagents import UnityEnvironment
import numpy as np

g_env = UnityEnvironment(file_name='./Reacher_Linux_Multi/Reacher.x86_64')


# In[2]:


g_brain_name = g_env.brain_names[0]
g_brain = g_env.brains[g_brain_name]
g_env_info = g_env.reset(train_mode=True)[g_brain_name]
g_num_agents = len(g_env_info.agents)
g_action_size = g_brain.vector_action_space_size
g_state_size = g_env_info.vector_observations.shape[1]
print("g_action_size:{}".format(g_action_size))
print("g_state_size:{}".format(g_state_size))


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F

g_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

g_dist_std = torch.Tensor(nn.Parameter(torch.zeros(1, g_action_size)))

def states_to_prob(policy, states):
    states = torch.stack(states)
    #policy_input = states.view(-1,*states.shape[-3:])
    #return policy(policy_input).view(states.shape[:-3])
    return policy(states)

def clipped_surrogate(policy, old_probs, 
                      states, actions, 
                      rewards, values, 
                      gea, target_value,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):

    advantages_mean = np.mean(gea)
    advantages_std = np.std(gea) + 1.0e-10
    advantages_normalized = (gea - advantages_mean)/advantages_std
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.float, device=g_device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=g_device)
    advantages = torch.tensor(advantages_normalized, dtype=torch.float, device=g_device)
    values = torch.tensor(np.array(values), dtype=torch.float, device=g_device)
    target_value = torch.tensor(target_value, dtype=torch.float, device=g_device, requires_grad=False)
    
    states = torch.stack(states)
    actions, new_probs, entropy_loss, est_values = policy(state=states,
                                                          action=actions)
    gea_list = []
    target_value_list = []
    for n in range(g_num_agents):
        gea = [0.0] * len(state_list)
        target_value = [0.0] * len(state_list)
        i_max = len(state_list)
        done = 0
        TAU = 0.95
        discount = 0.99
        returns_ = 0.0
        advantages_ = 0.0
        done = 1.0
        for i in reversed(range(i_max)):
            rwrds_ = reward_list[i][n]
            values_ = value_list[i][n]
            next_value_ = value_list[min(i_max-1, i + 1)][n]
            td_error = rwrds_ + (discount * next_value_*(1.0-done)) - values_
            advantages_ = (advantages_ * TAU * discount*(1.0-done)) + td_error
            gea[i] = advantages_
            returns_ = (discount*returns_*(1.0-done)) + rwrds_
            target_value[i] = returns_
            done = 0.0

        gea = np.cumsum(gea)
        gea_list.append(deepcopy(gea))
        target_value_list.append(deepcopy(target_value))

    gea_list= list(map(list, zip(*gea_list)))
    target_value_list= list(map(list, zip(*target_value_list)))
    """
    dists = torch.distributions.Normal(est_actions, F.softplus(g_dist_std.to(g_device)))
    #dists = torch.distributions.Normal(est_actions, g_dist_std.to(g_device))
    #actions = dists.sample()
    #actions.clamp_(min=-1.0, max=1.0)
        
    log_prob = dists.log_prob(actions)
    log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
    
    new_probs = log_prob
        
    # entropy_loss = torch.Tensor(np.zeros((log_prob.size(0), 1)))
    entropy_loss = dists.entropy()
    entropy_loss = torch.sum(entropy_loss, dim=-1, keepdim=True)/4.0
    """
    # ratio for clipping
    ratio = torch.exp(new_probs-old_probs).squeeze(-1)

    # clipped function
    clip = torch.clamp(ratio, 1.0-epsilon, 1.0+epsilon)
    clipped_sur = torch.min(ratio*advantages,clip*advantages)
    
    #critic_loss = F.smooth_l1_loss(est_values.squeeze(),target_value.squeeze())
    critic_loss = 0.5 * (est_values.squeeze() - target_value.squeeze()).pow(2).mean()
    return torch.mean(clipped_sur + beta*entropy_loss), critic_loss, entropy_loss, clipped_sur

# In[4]:


from copy import deepcopy

def collect_trajectories(envs, policy, tmax=200, nrand=5, train_mode=False):

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]
    value_list=[]

    env_info = envs.reset(train_mode=train_mode)[g_brain_name]
    #env_info = envs.reset(train_mode=train_mode,config={'goal_speed':0.0,'goal_size':10.0})[g_brain_name]

    # perform nrand random steps
    for _ in range(nrand):
        action = np.random.randn(g_num_agents, g_action_size)
        action = np.clip(action, -1.0, 1.0)
        env_info = envs.step(action)[g_brain_name]

    for t in range(tmax):
        #state = torch.from_numpy(env_info.vector_observations).float().unsqueeze(0).to(g_device)
        state = torch.from_numpy(env_info.vector_observations).float().to(g_device)
        action, log_prob, _, value = policy(state=state)
        """
        est_action = est_action.squeeze().cpu().detach()
        dist = torch.distributions.Normal(est_action, F.softplus(g_dist_std))
        action = dist.sample().numpy()
        action = np.clip(action,-1.0,1.0)
        value = value.squeeze().cpu().detach().numpy()

        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True).cpu().detach().numpy()
        """
        action = action.cpu().detach().numpy()
        action = np.clip(action,-1.0,1.0)
        log_prob = log_prob.cpu().detach().numpy()
        value = value.cpu().detach().numpy()
        env_info = envs.step(action)[g_brain_name]

        reward = env_info.rewards
        dones = env_info.local_done

        state_list.append(state)
        reward_list.append(reward)
        prob_list.append(log_prob)
        action_list.append(action)
        value_list.append(value)

        # stop if any of the trajectories is done to have retangular lists
        if np.any(dones):
            break
    # return states, actions, rewards
    return prob_list, state_list, action_list, reward_list, value_list, gea_list, target_value_list


# In[5]:


import torch.optim as optim
from policy import Policy

# run your own policy!
policy=Policy(state_size=g_state_size,
              action_size=g_action_size,
              hidden_layers=[64, 64],
              seed=0).to(g_device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
optimizer = optim.Adam(policy.parameters(), lr=1e-4)


# In[7]:


import sys
from collections import deque
import timeit
from datetime import timedelta

scores_window = deque(maxlen=100)  # last 100 scores

discount = 0.99
epsilon = 0.1
beta = .01 #0.01
SGD_epoch = 4
episode = 150
batch_size = 128
#tmax = max(10*batch_size,int(30.0/0.1),1024)
tmax = batch_size

print_per_n = min(10,episode/10)
counter = 0
start_time = timeit.default_timer()

for e in range(episode):
    old_probs, states, actions, rewards, values, gea, target_value = collect_trajectories(envs=g_env, 
                                                                                          policy=policy, 
                                                                                          tmax=tmax,
                                                                                          nrand = 0,
                                                                                          train_mode=True)

    total_rewards = np.sum(rewards)
    scores_window.append(total_rewards)

    # gradient ascent step
    n_sample = len(old_probs)//batch_size
    idx = np.arange(len(old_probs))
    np.random.shuffle(idx)
    for b in range(n_sample):
        #ind = np.random.randint(len(old_probs),size=batch_size)
        #ind = np.random.randint(len(old_probs)-batch_size-1,size=1)
        #ind = range(b*batch_size,(b+1)*batch_size)
        ind = idx[b*batch_size:(b+1)*batch_size]
        #ind = ind[0]
        #ind = 0
        op = [old_probs[i] for i in ind]
        s = [states[i] for i in ind]
        a = [actions[i] for i in ind]
        r = [rewards[i] for i in ind]
        v = [values[i] for i in ind]
        g = [gea[i] for i in ind]
        tv = [target_value[i] for i in ind]
        for epoch in range(SGD_epoch):
            l_clip, critic_loss, entropy_loss, clipped_sur = clipped_surrogate(policy=policy,
                                                                               old_probs=op,
                                                                               states=s,
                                                                               actions=a,
                                                                               rewards=r,
                                                                               values=v,
                                                                               gea = g,
                                                                               target_value = tv,
                                                                               discount = discount,
                                                                               epsilon=epsilon,
                                                                               beta=beta)
            L = -l_clip+critic_loss
            #print("-l_clip:{}".format(-l_clip), end="\n")
            #print("critic_loss:{}".format(critic_loss), end="\n")
            #print("clipped_sur:{}".format(torch.mean(clipped_sur)), end="\n")
            #print("L:{}".format(L))
            optimizer.zero_grad()
            # we need to specify retain_graph=True on the backward pass
            # this is because pytorch automatically frees the computational graph after
            # the backward pass to save memory
            # Without the computational graph, the chain of derivative is lost
            L.backward(retain_graph=True)
            #torch.nn.utils.clip_grad_norm_(policy.parameters(), 100.0)
            optimizer.step()
            del L

    # the clipping parameter reduces as time goes on
    #epsilon*=.999
    
    # the regulation term also reduces
    # this reduces exploration in later runs
    #beta*=.995
    
    # display some progress every 25 iterations
    if (e+1)%print_per_n ==0 :
        print("Episode: {0:d}, average score: {1:f}, beta: {2:f}".format(e+1,np.mean(scores_window), beta), end="\n")
    else:
        print("Episode: {0:d}, score: {1}".format(e+1, total_rewards), end="\r")
    if np.mean(scores_window)<5.0:
        counter = 0
    if e>=100 and np.mean(scores_window)>30.0:
        counter += 1
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e+1, np.mean(scores_window)))
    if counter > 100:
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


# In[ ]:


g_env.close()


# In[ ]:





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
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.float, device=g_device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=g_device)
    values = torch.tensor(np.array(values), dtype=torch.float, device=g_device)
    advantages = gea

    states = torch.stack(states)
    actions = torch.stack(tuple(actions))
    actions, new_probs, entropy_loss, est_values = policy(state=states,
                                                          action=actions)
    # ratio for clipping
    ratio = torch.exp(new_probs-old_probs).squeeze(-1)

    # clipped function
    clip = torch.clamp(ratio, 1.0-epsilon, 1.0+epsilon)
    clipped_sur = torch.min(ratio*advantages,clip*advantages)
    
    #critic_loss = F.smooth_l1_loss(est_values.squeeze(),target_value.squeeze())
    critic_loss = 0.5 * (est_values.squeeze() - target_value.squeeze()).pow(2).mean()
    return torch.mean(clipped_sur + beta*entropy_loss), critic_loss, entropy_loss, clipped_sur, states, actions, old_probs, advantages

# In[4]:


from copy import deepcopy

def collect_trajectories(envs, policy, tmax=200, nrand=5, train_mode=False):

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

        #reward = torch.tensor(env_info.rewards, dtype=torch.float, device=g_device)
        reward = env_info.rewards
        dones = env_info.local_done

        state_list.append(state)
        prob_list.append(log_prob)
        action_list.append(action)
        reward_list.append(reward)
        value_list.append(value)
        done_list.append(dones)
        if np.any(dones):
            env_info = envs.reset(train_mode=train_mode)[g_brain_name]

    def calc_returns(rewards, values, dones):
        n_step = len(rewards)
        n_agent = len(rewards[0])

        rewards = torch.from_numpy(np.array(rewards).astype(np.float32)).to(g_device)
        values = torch.from_numpy(np.array(values).astype(np.float32)).to(g_device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).to(g_device)

        # Create empty buffer
        GAE = torch.zeros(n_step,n_agent).float().to(g_device)
        returns = torch.zeros(n_step,n_agent).float().to(g_device)
        

        # Set start values
        GAE_current = torch.zeros(n_agent).float().to(g_device)

        TAU = 0.95
        discount = 0.99
        values_next = values[-1]
        returns_current = values_next
        for irow in reversed(range(n_step)):
            values_current = values[irow]
            rewards_current = rewards[irow]
            gamma = discount * (1.0 - dones[irow].float())

            # Calculate TD Error
            td_error = rewards_current + gamma*values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * TAU * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE, returns to buffer
            GAE[irow] = GAE_current
            returns[irow] = returns_current

            values_next = values_current

        return GAE, returns
    gea_list, target_value_list = calc_returns(rewards = reward_list,
                                               values = value_list,
                                              dones=done_list)
    gea_list = (gea_list - gea_list.mean()) / (gea_list.std() + 1e-6)
    # return states, actions, rewards
    return prob_list, state_list, action_list, reward_list, value_list, gea_list, target_value_list


# In[5]:


import torch.optim as optim
from policy import Policy

# run your own policy!
policy=Policy(state_size=g_state_size,
              action_size=g_action_size,
              hidden_layers=[512, 256],
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
beta = .01
SGD_epoch = 10
episode = 150
batch_size = 128
#tmax = max(10*batch_size,int(30.0/0.1),1024)
tmax = batch_size

print_per_n = min(10,episode/10)
counter = 0
start_time = timeit.default_timer()

for e in range(episode):
    policy.eval()
    old_probs_lst, states_lst, actions_lst, rewards_lst, values_lst, gea, target_value = collect_trajectories(envs=g_env, 
                                                                                                              policy=policy, 
                                                                                                              tmax=tmax,
                                                                                                              nrand = 0,
                                                                                                              train_mode=True)

    average_total_rewards = np.sum(rewards_lst,0).mean()
    scores_window.append(average_total_rewards)

    # cat all agents
    old_probs_lst = old_probs_lst.reshape([-1])
    states_lst = states_lst.reshape([-1])
    actions_lst = actions_lst.reshape([-1])
    rewards_lst = rewards_lst.reshape([-1])
    values_lst = values_lst.reshape([-1])
    gea = gea.reshape([-1])
    target_value = target_value.reshape([-1])

    # gradient ascent step
    n_sample = len(old_probs_lst)//batch_size
    idx = np.arange(len(old_probs_lst))
    np.random.shuffle(idx)
    for b in range(n_sample):
        ind = idx[b*batch_size:(b+1)*batch_size]
        op = [old_probs_lst[i] for i in ind]
        s = [states_lst[i] for i in ind]
        a = [actions_lst[i] for i in ind]
        r = [rewards_lst[i] for i in ind]
        v = [values_lst[i] for i in ind]
        g = gea[ind]
        tv = target_value[ind]
        policy.train()
        for epoch in range(SGD_epoch):
            l_clip, critic_loss, entropy_loss, clipped_sur, states, actions, old_log_probs, advantages_batch = clipped_surrogate(policy=policy,
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
            _, log_probs, entropy, values = policy(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            ratio_clamped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            adv_PPO = torch.min(ratio * g, ratio_clamped * g)
            loss_actor = -torch.mean(adv_PPO) - 0.01 * entropy.mean()
            loss_critic = 0.5 * (tv - values).pow(2).mean()
            loss = loss_actor + loss_critic

            #L = -l_clip+critic_loss
            #L = -l_clip
            #print("-l_clip:{}".format(-l_clip), end="\n")
            #print("critic_loss:{}".format(critic_loss), end="\n")
            #print("clipped_sur:{}".format(torch.mean(clipped_sur)), end="\n")
            #print("L:{}".format(L))
            optimizer.zero_grad()
            # we need to specify retain_graph=True on the backward pass
            # this is because pytorch automatically frees the computational graph after
            # the backward pass to save memory
            # Without the computational graph, the chain of derivative is lost
            #L.backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
            optimizer.step()
            del(loss)

    # the clipping parameter reduces as time goes on
    #epsilon*=.999
    
    # the regulation term also reduces
    # this reduces exploration in later runs
    #beta*=.999
    
    # display some progress every 25 iterations
    if (e+1)%print_per_n ==0 :
        print("Episode: {0:d}, average score: {1:f}, beta: {2:f}".format(e+1,np.mean(scores_window), beta), end="\n")
    else:
        print("Episode: {0:d}, score: {1}".format(e+1, average_total_rewards), end="\r")
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


# In[ ]:


g_env.close()


# In[ ]:





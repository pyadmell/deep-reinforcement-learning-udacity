def collect_trajectories(envs, policy, tmax=200, nrand=5, train_mode=False):

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]
    value_list=[]

    env_info = envs.reset(train_mode=train_mode)[brain_name]

    # perform nrand random steps
    for _ in range(nrand):
        actions = np.random.randn(num_agents, action_size)
        actions = np.clip(actions, -1.0, 1.0)
        env_info = envs.step(actions)[brain_name]

    for t in range(tmax):
        state = torch.from_numpy(env_info.vector_observations).float().unsqueeze(0).to(device)
        est_action, value = policy(state)
        est_action = est_action.squeeze().cpu().detach()
        dist = torch.distributions.Normal(est_action, F.softplus(dist_std))
        action = dist.sample().numpy()
        value = value.squeeze().cpu().detach().numpy()

        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        env_info = envs.step(action)[brain_name]

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
    return prob_list, state_list, action_list, reward_list, value_list
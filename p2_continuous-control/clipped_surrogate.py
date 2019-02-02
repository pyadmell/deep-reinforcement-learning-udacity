import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#dist_std = torch.Tensor(nn.Parameter(torch.ones(1, action_size)))
dist_std = torch.Tensor(nn.Parameter(torch.zeros(1, action_size)))

def states_to_prob(policy, states):
    states = torch.stack(states)
    #policy_input = states.view(-1,*states.shape[-3:])
    #return policy(policy_input).view(states.shape[:-3])
    return policy(states)

def clipped_surrogate(policy, old_probs, states, actions, rewards, values,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):

    gea = [0.0] * len(states)
    target_value = [0.0] * len(states)
    advantages_ = 0.0
    returns_ = 0.0
    i_max = len(states)
    done = 0
    TAU = 0.95
    for i in reversed(range(i_max)):
        rwrds_ = rewards[i][0]
        values_ = values[i]
        next_value_ = values[min(i_max-1, i + 1)]
        if i+1==i_max:
            done = 1
        td_error = values_ - rwrds_ + discount * (1-done) * next_value_
        advantages_ = advantages_ * TAU * discount * (1-done) + td_error
        gea[i] = advantages_
        returns_ = discount*returns_ + rewards[i][0]
        target_value[i] = returns_

    gea = np.cumsum(gea)
    advantages_mean = np.mean(gea)
    advantages_std = np.std(gea) + 1.0e-10
    advantages_normalized = (gea - advantages_mean)/advantages_std

    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    advantages = torch.tensor(advantages_normalized, dtype=torch.float, device=device)
    values = torch.tensor(np.array(values), dtype=torch.float, device=device)
    target_value = torch.tensor(target_value, dtype=torch.float, device=device, requires_grad=False)

    states = torch.stack(states)
    est_actions, est_values = policy(states)
    dists = torch.distributions.Normal(est_actions, F.softplus(dist_std.to(device)))
    actions = dists.sample()

    log_prob = dists.log_prob(actions)
    log_prob = torch.sum(log_prob, dim=3, keepdim=True)
    new_probs = log_prob

    # entropy_loss = torch.Tensor(np.zeros((log_prob.size(0), 1)))
    entropy_loss = dists.entropy()
    entropy_loss = torch.sum(entropy_loss, dim=3, keepdim=True)/4.0

    # ratio for clipping
    ratio = (new_probs-old_probs).exp()

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_sur = torch.min(ratio*advantages,clip*advantages)

    #critic_loss = F.smooth_l1_loss(est_values.squeeze(),target_value.squeeze())
    critic_loss = 0.5 * (est_values.squeeze() - target_value.squeeze()).pow(2).mean()
    return torch.mean(clipped_sur + beta*entropy_loss), critic_loss, entropy_loss

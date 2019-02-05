import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, seed):
        """Initialize parameters and build policy.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (list(int)): Dimension of hidden layers
            seed (int): Random seed
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_size))
        # Shared hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        # Critic output layer
        c_hid = 32
        a_hid = 32
        self.critic_hidden = nn.Linear(hidden_layers[-1],c_hid)
        self.critic = nn.Linear(c_hid, 1)
        # Actor output layer
        self.actor_hidden = nn.Linear(hidden_layers[-1], a_hid)
        self.actor = nn.Linear(a_hid, action_size)
        # Apply Tanh() to bound the actions
        self.tanh = nn.Tanh()

        def init(n,mode='xavier-uniform'):
            if isinstance(n, nn.Linear):
                if mode=='xavier-uniform':
                    nn.init.xavier_uniform_(n.weight.data)
                elif mode=='xavier-normal':
                    nn.init.xavier_normal_(n.weight.data)
                elif mode=='orthogonal':
                    nn.init.orthogonal_(n.weight.data)
                elif mode=='uniform':
                    nn.init.uniform_(n.weight.data)
                elif mode=='normal':
                    nn.init.normal_(n.weight.data)
                else:
                    return
        self.hidden_layers.apply(init)
        self.critic_hidden.apply(init)
        self.actor_hidden.apply(init)
        self.critic.apply(init)
        self.actor.apply(init)

    def forward(self, state, action=None):
        """Build a network that maps state -> action and state -> value function."""
        for linear in self.hidden_layers:
            #state = F.relu(linear(state))
            state = F.leaky_relu(linear(state))

        v_hid = F.leaky_relu(self.critic_hidden(state))
        a_hid = F.leaky_relu(self.actor_hidden(state))
        a = F.tanh(self.actor(a_hid))
        value = self.critic(v_hid).squeeze(-1)
        dist = torch.distributions.Normal(a, F.softplus(self.sigma))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return action, log_prob, entropy, value
        #return torch.tanh(self.actor(state)), self.critic(state)

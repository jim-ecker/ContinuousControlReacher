import torch
import torch.nn as nn
import torch.nn.functional as F


# class FC(nn.Module):
#
#     def __init__(self, state_size, action_size, output_gate=None):
#
#         super(FC, self).__init__()
#
#         self.fc1 = nn.Linear(state_size, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, action_size)
#         self.output_gate = output_gate
#
#     def forward(self, input):
#         x = F.leaky_relu(self.fc1(input))
#         x = F.leaky_relu(self.fc2(x))
#         x = self.fc3(x)
#
#         if self.output_gate is not None:
#             x = self.output_gate(x)
#
#         return x

class FullyConnected(nn.Module):
    def __init__(self, input, layers, activation=F.leaky_relu):
        super(FullyConnected, self).__init__()
        self.activation = activation
        self.init_layers(nn.Linear(input, layers[0]), layers)

    def init_layers(self, fc1, layers):
        self.layers = nn.ModuleList([fc1])
        self.layers.extend(nn.Linear(l1, l2) for l1, l2 in zip(layers[:1], layers[1:]))
        self.layers.apply(self.init_xavier)

    def init_xavier(self, x):
        if isinstance(x, nn.Linear):
            nn.init.xavier_uniform_(x.weight.data)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class Policy(nn.Module):
    def __init__(self, states, actions, hidden_layers=[64, 64]):
        super(Policy, self).__init__()
        self.device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_states  = states
        self.num_actions = actions
        self.fc1         = FullyConnected(self.num_states, hidden_layers)
        self.actor       = nn.Linear(hidden_layers[-1], self.num_actions)
        self.critic      = nn.Linear(hidden_layers[-1], 1)
        self.std         = nn.Parameter(torch.ones(1, self.num_actions)).to(self.device)
        self.to(self.device)

    def forward(self, states, actions=None):
        features = self.fc1(states)
        mean     = F.tanh(self.actor(features))
        values   = self.critic(features)
        distribution = torch.distributions.Normal(mean, self.std)
        if actions is None:
            actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        return actions, log_prob, values

    def state_values(self, states):
        features = self.fc1(states)
        return self.critic(features).squeeze(-1)


# class Policy(nn.Module):
#
#     def __init__(self, state_size, action_size, init=True):
#         super(Policy, self).__init__()
#         self.device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
#         self.actor  = FC(state_size, action_size, F.tanh)
#         self.critic = FC(state_size, 1)
#         self.std = nn.Parameter(torch.ones(1, action_size)).to(self.device)
#         self.to(self.device)
#
#     def forward(self, state, action=None):
#         a = self.actor(state)
#         distribution = torch.distributions.Normal(a, self.std)
#         if action is None:
#             action = distribution.sample()
#         log_prob = distribution.log_prob(action)
#         log_prob = torch.sum(log_prob, dim=1, keepdim=True)
#         value = self.critic(state)
#
#         return action, log_prob, value

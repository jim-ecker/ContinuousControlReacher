import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple



# class FullyConnected(nn.Module):
#     def __init__(self, input, layers, activation=F.leaky_relu):
#         super(FullyConnected, self).__init__()
#         self.activation = activation
#         self.init_layers(nn.Linear(input, layers[0]), layers)
#
#     def init_layers(self, fc1, layers):
#         self.layers = nn.ModuleList([fc1])
#         self.layers.extend(nn.Linear(l1, l2) for l1, l2 in zip(layers[:1], layers[1:]))
#         self.layers.apply(self.init_xavier)
#
#     def init_xavier(self, x):
#         if isinstance(x, nn.Linear):
#             nn.init.xavier_uniform_(x.weight.data)
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = self.activation(layer(x))
#         return x



class FullyConnected(nn.Module):

    def __init__(self, layers, output_gate=None, activation=F.leaky_relu):

        super(FullyConnected, self).__init__()
        self.build_network(layers)
        self.output_gate = output_gate

    def __repr__(self):
        import pandas as pd
        from tabulate import tabulate

        network_table = []
        for k, v in self.layers.items():
            network_table.append((k, str(v)))
        return tabulate(pd.DataFrame.from_records(network_table), tablefmt='fancy_grid', showindex='never')

    def build_network(self, layers):
        self.layers = nn.ModuleList([nn.Linear(layers[0].input, layers[0].output)])
        self.layers.extend(nn.Linear(layer.input, layer.output) for layer in layers[1:])
        self.layers.apply(self.init_xavier)

    def init_xavier(self, x):
        if isinstance(x, nn.Linear):
            nn.init.xavier_uniform_(x.weight.data)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        if self.output_gate is not None:
            x = self.output_gate(x)
        return x


class PPONetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(PPONetwork, self).__init__()
        self.layer = namedtuple("Layer", field_names=['name', 'input', 'output'])
        self.device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
        self.actor = FullyConnected(
            layers=[
                self.layer('fc1', state_size, 256),
                self.layer('fc2', 256, 256),
                self.layer('fc3', 256, action_size)
            ],
            output_gate=F.tanh
        )
        print(self.actor)
        # self.actor  = FC(state_size, action_size, F.tanh)
        self.critic = FullyConnected(
            layers=[
                self.layer('fc1', state_size, 256),
                self.layer('fc2', 256, 256),
                self.layer('fc3', 256, 1)
            ]
        )
        print(self.critic)
        # self.critic = FC(state_size, 1)
        self.std = nn.Parameter(torch.ones(1, action_size)).to(self.device)
        self.to(self.device)

    def forward(self, state, action=None):
        a = self.actor(state)
        distribution = torch.distributions.Normal(a, self.std)
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        value = self.critic(state)

        return action, log_prob, value

# class PPONetwork(nn.Module):
#     def __init__(self, state_size, action_size, seed, architecture=None):
#         super(PPONetwork, self).__init__()
#         if architecture is None:
#             architecture = {
#
#             }
#         self.device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.state_size  = state_size
#         self.action_size = action_size
#         self.fc1         = FullyConnected(self.num_states, hidden_layers)
#         self.actor       = nn.Linear(hidden_layers[-1], self.num_actions)
#         self.critic      = nn.Linear(hidden_layers[-1], 1)
#         self.std         = nn.Parameter(torch.ones(1, self.num_actions)).to(self.device)
#         self.to(self.device)
#
#     def forward(self, states, actions=None):
#         features = self.fc1(states)
#         mean     = F.tanh(self.actor(features))
#         values   = self.critic(features)
#         distribution = torch.distributions.Normal(mean, self.std)
#         if actions is None:
#             actions = distribution.sample()
#         log_prob = distribution.log_prob(actions)
#         log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
#         return actions, log_prob, values
#
#     def state_values(self, states):
#         features = self.fc1(states)
#         return self.critic(features).squeeze(-1)




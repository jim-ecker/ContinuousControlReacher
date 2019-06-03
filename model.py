import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, state_size, action_size, output_gate=None):

        super(FC, self).__init__()

        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)
        self.output_gate = output_gate

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        if self.output_gate is not None:
            x = self.output_gate(x)

        return x


class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, init=True):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
        self.actor  = FC(state_size, action_size, F.tanh)
        self.critic = FC(state_size, 1)
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

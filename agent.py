import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import ActorCritic
from utils import timeit

class Agent:

    def __init__(self, env, name, num_agents, size_state, size_action, discount_rate=0.99, tau=0.95, num_rollout=10,
                 clip_ppo=0.2, clip_gradient=0.5, num_minibatch=64):
        self.env           = env
        self.name          = name
        self.num_agents    = num_agents
        self.size_state    = size_state
        self.size_action   = size_action
        self.network       = ActorCritic(size_state, size_action)
        self.optimizer     = optim.Adam(self.network.parameters(), 2e-4, eps=1e-5)
        self.discount_rate = discount_rate
        self.tau           = tau
        self.num_rollout   = num_rollout
        self.clip_ppo      = clip_ppo
        self.clip_gradient = clip_gradient
        self.num_minibatch = num_minibatch

    def generate_rollout(self):
        rollout = []
        episode_rewards = np.zeros(self.num_agents)

        #reset env
        env_info = self.env.reset(train_mode=True)[self.name]
        states   = env_info.vector_observations

        while True:
            states = torch.Tensor(states).cuda()
            actions, log_probs, values = self.network(states)
            env_info = self.env.step(actions.cpu().detach().numpy())[self.name]
            rewards = env_info.rewards
            dones = np.array(env_info.local_done)
            next_states = env_info.vector_observations

            episode_rewards += rewards
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - dones])
            states = next_states

            if np.any(dones):
                break

        states = torch.Tensor(states).cuda()
        _, _, last_value = self.network(states)
        rollout.append([states, last_value, None, None, None, None])

        return rollout, last_value, episode_rewards

    def process_rollout(self, rollout, last_value):
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.zeros((self.num_agents, 1)).cuda()
        returns = last_value.detach()

        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, dones = rollout[i]
            dones = torch.Tensor(dones).unsqueeze(1).cuda()
            rewards = torch.Tensor(rewards).unsqueeze(1).cuda()
            next_value = rollout[i + 1][1]
            returns = rewards + self.discount_rate * dones * returns

            td_error = rewards + self.discount_rate * dones * next_value.detach() - value.detach()
            advantages = advantages * self.tau * self.discount_rate * dones + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        return processed_rollout

    def train_network(self, states, actions, log_probs_old, returns, advantages):

        batcher = Batcher(states.size(0) // self.num_minibatch, [np.arange(states.size(0))])
        for _ in range(self.num_rollout):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = torch.Tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.clip_ppo, 1.0 + self.clip_ppo) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0)

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_gradient)
                self.optimizer.step()

    def step(self):
        # Run a single episode to generate a rollout
        rollout, last_value, episode_rewards = self.generate_rollout()
        # Process the rollout to calculate advantages
        processed_rollout = self.process_rollout(rollout, last_value)
        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0),
                                                                  zip(*processed_rollout))
        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / advantages.std()
        # Train the network
        self.train_network(states, actions, log_probs_old, returns, advantages)
        # Return the average reward across all agents for this episode
        return np.mean(episode_rewards)


class Batcher:

        def __init__(self, batch_size, data):
            self.batch_size = batch_size
            self.data = data
            self.num_entries = len(data[0])
            self.reset()

        def reset(self):
            self.batch_start = 0
            self.batch_end = self.batch_start + self.batch_size

        def end(self):
            return self.batch_start >= self.num_entries

        def next_batch(self):
            batch = []
            for d in self.data:
                batch.append(d[self.batch_start: self.batch_end])
            self.batch_start = self.batch_end
            self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
            return batch

        def shuffle(self):
            indices = np.arange(self.num_entries)
            np.random.shuffle(indices)
            self.data = [d[indices] for d in self.data]
            # JoshVarty: We must call reset() after shuffling or else we
            # won't be able to iterate over the newly shuffled data
            self.reset()

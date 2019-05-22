from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from agent import Agent
from model import ActorCritic
import time

#Load the visualization of the environment
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', worker_id=12)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment (Don't use training mode so we can easily watch)
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# size of each state
state_size = brain.vector_observation_space_size
print('Size of each state:', state_size)


def visualize(agent):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations

    while True:
        states = torch.Tensor(states).cuda()
        actions, log_probs, values = agent.network(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        rewards = env_info.rewards
        dones = np.array(env_info.local_done)
        next_states = env_info.vector_observations

        states = next_states

        time.sleep(0.005)

        if np.any(dones):
            break


agent = Agent(env, brain_name, num_agents, state_size, action_size)
checkpoint = torch.load('solution.ckpt')
agent.network.load_state_dict(checkpoint)
visualize(agent)
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from agent import Agent
from utils import timeit

env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset env
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('num_agents:', num_agents)

# size of each action
size_action = brain.vector_action_space_size
print('size_action:', size_action)

# size of each state
size_state = brain.vector_observation_space_size
print('size_state:', size_state)


@timeit
def a2c(agent, num_agents, num_episodes=400):

    all_scores = []
    scores_window = deque(maxlen=100)
    print("\nTraining {} on {} environment".format(agent.__class__.__name__, brain_name))
    for i_episode in range(1, num_episodes + 1):

        avg_score = agent.step()
        scores_window.append(avg_score)
        all_scores.append(avg_score)

        if i_episode % 25 == 0:
            print("episode {}\tAverage score: {}{}".format(i_episode, np.mean(scores_window), " " * 100))
        else:
            print("episode {}\tAverage score: {}{}".format(i_episode, np.mean(scores_window), " "*100), end="\r")
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.network.state_dict(), 'solution.ckpt')
            break

    return all_scores


#build networks
agent = Agent(env, brain_name, num_agents, size_state, size_action)
scores = a2c(agent, num_agents)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
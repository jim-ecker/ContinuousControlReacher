from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
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

style.use('fivethirtyeight')
fig = plt.figure()
plt.ylim(0, 35)
axis = fig.add_subplot(1,1,1)


@timeit
def ppo(agent, num_agents, num_episodes=400):
    all_scores = []
    scores_window = deque(maxlen=100)
    print("\nTraining {} on {} environment".format(agent.__class__.__name__, brain_name))
    for i_episode in range(1, num_episodes + 1):

        avg_score = agent.step()
        scores_window.append(avg_score)
        all_scores.append(avg_score)
        with open('scores.txt', 'a') as f:
            f.write("{}\n".format("{},{}".format(str(i_episode), str(avg_score))))
        logstr = "episode {}\tAverage score: {}{}\r".format(i_episode, np.mean(scores_window), " " * 100)
        if i_episode % 25 == 0:
            logstr += "\n"
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.network.state_dict(), 'solution.ckpt')
            break
        else:
            print(logstr, end="")

    # return all_scores


def animate(i):
    with open('scores.txt', 'r') as f:
        data = f.read()
    lines = data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    axis.clear()
    axis.plot(xs, ys)


def liveplot():
    ani = animation.FuncAnimation(fig, animate, interval=10000)
    plt.show()


# build networks
agent = Agent(env, brain_name, num_agents, size_state, size_action)

p1 = multiprocessing.Process(target=liveplot)
p1.start()
ppo(agent, num_agents)
p1.join()

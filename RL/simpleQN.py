from curses import curs_set
import random
from cv2 import QT_STYLE_ITALIC

from torch import q_scale
import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
SHOW_EVERY = 20000
high = (env.observation_space.high)
low = (env.observation_space.low)
TABLE_SIZE = 20
DIM = len(high)
TABLE_DIM = [TABLE_SIZE for i in range(DIM)]
step_size = (high - low ) / TABLE_DIM

q_table = np.random.uniform(low=-2, high = 0, size=(TABLE_DIM + [env.action_space.n]))

def disctretise(state):
    return tuple(((state - low) / step_size).astype(np.int))


for episode in range(EPISODES):
    cur_state = disctretise(env.reset())
    done = False
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    while not done:
        action = (np.argmax(q_table[cur_state]))
        new_state, reward, done , _ = env.step(action)
        new_disc_state = disctretise(new_state)
        if render:
            env.render()
        if not done:
            exp_maxq = np.max(q_table[new_disc_state])
            cur_q = q_table[cur_state + (action, )]

            new_q = (1 - LEARNING_RATE) * cur_q + LEARNING_RATE * (reward + DISCOUNT * exp_maxq)
            q_table[cur_state + (action, )] = new_q
        elif new_state[0] > env.goal_position:
            print(f"We made it on episode:{episode}")
            q_table[cur_state + (action, )] = 0

        cur_state = new_disc_state

env.close()
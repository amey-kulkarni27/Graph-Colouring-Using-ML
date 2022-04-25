from tkinter.tix import MAX
import numpy as np
import networkx as nx
import matplotlib
from PIL import Image
import cv2
import pickle
import time
import warnings
from matplotlib import style
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/amey.kulkarni/Graph-Colouring-Using-ML/ML')
from generate_kpart import gen_kpart, display_graph
from feature_vector import feature_vector
from operations import operations, vertex_pair_non_edge, get_action


style.use('ggplot')

K = 3 # colours
N = 30 # vertices of each colour
DENSITY = 0.3
FV_LEN = 3 # length of feature vector
METHOD = 'topk'
MAX_DIST = 2
BUCKETS = 50
NUM_ACTIONS = 3 # 3rd action is don't do anything
UPDATE_INTERVAL = 1 # update the feature vector after every _ turns

HM_EPISODES = 50_000
REWARD = 20
PENALTY = 10
TURN = 1

epsilon = 0.9
EPS_DECAY = 0.9998

SHOW_EVERY = 100

start_q_table = None # or filename to load q_table from

LEARNING_RATE = 0.1
DISCOUNT = 0.95

class Graph:
    def __init__(self):
        self.G, self.coords = gen_kpart(K, N, DENSITY)
        # display_graph(G, coords)
    def update_fv(self):
        self.fv = feature_vector(self.G, method=METHOD, k=FV_LEN)
    def dist(self, u, v):
        # print(len(self.G.nodes), len(self.fv))
        return np.linalg.norm(self.fv[u] - self.fv[v])

# if we don't have a q_table to load from, we create one
if start_q_table is None:
    q_table = {}
    for d in range(BUCKETS):
        q_table[d] = [np.random.uniform(-5, 0) for i in range(NUM_ACTIONS)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


episode_rewards = []
for episode in range(HM_EPISODES):
    G_obj = Graph()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        # print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    cnt = 0
    while True:
        if cnt == 0:
            G_obj.update_fv()
        nodes = vertex_pair_non_edge(G_obj.G)
        d = G_obj.dist(nodes[0], nodes[1])
        if d >= MAX_DIST:
            d = MAX_DIST
        pos = int((d / MAX_DIST) * BUCKETS)
        if pos == BUCKETS:
            pos -= 1
        
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[pos])
        else:
            action = np.random.randint(0, NUM_ACTIONS)
        if action < 2:
            G_obj.G = operations(G_obj.G, action, nodes)
            N = len(G_obj.G.nodes)
            mapping = {old: new for (old, new) in zip(G_obj.G.nodes, [i for i in range(N)])}
            G_obj.G = nx.relabel_nodes(G_obj.G, mapping)
            # print(len(G_obj.G.nodes), action)
            cnt += 1
            cnt %= UPDATE_INTERVAL

        if (vertex_pair_non_edge(G_obj.G)) == False:
            break

        reward = -TURN
        nxt_nodes = vertex_pair_non_edge(G_obj.G)
        d = G_obj.dist(nxt_nodes[0], nxt_nodes[1])
        if d >= MAX_DIST:
            d = MAX_DIST
        new_pos = int((d / MAX_DIST) * BUCKETS)
        if new_pos == BUCKETS:
            new_pos -= 1
        max_future_q = np.max(q_table[new_pos])
        current_q = q_table[pos][action]

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[pos][action] = new_q
        episode_reward += reward

    cols = len(G_obj.G.nodes())
    if cols == K:
        reward = REWARD
    elif cols == K + 1:
        reward = REWARD // 4
    else:
        reward = -PENALTY
    q_table[pos][action] = reward
    episode_reward += reward
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode="valid")


plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel("episode #")
# plt.show()
plt.savefig('qtable.png')

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

# if __name__ == 'main':
# G_obj = Graph()
# G_obj.update_fv()
# print(G_obj.fv)


# 1) Give smaller reward for reaching close to optimal (increase penalty with mistakes)
# 2) Do not perform any action. Done
# 3) Calculate distance between 5 pairs while calculating next_nodes, keep the one with the smallest d for max_future_q
# 4) (Euclidean Norm, Similarity (dot product), )
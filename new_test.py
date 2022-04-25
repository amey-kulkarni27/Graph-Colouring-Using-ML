import pickle
import numpy as np
import networkx as nx
import sys
from matplotlib import style
import matplotlib.pyplot as plt
style.use('ggplot')


sys.path.insert(0, '/home/amey.kulkarni/Graph-Colouring-Using-ML/ML')
from generate_kpart import gen_kpart, display_graph
from feature_vector import feature_vector
from operations import operations, vertex_pair_non_edge, get_action

start_q_table = "RL/qtable-1650879401.pickle"
with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

k = 3
n = 30 # Number of nodes in a single partition
p = 0.3
num_tests = 100
MAX_DIST = 2
BUCKETS = 50
FV_LEN = 3 # length of feature vector
METHOD = 'topk'

colours = [0 for i in range(91)]

def dist(fv, u, v):
    return np.linalg.norm(fv[u] - fv[v])

for _ in range(num_tests):
    G, coords = gen_kpart(k, n, p)
    while (vertex_pair_non_edge(G)) != False:
        fv = feature_vector(G, method=METHOD, k=FV_LEN)
        nodes = vertex_pair_non_edge(G)
        d = dist(fv, nodes[0], nodes[1])
        if d >= MAX_DIST:
            d = MAX_DIST
        pos = int((d / MAX_DIST) * BUCKETS)
        if pos == BUCKETS:
            pos -= 1
        action = np.argmax(q_table[pos])
        if action < 2:
            G = operations(G, action, nodes)
            N = len(G.nodes)
            mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
            G = nx.relabel_nodes(G, mapping)
    cols = len(G.nodes)
    colours[cols] += 1
print(colours)
plt.bar([i for i in range(91)], colours, )
plt.xlabel("Final Chromatic Number")
plt.ylabel("Frequency")
plt.title("Q Table performance")
plt.savefig("bar_qtable.png")
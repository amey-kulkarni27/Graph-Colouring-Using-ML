from keras.models import model_from_json
import numpy as np
import networkx as nx
import sys
from matplotlib import style
import matplotlib.pyplot as plt
from tqdm import tqdm
style.use('ggplot')


sys.path.insert(0, '/home/amey.kulkarni/Graph-Colouring-Using-ML/ML')
from generate_kpart import gen_kpart, display_graph
from feature_vector import feature_vector
from operations import operations, vertex_pair_non_edge, get_action

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

k = 3
n = 30 # Number of nodes in a single partition
p = 0.3
num_tests = 100
FV_LEN = 4 # length of feature vector
OBSERVATION_SPACE_VALUES = FV_LEN
ACTION_SPACE_VALUES = 3
METHOD = 'topk'

colours = [0 for i in range(91)]


for _ in tqdm(range(num_tests)):
    G, coords = gen_kpart(k, n, p)
    streak = 0 # streak of action=2
    while (vertex_pair_non_edge(G)) != False:
        fv = feature_vector(G, method=METHOD, k=FV_LEN)
        nodes = vertex_pair_non_edge(G)
        inp = abs(fv[nodes[0]] - fv[nodes[1]])
        inp = (np.asarray(inp)).reshape(-1, OBSERVATION_SPACE_VALUES)
        action = np.argmax(loaded_model.predict(inp))
        if action < 2:
            G = operations(G, action, nodes)
            N = len(G.nodes)
            mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
            G = nx.relabel_nodes(G, mapping)
            streak = 0
        else:
            streak += 1
            if streak == 3:
                action = 1 ^ np.argmin(loaded_model.predict(inp))
                G = operations(G, action, nodes)
                N = len(G.nodes)
                mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
                G = nx.relabel_nodes(G, mapping)
                streak = 0
    cols = len(G.nodes)
    colours[cols] += 1
print(colours)
plt.bar([i for i in range(91)], colours, )
plt.xlabel("Final Chromatic Number")
plt.ylabel("Frequency")
plt.title("DQN performance")
plt.savefig("dqn_bar_qtable.png")
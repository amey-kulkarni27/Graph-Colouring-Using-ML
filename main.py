from operations import operations, vertex_pair, vertex_pair_opt, get_action
from generate_kpart import gen_kpart, display_graph
from feature_vector import feature_vector
from operations import operations
import numpy as np
from sklearn.linear_model import LogisticRegression


num_graphs = 10
k = 3
n = 4 # Number of nodes in a single partition
p = 0.4
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
X = []
y = []
for G, coords in G_list:
    vec = feature_vector(G, method='topk')
    # nodes = vertex_pair(G, n * k)
    nodes = vertex_pair_opt(G)
    action = get_action(n, nodes)
    X.append(np.concatenate((vec[nodes[0]], vec[nodes[1]])))
    y.append(action)
    # display_graph(G, coords)
    G = operations(G, action, nodes)
    # display_graph(G, coords)

X = np.array(X)
y = np.array(y)
print(X)
print(y)
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.predict(X))
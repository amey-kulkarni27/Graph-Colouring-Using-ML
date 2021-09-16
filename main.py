from operations import operations, vertex_pair, vertex_pair_opt, get_label
from generate_kpart import gen_kpart, display_graph
from feature_vector import feature_vector
from operations import operations
import random


num_graphs = 1
k = 3
n = 4
p = 0.4
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
for G, coords in G_list:
    vec = feature_vector(G)
    merge = random.uniform(0, 1) < 0.5
    # nodes = vertex_pair(G, n * k)
    nodes = vertex_pair_opt(G)
    label = get_label(n, merge, nodes)
    print(merge, nodes, label)
    display_graph(G, coords)
    G = operations(G, merge, nodes)
    display_graph(G, coords)
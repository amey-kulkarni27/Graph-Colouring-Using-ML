import numpy as np
from operations import operations, vertex_pair, vertex_pair_opt, get_action
from feature_vector import feature_vector
from generate_kpart import display_graph
import networkx as nx

def train(G, coords, X, y, n, k, interval=1):
    '''
    G -> Graph on which we train
    X -> Feature vectors to be appended
    y -> Labels to be appended
    n -> Number of nodes in each of the independent sets
    k -> Number of independent sets
    interval -> Feature vector for graph to be updated in these many steps
    
    Update X and y as the graph is completed edge by edge
    '''
    # display_graph(G, coords)
    cnt = 0
    z=0
    while (vertex_pair_opt(G)) != False:
        if cnt == 0:
            N = len(G.nodes)
            vec = feature_vector(G, method='topk', k=k)
            mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
            G = nx.relabel_nodes(G, mapping)

        # nodes = vertex_pair(G, n * k)
        nodes = vertex_pair_opt(G)
        action = get_action(G, nodes)
        x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
        X.append(x)
        y.append(action)
        G = operations(G, action, nodes)
        # display_graph(G, coords)
        cnt += 1
        cnt %= interval
        z+=1
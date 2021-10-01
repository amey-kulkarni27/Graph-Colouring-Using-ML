import numpy as np
from operations import operations, vertex_pair, vertex_pair_opt, get_action
from feature_vector import feature_vector
from generate_kpart import display_graph
import networkx as nx

def train(G, coords, X, y, n, interval=3):
    '''
    G -> Graph on which we train
    X -> Feature vectors to be appended
    y -> Labels to be appended
    n -> Number of nodes in each of the independent sets
    interval -> Feature vector for graph to be updated in these many steps
    
    Update X and y as the graph is completed edge by edge
    '''
    # display_graph(G, coords)
    cnt = 0
    z=0
    while (vertex_pair_opt(G)) != False:
        if cnt == 0:
            vec = feature_vector(G, method='topk', k=2)
            N = len(G.nodes)
            mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
            G = nx.relabel_nodes(G, mapping)

        # nodes = vertex_pair(G, n * k)
        nodes = vertex_pair_opt(G)
        action = get_action(G, nodes)
        x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
        # print(x.shape)
        X.append(x)
        y.append(action)
        G = operations(G, action, nodes)
        # display_graph(G, coords)
        cnt += 1
        cnt %= interval
        z+=1
    print(z)
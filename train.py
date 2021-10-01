import numpy as np
from operations import operations, vertex_pair, vertex_pair_opt, get_action
from feature_vector import feature_vector

def train(G, X, y, n):
    '''
    G -> Graph on which we train
    X -> Feature vectors to be appended
    y -> Labels to be appended
    n -> Number of nodes in each of the independent sets
    
    '''
    vec = feature_vector(G, method='topk')
    # nodes = vertex_pair(G, n * k)
    nodes = vertex_pair_opt(G)
    action = get_action(n, nodes)
    X.append(np.concatenate((vec[nodes[0]], vec[nodes[1]])))
    y.append(action)
    G = operations(G, action, nodes)
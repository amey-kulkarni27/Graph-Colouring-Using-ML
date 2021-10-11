import numpy as np
from operations import operations, vertex_pair_opt
from feature_vector import feature_vector
from generate_kpart import display_graph
import networkx as nx
import copy

def correctness(G_init, G):
    for u in G.nodes():
        lbls = list(G.nodes[u]['label'])
        for i in range(len(lbls)):
            for j in range(i + 1, len(lbls)):
                assert(G_init.has_edge(lbls[i], lbls[j]) == False)

def test(G, coords, clf, n, k, interval=1):
    '''
    G -> Graph on which we train
    n -> Number of nodes in each of the independent sets
    k -> Number of independent sets
    interval -> Feature vector for graph to be updated in these many steps
    clf -> Classifier that has been trained
    
    Perform actions on X and y depending upon the classifier
    '''
    
    G_init = copy.deepcopy(G)
    # display_graph(G, coords)
    cnt = 0
    z=0
    while (vertex_pair_opt(G)) != False:
        if cnt == 0:
            vec = feature_vector(G, method='topk', k=k-1)

        nodes = vertex_pair_opt(G)
        x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
        x = x.reshape(1, -1)
        action = clf.predict(x)
        G = operations(G, action, nodes)
        # display_graph(G, coords)
        cnt += 1
        cnt %= interval
        z+=1
        N = len(G.nodes)
        mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
        G = nx.relabel_nodes(G, mapping)
    print(len(G.nodes()), k)

    correctness(G_init, G)
    
    
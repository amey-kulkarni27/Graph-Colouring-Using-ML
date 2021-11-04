from operator import concat
import numpy as np
from scipy.sparse import coo
from operations import operations, vertex_pair_opt, vertex_pair_non_edge
from feature_vector import feature_vector
from generate_kpart import display_graph
import networkx as nx
import copy
from timeit import default_timer as timer

def correctness(G_init, G):
    for u in G.nodes():
        lbls = list(G.nodes[u]['label'])
        for i in range(len(lbls)):
            for j in range(i + 1, len(lbls)):
                # if G_init.has_edge(lbls[i], lbls[j]):
                    # print(lbls[i], lbls[j])
                    # display_graph(G_init)
                    # display_graph(G)
                assert(G_init.has_edge(lbls[i], lbls[j]) == False)

def test(G, coords, clf, n, k, interval=1, method="top_k"):
    '''
    G -> Graph on which we test
    n -> Number of nodes in each of the independent sets
    k -> Number of independent sets
    interval -> Feature vector for graph to be updated in these many steps
    clf -> Classifier that has been trained
    
    Perform actions on X and y depending upon the classifier
    '''
    
    G_init = copy.deepcopy(G)
    # display_graph(G, coords)
    cnt = 0
    steps, update_steps = 0, 0
    fvt, nt, concatt, at, mpt = 0, 0, 0, 0, 0
    while (vertex_pair_non_edge(G)) != False:
        if cnt == 0:
            t1 = timer()
            vec = feature_vector(G, method=method, k=k)
            t2 = timer()
            update_steps += 1
            fvt += t2 - t1

        t3 = timer()
        nodes = vertex_pair_non_edge(G)
        t4 = timer()
        x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
        x = x.reshape(1, -1)
        action = clf.predict(x)
        t5 = timer()
        G = operations(G, action, nodes)
        # display_graph(G, coords)
        cnt += 1
        cnt %= interval
        N = len(G.nodes)
        t6 = timer()
        mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
        G = nx.relabel_nodes(G, mapping)
        t7 = timer()
        steps += 1
        nt += t4 - t3
        at += t5 - t4
        mpt += t7 - t6
    print("Steps Taken:", steps)
    print("Feature Vector:", round(fvt, 2))
    print("Nodes:", round(nt, 2))
    print("Concatenation:", round(concatt, 2))
    print("Predict:", round(at, 2))
    print("Remapping:", round(mpt, 2))
    print()
    # display_graph(G, coords)

    # correctness(G_init, G)
    return G
    
    
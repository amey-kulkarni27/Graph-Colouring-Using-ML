import numpy as np
from operations import operations, vertex_pair, vertex_pair_opt, vertex_pair_non_edge, get_action
from feature_vector import feature_vector
from generate_kpart import display_graph
import networkx as nx
from timeit import default_timer as timer

def train(G, coords, X, y, n, k, interval=1, method="top_k"):
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
    steps, update_steps = 0, 0
    condt, fvt, nt, concatt, at, mpt = 0, 0, 0, 0, 0, 0
    # while (vertex_pair_non_edge(G)) != False:
    while True:
        t01 = timer()
        if (vertex_pair_non_edge(G)) == False:
            break
        t02 = timer()
        if cnt == 0:
            t1 = timer()
            vec = feature_vector(G, method=method, k=k)
            update_steps += 1
            t2 = timer()
            fvt += t2 - t1

        # nodes = vertex_pair(G, n * k)
        t3 = timer()
        nodes = vertex_pair_non_edge(G)
        t4 = timer()
        action = get_action(G, nodes)
        t5 = timer()
        x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
        X.append(x)
        y.append(action)
        G = operations(G, action, nodes)
        # display_graph(G, coords)
        cnt += 1
        cnt %= interval
        steps+=1
        N = len(G.nodes)
        t6 = timer()
        mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
        G = nx.relabel_nodes(G, mapping)
        t7 = timer()
        condt += t02 - t01
        nt += t4 - t3
        at += t5 - t4
        concatt += t6 - t5
        mpt += t7 - t6
    print("Steps Taken:", steps)
    print("Condition Checking:", round(condt, 2))
    print("Feature Vector:", round(fvt, 2))
    print("Nodes:", round(nt, 2))
    print("Concatenation:", round(concatt, 2))
    print("Action:", round(at, 2))
    print("Remapping:", round(mpt, 2))
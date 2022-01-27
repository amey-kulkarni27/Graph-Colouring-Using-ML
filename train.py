import numpy as np
from numpy import random
from scipy.sparse.construct import rand
from operations import operations, vertex_pair, vertex_pair_opt, vertex_pair_non_edge, vertex_pair_coin_toss, get_action
from feature_vector import feature_vector
from generate_kpart import display_graph
import networkx as nx
from timeit import default_timer as timer

def train(G, coords, X, y, n, k, interval=1, method="top_k", coin_toss=False, sub=True):
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
    fvt, nt, concatt, at, mpt = 0, 0, 0, 0, 0
    while (vertex_pair_non_edge(G)) != False:
        if cnt == 0:
            t1 = timer()
            vec = feature_vector(G, method=method, k=k)
            update_steps += 1
            t2 = timer()
            fvt += t2 - t1

        # nodes = vertex_pair(G, n * k)
        t3 = timer()
        if coin_toss:
            op = random.randint(0, 2)
            nodes = vertex_pair_coin_toss(G, op)
        else:
            nodes = vertex_pair_non_edge(G)
        t4 = timer()
        action = get_action(G, nodes)
        t5 = timer()
        if(sub):
                x = abs(vec[nodes[1]] - vec[nodes[0]])
        else:
            x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
        X.append(x)
        y.append(action)
        G = operations(G, action, nodes) # p probability of correct action
        # display_graph(G, coords)
        cnt += 1
        cnt %= interval
        steps+=1
        N = len(G.nodes)
        t6 = timer()
        mapping = {old: new for (old, new) in zip(G.nodes, [i for i in range(N)])}
        G = nx.relabel_nodes(G, mapping)
        t7 = timer()
        nt += t4 - t3
        at += t5 - t4
        concatt += t6 - t5
        mpt += t7 - t6
    # print("Steps Taken:", steps)
    # print("Feature Vector:", round(fvt, 2))
    # print("Nodes:", round(nt, 2))
    # print("Concatenation:", round(concatt, 2))
    # print("Action:", round(at, 2))
    # print("Remapping:", round(mpt, 2))

def train_static(G, coords, X, y, n, k, method="top_k", equal_cl=True, sub=True):
    N = len(G.nodes)
    vec = feature_vector(G, method=method, k=k)
    if equal_cl:
        cl_size = ((n * (n - 1)) // 2) * k
        s1 = 0
        for col in range(k):
            start = col * n
            for i in range(start, start + n):
                for j in range(i + 1, start + n):
                    nodes = (i, j)
                    s1 += 1
                    action = get_action(G, nodes)
                    assert(action == 0)
                    if(sub):
                        x = abs(vec[nodes[1]] - vec[nodes[0]])
                    else:
                        x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
                    X.append(x)
                    y.append(action)
        assert(s1 == cl_size)
        # cl_size *= 2
        while(cl_size > 0):
            op = 1
            nodes = vertex_pair_coin_toss(G, op)
            action = get_action(G, nodes)
            assert(action == 1)
            if(sub):
                x = abs(vec[nodes[1]] - vec[nodes[0]])
            else:
                x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
            X.append(x)
            y.append(action)
            cl_size -= 1
    else:
        for i in range(N):
            for j in range(i + 1, N):
                nodes = (i, j)
            action = get_action(G, nodes)
            if(sub):
                x = abs(vec[nodes[1]] - vec[nodes[0]])
            else:
                x = np.concatenate((vec[nodes[0]], vec[nodes[1]]))
            X.append(x)
            y.append(action)
       
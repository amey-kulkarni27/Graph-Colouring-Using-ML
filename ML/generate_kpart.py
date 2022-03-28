import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
# Generate k-partite graphs.


def set_coords(k, n):
    x = 0
    y = 0
    delta = 0.1
    coords = dict()
    for i in range(n * k):
        coords[i] = (x, y)
        y += 1
        if i % n == n - 1:
            x += 1
            y -= n + delta * i
    return coords


def display_graph(G, coords=None):
    fig = plt.figure()
    nx.draw(G, with_labels=True, font_weight='bold', pos=coords)
    fig.savefig("9.png")
    # plt.show()


def set_colour(G, n, k):
    '''
    Give a different colour to each partition
    '''
    for u in range(n * k):
        col = u // n
        G.nodes[u]['colour'] = set([col])


def set_label(G, n, k):
    '''
    Give a label to each node. These can be merged later on
    '''
    for u in range(n * k):
        G.nodes[u]['label'] = set([u])


def gen_kpart(k, n, p):
    '''
    k -> minimum number of independent sets
    n -> number of nodes in each
    p -> probability of having an edge between two vertices from different independent sets

    1) Create a k-clique
    2) Add edges across different partitions with probability p

    Return -> networkx graph G following the above criteria
    '''

    N = k * n

    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    set_colour(G, n, k)
    set_label(G, n, k)
    # Create a k-clique
    G.add_edges_from([(i, j) for i in range(0, N, n)
                      for j in range(i + n, N, n)])
    # Add edges with probability p
    for i in range(0, N, n):
        for j in range(i, i + n):
            for l in range(i + n, N):
                if j % n == 0 and l % n == 0:
                    continue
                x = random.uniform(0, 1)
                if x < p:
                    G.add_edge(j, l)

    # print edges of a node
    # for i in range(N):
    #     print(i, G.neighbors(i))

    # print(G.nodes.data())
    # Set node coordinates
    coords_dict = set_coords(k, n)
    return G, coords_dict


def combinations(n, r):
    return len(list(itertools.combinations(range(n), r)))


def fixed_kpart(k, n, p):
    '''
    k -> minimum number of independent sets
    n -> number of nodes in each
    p -> probability of having an edge between two vertices from different independent sets

    1) Pick one node at a time from each cluster
    2) Add one edge to every cluster if not already present

    3) if fill_more_edges is True, add remaining edges randomly

    Return -> networkx graph G following the above criteria
    '''

    N = k * n
    total_edges = p*(combinations(N, 2) - k*combinations(n, 2))//2

    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    set_colour(G, n, k)
    set_label(G, n, k)

    node_dict = {}

    # choose a cluster, choose a node and make edges to all other cluster
    print("Total edges: ", total_edges)
    for i in range(k):
        for j in range(i*n, i*n+n):
            count = 0
            for l in range(k):
                if l == i or l in node_dict.get(j, []):
                    continue
                u = j
                v = np.random.randint(n) + l*n
                G.add_edge(u, v)
                node_dict[u] = node_dict.get(u, []) + [l]
                node_dict[v] = node_dict.get(v, []) + [i]
                # print(u,v)
                count += 1
                total_edges -= 1
            # print(f"{count} edges added to {j}")

    # Add remaining edges
    fill_more_edges = True

    if fill_more_edges:
        print(f"Adding {min(total_edges,0)} remaining edges")
        while total_edges>0:
            u = np.random.randint(N)
            choices_avalable = list(range(u//n*n)) + list(range((u//n+1)*n, N))
            v = np.random.choice(choices_avalable)
            if v in G.neighbors(u):
                continue
            G.add_edge(u, v)
            # print(u,v, total_edges)
            total_edges -= 1

    print("Setting node coordinates")
    coords_dict = set_coords(k, n)
    return G, coords_dict


if __name__ == '__main__':
    G, coords = fixed_kpart(5, 5, 0.4)
    display_graph(G, coords)

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
    nx.draw(G, with_labels=True, font_weight='bold', pos=coords)
    plt.show()

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
    G.add_edges_from([(i, j) for i in range(0, N, n) for j in range(i + n, N, n)])
    # Add edges with probability p
    for i in range(0, N, n):
        for j in range(i, i + n):
            for l in range(i + n, N):
                if j % n == 0 and l % n == 0:
                    continue
                x = random.uniform(0, 1)
                if x < p:
                    G.add_edge(j, l)

    # print(G.nodes.data())
    # Set node coordinates
    coords_dict = set_coords(k, n)
    return G, coords_dict
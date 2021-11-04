import random


def inverse_map(G, N):
    '''
    :G -> Processed Graph
    :N -> Original number of vertices

    return: A mapping from each node in the initial graph to where it has ended up finally
    '''
    inverse = [0 for i in range(N)]
    for u in G.nodes():
        lbls = list(G.nodes[u]['label'])
        for lbl in lbls:
            inverse[lbl] = u
    return inverse


def num_nodes(G):
    '''
    G -> Processed Graph

    return: Number of nodes in G
    '''
    return len(G.nodes())


def pairwise_accuracy(G, G_init, n_samples=1000):
    '''
    :G -> Processed Graph
    :G_init -> Original Graph
    :n_samples -> Number of pairwise samples to be taken
    
    return: Fraction of pairs of nodes that have been assigned correctly
    '''
    N = len(G_init.nodes())
    inv_mp = inverse_map(G, N)
    correct = 0
    for _ in range(n_samples):
        u, v = random.sample(range(N), 2)

        # Ground truth, 0 if belonging to different partitions, 1 if belonging to the same
        gt = G_init.nodes()[u]['colour'] == G_init.nodes()[v]['colour']

        # Finally, 0 if belonging to different vertices, 1 if merged into one vertex
        actual = inv_mp[u] == inv_mp[v]

        if actual == gt:
            correct += 1
    
    return correct / n_samples
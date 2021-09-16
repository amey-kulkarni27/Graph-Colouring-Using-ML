import random
import networkx as nx

def vertex_pair(G, N):
    '''
    G -> Graph on which we perform the operation
    N -> Total number of vertices

    Return: Pair of vertices that do not share an edge between them
    '''
    while True:
        u, v = random.sample(range(N), 2)
        if ~G.has_edge(u, v):
            break
    return (u, v)

def vertex_pair_opt(G):
    '''
    G -> Graph on which we perform the operation

    Return: Pair of vertices that do not share an edge between them
    '''
    Gdash = nx.complement(G)
    if len(Gdash.edges()) == 0:
        print("Exception: Complete graph given")
        return None
    edges = random.sample(Gdash.edges(), 1)
    return edges[0]

def operations(G, merge, nodes):
    '''
    G -> Graph on which we perform the operation
    merge -> Boolean variable
    - True: Perform the "Merge" operation, combine two nodes
    - False: Perform the "Add Edge" operation, add an edge between two nodes
    nodes -> Tuple containing the two nodes on which we perform the given operation

    Return: Modified graph after operation
    '''
    if merge:
        G = nx.contracted_nodes(G, nodes[0], nodes[1])
    else:
        G.add_edge(nodes[0], nodes[1])
    return G


def get_label(n, merge, nodes):
    '''
    n -> Number of nodes in each of the independent sets
    merge -> Boolean variable
    - True: Perform the "Merge" operation, combine two nodes
    - False: Perform the "Add Edge" operation, add an edge between two nodes
    nodes -> Tuple containing the two nodes on which we perform the given operation

    Return: 0/1 label, denoting whether the operation was optimal or not
    0: if non-independent set vertices merged, or added edge between independent
    1: if independent merged, or added edge between non-independent set vertices
    '''

    u, v = nodes
    if merge:
        if u // n == v // n:
            return 1
        else:
            return 0
    else:
        if u // n == v // n:
            return 0
        else:
            return 1
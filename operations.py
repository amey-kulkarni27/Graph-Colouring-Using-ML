import random
import networkx as nx
from networkx.classes.function import non_edges

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
    return (min(u, v), max(u, v))

def vertex_pair_opt(G):
    '''
    G -> Graph on which we perform the operation

    Return: Pair of vertices that do not share an edge between them
    If G is a complete graph, return False
    '''
    Gdash = nx.complement(G)
    if len(Gdash.edges()) == 0:
        # print("Exception: Complete graph given")
        return False
    edges = random.sample(Gdash.edges(), 1)
    n_edge = (min(edges[0]), max(edges[0]))
    return n_edge

def vertex_pair_non_edge(G):
    '''
    :param G -> Graph on which we perform the operation

    Return: Pair of vertices that do not share an edge between them
    If G is a complete graph, return False
    '''
    non_edges = list(nx.non_edges(G))
    if(len(non_edges) == 0):
        return False
    n_edges = random.sample(non_edges, 1)
    n_edge = (min(n_edges[0]), max(n_edges[0]))
    return n_edge

def vertex_pair_coin_toss(G, op):
    '''
    :param G -> Graph on which we perform the operation
    :param op -> Merge if 0, Add edge if 1

    Return: Pair of vertices that do not share an edge between them
    If G is a complete graph, return False
    '''
    non_edges = list(nx.non_edges(G))
    if(len(non_edges) == 0):
        return False
    random.shuffle(non_edges)
    for n_edge in non_edges:
        u, v = n_edge
        u_c = G.nodes[u]['colour']
        v_c = G.nodes[v]['colour']
        if op == 0 and u_c == v_c:
            return n_edge
        elif op == 1 and u_c != v_c:
            return n_edge
    # If no non-edge of the given type exists
    return non_edges[0]

def operations(G, action, nodes):
    '''
    G -> Graph on which we perform the operation
    merge -> Boolean variable
    - 0: Perform the "Merge" operation, combine two nodes
    - 1: Perform the "Add Edge" operation, add an edge between two nodes
    nodes -> Tuple containing the two nodes on which we perform the given operation

    Return: Modified graph after operation
    '''
    if action==0:
        assert(nodes[0] < nodes[1])
        # Larger node gets merged into the smaller
        G.nodes[nodes[0]]['label'].update(G.nodes[nodes[1]]['label'])
        G.nodes[nodes[0]]['colour'].update(G.nodes[nodes[1]]['colour'])
        G = nx.contracted_nodes(G, nodes[0], nodes[1])
        # print(G.nodes[nodes[0]])
        # print()
    else:
        G.add_edge(nodes[0], nodes[1])
    return G


# def get_label(n, merge, nodes):
#     '''
#     n -> Number of nodes in each of the independent sets
#     merge -> Boolean variable
#     - True: Perform the "Merge" operation, combine two nodes
#     - False: Perform the "Add Edge" operation, add an edge between two nodes
#     nodes -> Tuple containing the two nodes on which we perform the given operation

#     Return: 0/1 label, denoting whether the operation was optimal or not
#     0: if non-independent set vertices merged, or added edge between independent
#     1: if independent merged, or added edge between non-independent set vertices
#     '''

#     u, v = nodes
#     if merge:
#         if u // n == v // n:
#             return 1
#         else:
#             return 0
#     else:
#         if u // n == v // n:
#             return 0
#         else:
#             return 1

def get_action(G, nodes):
    '''
    G -> Graph on which we perform the operation
    nodes -> Tuple containing the two nodes on which we perform the given operation

    Return: 0/1 action, denoting whether the operation was merge or join
    0: if vertices belonging to independent set merged
    1: if added edge between non-independent set vertices
    '''
    u, v = nodes
    if G.nodes[u]['colour'] == G.nodes[v]['colour']:
        return 0
    else:
        return 1


'''
a1 a2 a3
1 2 3
1 2 3
1 2 3
1 2 3
1 2 3
'''
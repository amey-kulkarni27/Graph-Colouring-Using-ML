def feature_vector(G):
    '''
    G -> Input Graph

    Returns a feature vector comprising the following features for each node
    1) Number of neighbours
    2) Is it a leaf?
    
    Return -> vec: N x 2 sized feature vector
    '''
    N = len(G.nodes)
    f = 2 # Only 2 features currently
    vec = [[0, 0] for i in range(N)]
    for i in range(N):
        vec[i][0] = len([i for i in G.neighbors(i)])
        vec[i][1] = (vec[i][0] == 1)
    return vec
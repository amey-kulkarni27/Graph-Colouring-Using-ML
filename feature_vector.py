import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg.decomp import eig
from node2vec import Node2Vec

def manual_fv(G):
    '''
    G -> Input Graph

    Returns a feature vector comprising the following features for each node
    1) Number of neighbours
    2) Is it a leaf?
    
    Return -> vec: N x 2 sized feature vector
    '''
    N = len(G.nodes)
    f = 2 # Only 2 features currently
    fv = [[0, 0] for i in range(N)]
    for i in range(N):
        fv[i][0] = len([i for i in G.neighbors(i)])
        fv[i][1] = (fv[i][0] == 1)
    return fv

def topk_fv(G, k):
    '''
    G -> Input Graph
    k -> Top number of eigenvector rows to be chosen

    Return -> N x k matrix, every node has a feature vector of dimension k
    '''
    N = len(G.nodes)
    A = adjacency_matrix(G).todense()
    A = np.array(A)
    A = A.astype(float)
    _, eigvecs = eigs(A, k=k) # Top k eigenvectors
    eigvecs = np.real(eigvecs)
    fv = eigvecs
    # fv = list(map(list, zip(*eigvecs)))
    # print(fv.shape)
    return fv

def node2vec_fv(G, k=32):
    '''
    G -> Input Graph

    Return -> N feature vectors
    '''
    node2vec = Node2Vec(G, dimensions=k, walk_length=20, num_walks=100, workers=4)  # Use temp_folder for big graphs
    # # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    fv = model.wv[[i for i in range(len(G.nodes))]]
    return fv

def feature_vector(G, method='topk', k=3):
    '''
    G -> Input Graph
    method -> Method for calculating feature vector, options include:
              1. 'custom': manual features
              2. 'topk': Top k eigenvectors
              3. 'node2vec': Node2vec assigns features

    Return -> N feature vectors
    '''
    N = len(G.nodes)
    if method == 'custom':
        fv = manual_fv(G)
    elif method == 'topk':
        fv = topk_fv(G, k=k)
    elif method == 'node2vec':
        fv = node2vec_fv(G, k)
    
    return fv
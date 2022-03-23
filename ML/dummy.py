from node2vec import Node2Vec
from generate_kpart import gen_kpart
import networkx as nx

k = 3
n = 4
p = 0.5
G, _ = gen_kpart(k, n, p)
# G = nx.fast_gnp_random_graph(n=100, p=0.5)
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
# # Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
print((model.wv[[i for i in range(12)]]).shape)
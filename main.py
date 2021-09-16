from generate_kpart import gen_kpart
from feature_vector import feature_vector

num_graphs = 5
k = 3
n = 4
p = 0.4
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
for G in G_list:
    vec = feature_vector(G)
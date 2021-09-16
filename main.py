from generate_kpart import gen_kpart
from feature_vector import feature_vector

k = 3
n = 4
p = 0.4
G = gen_kpart(k, n, p)
vec = feature_vector(G)
from generate_kpart import gen_kpart, display_graph, fixed_kpart
from operations import get_action, operations, vertex_pair_opt, vertex_pair_non_edge
import random
import copy

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


num_graphs = 5
k = 2
n = 3 # Number of nodes in a single partition
p = 0.0
error = 0.05
errors = [0.1 * i for i in range(11)]
probs = [0.1 * i for i in range(11)]

change_err = True # either vary error or vary graph density
y = [] # no of colours
if change_err:
    x = errors
    G, coords = gen_kpart(k, n, p)
else:
    x = probs

for i in range(len(x)):
    if change_err:
        error = x[i]
        Gdash = copy.deepcopy(G)
        if i == 10:
            display_graph(Gdash, coords)
    else:
        p = x[i]
        Gdash, _ = gen_kpart(k, n, p)
    while (vertex_pair_non_edge(Gdash)) != False:
        nodes = vertex_pair_non_edge(Gdash)
        action = get_action(Gdash, nodes)
        r = random.uniform(0, 1)
        if r < error:
            action ^= 1
        Gdash = operations(Gdash, action, nodes)
        if i == 10:
            display_graph(Gdash, coords)
    cols = len(Gdash.nodes)
    y.append(cols)
print(y)
fig = plt.figure()
plt.plot(x, y, label="Colours needed")
plt.xlabel("error/p")
plt.ylabel("Colours needed")
# plt.xticks([i for i in range(1, 21, 2)])
plt.title("Changing error/p, n=10, k=10")
# plt.legend()
fig.savefig('Images/app_ratio.png')
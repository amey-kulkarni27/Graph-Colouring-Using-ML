from numpy.lib.function_base import disp
from generate_kpart import gen_kpart, display_graph, fixed_kpart
from metrics import num_nodes, pairwise_accuracy
from train import train_static
from test import test_static
import numpy as np
from classifier import classifier
import copy
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

start = timer()
num_graphs = 5
k = 10
n = 10 # Number of nodes in a single partition
p = 0.6
delta = 5
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
# G_list = [fixed_kpart(k, n, p) for i in range(num_graphs)]
G_train_list, G_test_list = train_test_split(G_list, test_size=0.2)
X = []
y = []
ctr = 0
for G, coords in G_train_list:
    Gdash = copy.deepcopy(G)
    train_static(Gdash, coords, X, y, n, k-1, "topk", equal_cl=True, sub=True)
    # if ctr == 0:
    #     print(X)
    ctr = 1

# print(X)
# print(y)
X = np.array(X)
y = np.array(y)
t2 = timer()
clf = classifier("logistic", 0)
clf.fit(X, y)
t3 = timer()

for G, coords in G_test_list:
    for trials in range(delta):
        test_static(G, coords, clf, k-1, "topk", 0.5, sub=True, probvals=True)
from generate_kpart import gen_kpart, display_graph
from train import train
from test import test
import numpy as np
from sklearn.linear_model import LogisticRegression
import copy
import csv

num_graphs = 5
k = 5
n = 5 # Number of nodes in a single partition
p = 0.25
delta = 10
G_train_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
G_test_list = G_train_list[:]
X = []
y = []
update_interval = 5
for G, coords in G_train_list[:int(0.8*num_graphs)]:
    Gdash = copy.deepcopy(G)
    train(Gdash, coords, X, y, n, k, update_interval)
    
X = np.array(X)
y = np.array(y)
clf = LogisticRegression(random_state=0).fit(X, y)

for G, coords in G_test_list[int(0.8*num_graphs):]:
    new_nodes = []
    for trials in range(delta):
        new_nodes.append(test(G, coords, clf, n, k, update_interval))
    print(new_nodes)

yhat = clf.predict(X)
print(sum(y == yhat) / len(y))
# print(sum(y), sum(yhat))
# print(len(y))
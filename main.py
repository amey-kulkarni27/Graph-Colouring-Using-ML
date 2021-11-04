from numpy.lib.function_base import disp
from generate_kpart import gen_kpart, display_graph
from metrics import num_nodes, pairwise_accuracy
from train import train
from test import test
import numpy as np
from sklearn.linear_model import LogisticRegression
import copy
import csv
from timeit import default_timer as timer

start = timer()
num_graphs = 5
k = 10
n = 10 # Number of nodes in a single partition
p = 0.2
delta = 2
G_train_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
G_test_list = G_train_list[:]
X = []
y = []
update_interval = 2
for G, coords in G_train_list[:int(0.8*num_graphs)]:
    t1 = timer()
    Gdash = copy.deepcopy(G)
    t2 = timer()
    print("Copy: ", round(t2 - t1, 2))
    train(Gdash, coords, X, y, n, k//2, update_interval, "topk")
    t3 = timer()
    print("Train time: ", round(t3 - t2, 2))
    print()

X = np.array(X)
y = np.array(y)
clf = LogisticRegression(random_state=0).fit(X, y)

for G, coords in G_test_list[int(0.8*num_graphs):]:
    new_nodes = []
    pwise_acc = []
    for trials in range(delta):
        G_final = test(G, coords, clf, n, k//2, update_interval, "topk")
        new_num = num_nodes(G_final)
    #     acc = pairwise_accuracy(G_final, G, 1000)
        new_nodes.append(new_num)
    #     pwise_acc.append(acc)
    print(new_nodes)
    # print(pwise_acc)

yhat = clf.predict(X)
print(sum(y == yhat) / len(y))
# print(sum(y), sum(yhat))
# print(len(y))
end = timer()
print(end - start)
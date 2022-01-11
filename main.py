from numpy.lib.function_base import disp
from generate_kpart import gen_kpart, display_graph
from metrics import num_nodes, pairwise_accuracy
from train import train
from test import test
import numpy as np
from classifier import classifier
import copy
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

start = timer()
num_graphs = 5
k = 10
n = 10 # Number of nodes in a single partition
p = 0.7
delta = 5
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
G_train_list, G_test_list = train_test_split(G_list, test_size=0.2)
X = []
y = []
update_interval = 1
for G, coords in G_train_list:
    t1 = timer()
    Gdash = copy.deepcopy(G)
    t2 = timer()
    # print("Copy: ", round(t2 - t1, 2))
    train(Gdash, coords, X, y, n, k//2, update_interval, "topk", coin_toss=False)
    t3 = timer()
    # print("Train time: ", round(t3 - t2, 2))
    # print()

X = np.array(X)
y = np.array(y)
t2 = timer()
clf = classifier("xgb", 0)
clf.fit(X, y)
t3 = timer()
print("LR fit time: ", round(t3 - t2, 2))

for G, coords in G_test_list:
    new_nodes = []
    pwise_acc = []
    for trials in range(delta):
        G_final, steps = test(G, coords, clf, n, k//2, update_interval, "topk", 0.7)
        new_num = num_nodes(G_final)
        acc = pairwise_accuracy(G_final, G, 1000)
        new_nodes.append(new_num)
        pwise_acc.append(acc)
    print(new_nodes)
    # print(pwise_acc)

# yhat = clf.predict(X)
# yhat = lr.predict(X)
# print(sum(y), sum(yhat))
# print(len(y))
end = timer()
print(end - start)
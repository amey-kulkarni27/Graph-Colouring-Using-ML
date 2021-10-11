from generate_kpart import gen_kpart, display_graph
from train import train
from test import test
import numpy as np
from sklearn.linear_model import LogisticRegression
import copy


num_graphs = 1
k = 3
n = 5 # Number of nodes in a single partition
p = 0.4
G_train_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
G_test_list = G_train_list[:]
X = []
y = []
update_interval = 1
for G, coords in G_train_list:
    Gdash = copy.deepcopy(G)
    train(Gdash, coords, X, y, n, k, update_interval)
    
X = np.array(X)
y = np.array(y)
clf = LogisticRegression(random_state=0).fit(X, y)

for G, coords in G_test_list:
    test(G, coords, clf, n, k, update_interval)

yhat = clf.predict(X)
print(sum(y == yhat) / len(y))
print(sum(y), sum(yhat))
print(len(y))
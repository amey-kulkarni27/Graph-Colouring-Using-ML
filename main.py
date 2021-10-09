from generate_kpart import gen_kpart, display_graph
from train import train
import numpy as np
from sklearn.linear_model import LogisticRegression


num_graphs = 1
k = 5
n = 100 # Number of nodes in a single partition
p = 0.5
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
X = []
y = []
update_interval = 5
for G, coords in G_list:
    train(G, coords, X, y, n, k, update_interval) 

X = np.array(X)
y = np.array(y)
clf = LogisticRegression(random_state=0).fit(X, y)
yhat = clf.predict(X)
print(sum(y == yhat) / len(y))
print(sum(y), sum(yhat))
print(len(y), len(yhat))
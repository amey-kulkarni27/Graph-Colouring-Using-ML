from generate_kpart import gen_kpart, display_graph
from train import train
import numpy as np
from sklearn.linear_model import LogisticRegression


num_graphs = 5
k = 5
n = 10 # Number of nodes in a single partition
p = 0.5
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
X = []
y = []
for G, coords in G_list:
    train(G, coords, X, y, n) 

X = np.array(X)
y = np.array(y)
print(len(y))
clf = LogisticRegression(random_state=0).fit(X, y)
yhat = clf.predict(X)
print(sum(y == yhat) / num_graphs)
print(sum(yhat))
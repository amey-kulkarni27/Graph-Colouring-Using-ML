from generate_kpart import gen_kpart, display_graph
from train import train
import numpy as np
from sklearn.linear_model import LogisticRegression


num_graphs = 3
k = 3
n = 4 # Number of nodes in a single partition
p = 0.4
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
X = []
y = []
for G, coords in G_list:
    # display_graph(G, coords)
   train(G, X, y, n) 
    # display_graph(G, coords)

X = np.array(X)
y = np.array(y)
print(X)
print(y)
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.predict(X))
from numpy.lib.function_base import disp
from generate_kpart import gen_kpart, display_graph, fixed_kpart
from metrics import num_nodes, pairwise_accuracy
from train import train, train_static
from test import test
import numpy as np
from classifier import classifier
import copy
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from matplotlib import style
import matplotlib.pyplot as plt
from tqdm import tqdm
style.use('ggplot')


start = timer()
num_graphs = 20
k = 10
n = 10 # Number of nodes in a single partition
p = 0.5
delta = 5
G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
# G_list = [fixed_kpart(k, n, p) for i in range(num_graphs)]
G_train_list, G_test_list = train_test_split(G_list, test_size=0.2)
X = []
y = []
update_interval = 1
for G, coords in tqdm(G_train_list):
    t1 = timer()
    Gdash = copy.deepcopy(G)
    t2 = timer()
    # print("Copy: ", round(t2 - t1, 2))
    train(Gdash, coords, X, y, n, k-1, update_interval, "topksv", coin_toss=True, sub=False)
    t3 = timer()
    # print("Train time: ", round(t3 - t2, 2))
    # print()

X = np.array(X)
y = np.array(y)
t2 = timer()
clf = classifier("logistic", 0)
clf.fit(X, y)
t3 = timer()
# print("LR fit time: ", round(t3 - t2, 2))

colours = [0 for i in range(101)]
for G, coords in tqdm(G_test_list):
    new_nodes = []
    pwise_acc = []
    # threshs = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshs = [0.5, 0.5, 0.5, 0.5, 0.5]
    for trials in range(delta):
        G_final, steps = test(G, coords, clf, n, k-1, update_interval, "topksv", threshs[trials], sub=False)
        new_num = num_nodes(G_final)
        acc = pairwise_accuracy(G_final, G, 1000)
        new_nodes.append(new_num)
        colours[new_num] += 1
        pwise_acc.append(acc)
    print(new_nodes)
    # print(pwise_acc)

# yhat = clf.predict(X)
# yhat = lr.predict(X)
# print(sum(y), sum(yhat))
# print(len(y))
end = timer()
print(end - start)

plt.bar([i for i in range(101)], colours, )
plt.xlabel("Final Chromatic Number")
plt.ylabel("Frequency")
plt.title("Concatenate Feature Vectors")
plt.savefig("bar_concat.png")
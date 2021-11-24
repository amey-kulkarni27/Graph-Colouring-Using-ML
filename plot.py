from numpy.lib.function_base import disp
from generate_kpart import gen_kpart, display_graph
from metrics import num_nodes, pairwise_accuracy
from train import train
from test import test
import numpy as np
from sklearn.linear_model import LogisticRegression
from classifier import Logistic
import copy
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

num_graphs = 5
k = 10
n = 10
p = 0.7
delta = 1
update_interval = 1
ratio = []
times = []
accuracy = []
iter_list = [i for i in range(1, 21)]
for update_interval in iter_list:
    start = timer()
    G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
    G_train_list, G_test_list = train_test_split(G_list, test_size=0.2)
    X = []
    y = []
    for G, coords in G_train_list:
        Gdash = copy.deepcopy(G)
        train(Gdash, coords, X, y, n, k//2, update_interval, "topk", coin_toss=False)

    X = np.array(X)
    y = np.array(y)
    # lr = LogisticRegression(random_state=0)
    # clf = lr.fit(X, y)
    lr = Logistic(rand_state=0)
    lr.fit(X, y)
    t3 = timer()

    for G, coords in G_test_list:
        new_nodes = []
        pwise_acc = []
        for trials in range(delta):
            G_final = test(G, coords, lr, n, k//2, update_interval, "topk", 0.6)
            new_num = num_nodes(G_final)
            # acc = pairwise_accuracy(G_final, G, 1000)
            new_nodes.append(new_num)
            # pwise_acc.append(acc)
        ratio.append(min(new_nodes) / k)
        # accuracye.append(max(pwise_acc))

    end = timer()
    tot_time = end - start
    times.append(tot_time)

fig = plt.figure()
plt.plot(iter_list, ratio, label="Approx Ratio")
plt.plot(iter_list, times, label="Time Taken (secs)")
plt.xlabel("Update Interval For Feature Vector")
plt.ylabel("Approx Ratio / Time Req (secs)")
plt.xticks([i for i in range(1, 21, 4)])
plt.title("Changing Update Intervals, k=10, n=10")
plt.legend()
fig.savefig('Images/upd_intervs.png')
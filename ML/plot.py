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
p = 0.65
delta = 1
update_interval = 1
ratio = []
times = []
accuracy = []
steps = []
# iter_list = [i for i in range(2, 11)]
iter_list = [10]
for n in iter_list:
    start = timer()
    G_list = [gen_kpart(k, n, p) for i in range(num_graphs)]
    G_train_list, G_test_list = train_test_split(G_list, test_size=0.2)
    X = []
    y = []
    for G, coords in G_train_list:
        Gdash = copy.deepcopy(G)
        train(Gdash, coords, X, y, n, k//2, update_interval, "topk", coin_toss=True)

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
            G_final, step = test(G, coords, lr, n, k//2, update_interval, "topk", 0.5)
            new_num = num_nodes(G_final)
            acc = pairwise_accuracy(G_final, G, 1000)
            new_nodes.append(new_num)
            pwise_acc.append(round(acc, 2))
            steps.append(step)
        ratio.append(min(new_nodes) / k)
        accuracy.append(max(pwise_acc))

    end = timer()
    tot_time = end - start
    times.append(tot_time)
print(ratio, accuracy, times)

# fig = plt.figure()
# plt.plot(iter_list, ratio, label="Approx Ratio")
# plt.xlabel("Thresholds")
# plt.ylabel("Approx Ratio")
# # plt.xticks([i for i in range(1, 21, 2)])
# plt.title("Changing Thresholds, n=10, k=10")
# # plt.legend()
# fig.savefig('Images/x_approx.png')

# fig2 = plt.figure()
# plt.plot(iter_list, accuracy, label="Accuracy")
# plt.xlabel("Thresholds")
# plt.ylabel("Pairwise Accuracy")
# # plt.xticks([i for i in range(1, 21, 2)])
# plt.title("Changing Thresholds, n=10, k=10")
# # plt.legend()
# fig2.savefig('Images/x_acc.png')

# fig3 = plt.figure()
# plt.plot(iter_list, times, label="Time")
# plt.xlabel("Thresholds")
# plt.ylabel("Time Taken")
# # plt.xticks([i for i in range(1, 21, 2)])
# plt.title("Changing Thresholds, n=10, k=10")
# # plt.legend()
# fig3.savefig('Images/x_time.png')
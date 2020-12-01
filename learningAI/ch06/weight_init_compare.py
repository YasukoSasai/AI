# coding: utf-8
import sys
sys.path.append('/Users/eb604/deep-learning-from-scratch-master')

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0] #60000枚
batch_size = 128 #128枚
max_iterations = 2000


# 1:実験の設定==========
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {} #各初期値に対するnetworkを格納するためのディクショナリ
train_loss = {} #各NNに対する誤差のディクショナリ
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []


# 2:訓練の開始==========
for i in range(max_iterations): #2000回繰り返す
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        # print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            # print(key + ":" + str(loss))


# 3.グラフの描画==========
# markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
# x = np.arange(max_iterations)
# for key in weight_init_types.keys():
#     plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
# plt.xlabel("iterations")
# plt.ylabel("loss")
# plt.ylim(0, 2.5)
# plt.legend()
# plt.show()
# print(networks)

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
# plt.show()


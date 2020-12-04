# coding: utf-8
import sys
sys.path.append('../../../deep-learning-from-scratch')
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *
import numpy as np


# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0] #60000枚
batch_size = 128 #128枚ずつ
max_iterations = 2000 #学習の繰り返し2000回
iter_per_epoch = max(train_size / batch_size, 1) #60000/128 = エポック毎の学習回数。何回訓練させているか。


# 1:最適化アルゴリズムの設定==========
optimizers = {}

optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []    

train_loss_list = [] #学習ごとの損失関数を格納するためのリスト
train_acc_list = [] #学習における正確率
test_acc_list = [] 


# 2:訓練の開始==========
for i in range(max_iterations): #2000回まで
    batch_mask = np.random.choice(train_size, batch_size) #60000枚から128枚
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys(): #パラメータの最適化
        grads = networks[key].gradient(x_batch, t_batch) #network.gradientから得た勾配
        optimizers[key].update(networks[key].params, grads) #gradsを入れてoptimizer.updateを実行
    
        loss = networks[key].loss(x_batch, t_batch) #network.lossを実行
        train_loss[key].append(loss) #train_lossにlossを追加
    
    if i % 100 == 0: #学習100回(128枚*100)ごとに精度を表示
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3.誤差グラフの描画==========
# markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

# 3. 認識精度グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


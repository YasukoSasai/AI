import os
import sys
sys.path.append('/Users/eb604/deep-learning-from-scratch-master')
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True)

#過学習再現のためデータを削減
x_train = x_train[:300]
t_train = t_train[:300]

#Dropoutの有無、割合の設定
use_dropout = False #DropoutなしのときはFalse
dropout_ratio = 0.15

#networkインスタンス生成
network = MultiLayerNetExtend(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100], 
                              output_size = 10, use_dropout=use_dropout, dropout_ration = dropout_ratio)
#NNの訓練を行うクラスのインスタンス生成
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=301, 
                  mini_batch_size = 100, optimizer = 'sgd', optimizer_param = {'lr':0.01}, 
                  verbose = True)
#Trainerのtrainを実行
trainer.train()
#trainerのそれぞれリストを代入
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# class Dropout:
#     def __init__(self, dropout_ratio = 0.5):
#         self.dropout_ratio = dropout_ratio
#         self.mask = None
    
#     def forward(self, x, train_flg = True):
#       if train_flg:
#           #xと同じ形状のランダムなデータの配列を生成し、値がdrop_outよりも大きい場合その要素をTrueとする
#           self.mask = np.random.rand(*x.shape) > self.dropout_ratio
          
#           return x * self.mask 
#       else:
#           return x * (1.0 - self.dropout_ratio)

#     def backward(self, dout):
#         #残ったニューロンと勾配を駆ける
#         return dout * self.mask

# グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
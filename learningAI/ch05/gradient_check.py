import sys
sys.path.append('../')
import numpy as np
from dataset.mnist import load_mnist
from two_layers_net import TwoLayerNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
#TwoLayersNetのインスタンス生成
network = TwoLayerNet(input_size=784, hidden_size=50, output_size =10)

#0~3番目までのデータを格納
x_batch = x_train[:3]
t_batch = t_train[:3]
# print(x_train.shape) #60000枚＊784ピクセル
# print(x_train[0].shape) #784ピクセル
# print(x_train[0]) #784ピクセル
# print(t_train.shape) #60000枚＊10
# print(t_train[0].shape) #10個の正解データ
# print(t_train[0]) #10個の正解データ

#勾配
grad_numerical = network.numerical_gradient(x_batch, t_batch)
#誤差逆伝播
grad_backprop = network.gradient(x_batch, t_batch)

#各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))


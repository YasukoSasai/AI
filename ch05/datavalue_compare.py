#２層ニューラルネットワークのクラス
import sys
sys.path.append('../')

from dataset.mnist import load_mnist
# from two_layers_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print("normlize, one_hot_labelのとき")
print(x_train[0])
print(t_train[0])


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print("flattenのとき")
print(x_train[0])
print(t_train[0])
import sys, os
sys.path.append('/Users/eb604/deep-learning-from-scratch-master')
import numpy as np
from dataset.mnist import load_mnist
from two_layers_net import TwoLayerNet
import matplotlib.pylab as plt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0] #60000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) #1エポックあたりの学習イテレーション回数　60000 / 100 = 600 (1の意味は何？？？)
# print(iter_per_epoch)


for i in range(iters_num): #10000回繰り返す
    batch_mask = np.random.choice(train_size, batch_size) #60000枚データの中から100枚データをランダムに選び、
    x_batch = x_train[batch_mask] #訓練データの正解データのバッチを作る
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch) #バッチデータを入れて誤差逆伝播を行う

    for key in ('W1', 'b1', 'W2', 'b2'): #'W1'更新・loss追加 → 'W2'更新・loss追加 → 'b1'更新・loss追加 → ’b2’更新・loss追加
        network.params[key] -= learning_rate * grad[key] #network.params['key']からその勾配と学習率を掛けたものを引いて更新する
        
        loss = network.loss(x_batch, t_batch) #
        train_loss_list.append(loss) #最終的なlistのlenは40000(4*10000)        
        # print("[b1]", network.params['b1'][0]) #4回ごとに更新されていた

    if i % iter_per_epoch == 0: #i / 600の余りが0ならばlistに追加(1エポックごと。16回)
      
      train_acc = network.accuracy(x_train, t_train)
      test_acc = network.accuracy(x_test, t_test)
      train_acc_list.append(train_acc)
      test_acc_list.append(test_acc)
      # print("train_acc, test_acc", train_acc, test_acc)

#--------------- 誤差の推移 ------------------------------
# print("train_loss_list", train_loss_list)
plt.plot(train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show() #しかしここで得られた損失関数はミニバッチに対する損失関数(100枚)

#--------------- 精度の推移を表示してみた ---------------
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list)) #0~len(train_acc_list)を１間隔で
# print(len(train_acc_list))
# print(train_acc_list)
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
# plt.show()

# print(len(train_loss_list))
# print(len(train_acc_list))
# print(len(test_acc_list))
# print(train_loss_list[0])

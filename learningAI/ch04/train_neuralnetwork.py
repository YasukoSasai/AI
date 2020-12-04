#========= 学習=============(5.7.4)
#２層(3層)ニューラルネットワークのクラス
#精度の確認。学習できているか。
import sys
sys.path.append('../../../deep-learning-from-scratch')
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layers_net import TwoLayerNet

# *********** ミニバッチ学習の実装 ***************
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #NNのインスタンス生成

#====== ハイパーパラメータ ======
iters_num = 10000 #勾配によるパラメータ更新回数(iterarion = 繰り返し)
train_size = x_train.shape[0] #訓練データのサイズはx_trainデータ形状の0次元の数 = 60000枚　x_train.shape = 60000枚 * 784(28*28)ピクセル
batch_size = 100 #60000枚のうち100枚ごと学習を行う
learning_rate = 0.1 #学習率

train_loss_list = [] #学習ごとの損失関数を格納するためのリスト
train_acc_list = [] #学習における正確率
test_acc_list = [] #テストにおける正確率

#====== 重み・パラメータグラフ表示するとき ======
# w_update_list_0 = []
# w_update_list_4 = []
# w_update_list_9 = []

iter_per_epoch = max(train_size / batch_size, 1) #1エポックあたりの繰り返し数　エポック=訓練データをすべて使い切った回数。60000/100枚 回勾配を行った = １エポック学習を行った。

for i in range (iters_num): #10000回繰り返し
    #===== ミニバッチの取得 =====
    batch_mask = np.random.choice(train_size, batch_size) #train_size枚の中からbatch_size枚ランダムで配列で取り出す
    x_batch = x_train[batch_mask] #bacth_maskのx_trainをx_batchとする
    t_batch = t_train[batch_mask] #同様

    #====== 勾配計算 =====
    #数値微分
    # grad = network.numerical_gradient(x_batch, t_batch)
    #誤差逆伝播法　高速！ 
    grad = network.gradient(x_batch, t_batch) 

    #====== パラメータ更新 ======
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

        #==== 重みとパラメータのグラフを表示したかったら ====
        # weight_per_data = list(network.params['b2'])
        # weight_per_neuron = list(weight_per_data)

        # weight_first_0 = weight_per_neuron[0]
        # weight_first_4 = weight_per_neuron[4]
        # weight_first_9 = weight_per_neuron[9]

        # if key == 'b2':
        #     w_update_list_0.append(weight_first_0)
        #     w_update_list_4.append(weight_first_4)
        #     w_update_list_9.append(weight_first_9)

    #====== 学習経過の記録 ========
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #1エポックごとにテストデータで認識精度を計算　計算に時間がかかるのでざっくりと。
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# ************* グラフによる確認 ******************
# ========== 学習による誤差推移 ==============
# print("train_loss_list", train_loss_list)
plt.plot(train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show() #しかしここで得られた損失関数はミニバッチに対する損失関数(100枚)

# ========== 訓練データとテストデータで認識精度をグラフ化(汎化性能を見るため) ================
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


#========== 損失関数とパラメータのグラフ ================
# print("['W1'].shape",network.params['W1'].shape)
# print("weight_per_data[0]のサイズ",weight_per_data[0].size)
# print("weight_per_neuron[0]のサイズ",weight_per_neuron[0].size)
# print("weight_first",weight_first)
# weight_per_data = list(network.params['W1'])
# weight_per_neuron = list(weight_per_data[0])
# weight_first = weight_per_neuron[0]

# plt.plot(w_update_list_0, train_loss_list, '-', label='train_loss_list')
# plt.plot(w_update_list_4, train_loss_list, '-', label='train_loss_list')
# plt.plot(w_update_list_9, train_loss_list, '-', label='train_loss_list')
# plt.show()




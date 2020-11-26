#-----------------------------------------------------------------------------------------------
#学習アルゴリズムの実装
#２層(3層)ニューラルネットワークのクラス
# from Lesson1.functions import Affine, ReLu, SoftmaxWithLoss
import numpy as np
import sys, os
sys.path.append('/Users/eb604/deep-learning-from-scratch-master')
from dataset.mnist import load_mnist #load_mnist関数の呼び出し
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict
#------------------------------------- NNのクラス -----------------------------------------------------
class TwoLayerNet:
    #-------------- 初期化 ------------------------------------------------------
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): #__init__クラスの初期化メソッド。input_size=784,output_size=10クラス,hiddenは適当な数を設定する
        self.params = {} #ディクショナリ変数。それぞれNumpy配列で格納されている。
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #random.randn = 形状が(input_size*hidden_size)の(0以上1未満の乱数)
        self.params['b1'] = np.zeros(hidden_size) #形状は(hidden_size)で全て0のバイアス。
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #レイヤの作成
        self.layers = OrderedDict() #順番付きのディクショナリ。追加した順番を覚えることができる。→追加した順にレイヤのforward()を呼び出すだけで処理が完了。backwardも同様。AffineレイヤやReLuレイヤが順伝播・逆伝播をかんたんにしてる。
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    #------------------------ 予測 -------------------------------------------------
    def predict(self, x): 
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    #損失関数
    def loss(self, x, t): #x入力データ・t教師データ
        y = self.predict(x) #predictの値をyに代入

        return self.lastLayer.forward(y, t) #交差エントロピー誤差

    def accuracy(self, x, t): #正確率
        y = self.predict(x) #出力yにxのself.predictの値を代入。
        y = np.argmax(y, axis=1) #axis=1　1次元を(列)を軸に最大値を抜き出す。
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0]) #y==tの合計値/入力値の形状の0次元
        return accuracy

    #勾配
    def numerical_gradient(self, x, t): ##ここでなんか時間かかる‥ → [誤差逆伝播法]
        loss_W = lambda W: self.loss(x, t) #W重みを引数としたloss_W関数。入力と正解データを実引数としたlossの値を返却。数値微分

        grads = {} #勾配のディクショナリ変数。pramsと同じようにそれぞれの勾配が格納される。
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) #loss_Wとself.params['W1']を実引数としたnumerical_gradientの値を代入。
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads #全てのパラメータを配列に格納し終わったらgradsで返す。
    #誤差逆伝播法
    def gradient(self, x, t):
        #forward
        self.loss(x,t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        #設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
#---------------------------------------------------------------------------------
#ミニバッチ学習の実装
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #NNのインスタンス生成

#ハイパーパラメータ
iters_num = 10000 #勾配によるパラメータ更新回数(iterarion = 繰り返し)
train_size = x_train.shape[0] #訓練データのサイズはx_trainデータ形状の0次元の数 = 60000枚　x_train.shape = 60000枚 * 784(28*28)ピクセル
batch_size = 100 #60000枚のうち100枚ごと学習を行う
learning_rate = 0.1 #学習率

train_loss_list = [] #学習ごとの損失関数を格納するためのリスト
train_acc_list = [] #学習における正確率
test_acc_list = [] #テストにおける正確率

iter_per_epoch = max(train_size / batch_size, 1) #1エポックあたりの繰り返し数　エポック=訓練データをすべて使い切った回数。60000/100枚 回勾配を行った = １エポック学習を行った。

for i in range (iters_num): #10000回繰り返し
    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size) #train_size枚の中からbatch_size枚ランダムで配列で取り出す
    x_batch = x_train[batch_mask] #bacth_maskのx_trainをx_batchとする
    t_batch = t_train[batch_mask] #同様
    #勾配計算
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) #誤差逆伝播法　高速！ 

    #パラメータ更新
    for key in ('W1', 'b1', 'W2', 'b2'):
       network.params[key] -= learning_rate * grad[key]

    #学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #1エポックごとに認識精度を計算　計算に時間がかかるのでざっくりと確かめる。
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

#----------------------- その他 -----------------------------------------------------------------

#------- 学習による誤差推移 ------------
# print("train_loss_list", train_loss_list)
# plt.plot(train_loss_list)
# plt.xlabel("iteration")
# plt.ylabel("loss")
# plt.show() #しかしここで得られた損失関数はミニバッチに対する損失関数(100枚)

#--------------- 精度の推移を表示してみた ---------------
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()

#---------------- 誤差逆伝播法の勾配確認 ----------------------
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size =10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))





#-----------------------------------------------------------------------------------------------
#学習アルゴリズムの実装
#２層(3層)ニューラルネットワークのクラス
import sys, os
sys.path.append('/Users/eb604/deep-learning-from-scratch-master')
from common.gradient import numerical_gradient
from common.functions import sigmoid, sigmoid_grad, softmax
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

#-------------- 交差エントロピー(バッチ対応) ---------------------
def cross_entropy_error(y, t): #y=NNの出力、t=教師データ
    # print("y.ndim", y.ndim)
    if y.ndim == 1: #yが１次元のとき（データ一つあたりの交差エントロピー誤差を求めるとき）データの形状を整形。
        t = t.reshape(1, t.size) #バッチデータと次元を合わせる。
        y = y.reshape(1, y.size)
        # print(t.shape)
        # print(y.shape)
    batch_size = y.shape[0] # バッチサイズはｙの０次元の数。
    # print("t",t)
    # print("y",y)
    # print("batch_size",batch_size)
    return -np.sum(t * np.log(y)) / batch_size #バッチサイズで割って、一枚あたりの平均交差エントロピー誤差を求める。正解ラベル*

#------------------------------------- NNのクラス -----------------------------------------------------
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): #__init__クラスの初期化メソッド。input_size=784,output_size=10クラス,hiddenは適当な数を設定する
        #重みの初期化
        self.params = {} #ディクショナリ変数。それぞれNumpy配列で格納されている。
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #random.randn = 形状が(input_size*hidden_size)の(0以上1未満の乱数)
        self.params['b1'] = np.zeros(hidden_size) #形状は(hidden_size)で全て0のバイアス。
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #予測
    def predict(self, x): 
        W1, W2 = self.params['W1'], self.params['W2'] #それぞれ代入
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1 #中間層に渡す
        z1 = sigmoid(a1) #中間層の出力
        a2 = np.dot(z1, W2) + b2 #出力層に渡す
        y = softmax(a2) #最終的な出力(出力層の出力)

        return y

    #損失関数
    def loss(self, x, t): #x入力データ・t教師データ
        y = self.predict(x) #predictの値をyに代入

        return cross_entropy_error(y, t) #交差エントロピー誤差

    def accuracy(self, x, t): #正確率
        y = self.predict(x) #出力yにxのself.predictの値を代入。
        y = np.argmax(y, axis=1) #axis=1　1次元を(列)を軸に最大値を抜き出す。
        t = np.argmax(t, axis=1)

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
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0] #100枚
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

#-------------------------ディクショナリ変数の例----------------------------
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)

# x = np.random.rand(100, 784) #ダミーの入力データ100枚分*784ピクセル
# y = net.predict(x)

# x = np.random.rand(100, 784) #ダミーの入力データ100枚分*784ピクセル
# t = np.random.rand(100, 10) #ダミーの正解データ100枚分*10

# grads = net.numerical_gradient(x, t) #ここでなんか時間かかる‥

# print("grads['W1'].shape")
# print(grads['W1'].shape)
# print(grads['b1'].shape)
# print(grads['W2'].shape)
# print(grads['b2'].shape) 

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

#学習による誤差推移
# print("train_loss_list", train_loss_list)
# plt.plot(train_loss_list)
# plt.xlabel("iteration")
# plt.ylabel("loss")
# plt.show() #しかしここで得られた損失関数はミニバッチに対する損失関数(100枚)

#精度の推移を表示してみた 
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()




import numpy as np
import sys, os
sys.path.append('/Users/eb604/deep-learning-from-scratch-master')
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from dataset.mnist import load_mnist
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
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


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

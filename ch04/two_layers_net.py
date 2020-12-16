import sys
sys.path.append('../')
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np
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
        W1, W2 = self.params['W1'], self.params['W2'] 
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

    #勾配 これ時間かかる‥ → [誤差逆伝播法]
    def numerical_gradient(self, x, t): 
        #W重みを引数としたloss_W関数。入力と正解データを実引数としたlossの値を返却。数値微分
        loss_W = lambda W: self.loss(x, t) 
        #勾配のディクショナリ変数。pramsと同じようにそれぞれの勾配が格納される。
        grads = {} 
        #loss_Wとself.params['W1']を実引数としたnumerical_gradientの値を代入。
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) 
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

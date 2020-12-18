import numpy as np
import sys
sys.path.append('../')
from dataset.mnist import load_mnist #load_mnist関数の呼び出し
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict
#------------- 誤差逆伝播に対応したNNの実装 ------------------
#計算が早くなる。
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): 
        #__init__クラスの初期化メソッド。input_size=784,output_size=10クラス,hiddenは適当な数を設定
        self.params = {} #ディクショナリ変数。それぞれNumpy配列で格納。
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #random.randn = 形状が(784*50)の(0以上1未満の乱数)
        # print("W1", self.params['W1'][0]) #(50,)
        # print(input_size) #784
        # print(hidden_size) #50
        self.params['b1'] = np.zeros(hidden_size) #形状は(50)で全て0のバイアス。
        # print("b1", self.params['b1'])
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)#(50*10)
        # print("W2", self.params['W2'][0]) #(10,)
        self.params['b2'] = np.zeros(output_size)
        # print(output_size) #10
        # print("b2", self.params['b2'])
        #レイヤの作成
        self.layers = OrderedDict() #順番付きのディクショナリ。追加した順番を覚えることができる。→追加した順にレイヤのforward()を呼び出すだけで処理が完了。backwardも同様。AffineレイヤやReLuレイヤが順伝播・逆伝播をかんたんにしてる。
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu1'] = Relu()
        # self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        # print("self.layers", self.layers)
    #推測関数
    def predict(self, x): 
        for layer in self.layers.values():
            x = layer.forward(x)
            # print(layer, x.shape)
        return x        
    #損失関数
    def loss(self, x, t): #x入力データ・t教師データ
        y = self.predict(x) #predictの値をyに代入
        return self.lastLayer.forward(y, t) #交差エントロピー誤差
    #正確率
    def accuracy(self, x, t): 
        y = self.predict(x) #出力yにxのself.predictの値を代入。
        self.lastLayer.forward(y, t)
        print("予測確率", self.lastLayer.y)
        y = np.argmax(self.lastLayer.y, axis=1) #axis=1　1次元を(列)を軸に最大値を抜き出す。
        if t.ndim != 1 : t = np.argmax(t, axis = 1)
        print("予測結果", y)
        print("正解データ",t)
        accuracy = np.sum(y == t) / float(x.shape[0]) #y==tの合計値/入力値の形状の0次元
        return accuracy
    #勾配
    def numerical_gradient(self, x, t): ##ここでなんか時間かかる‥ → [誤差逆伝播法]
        loss_W = lambda W: self.loss(x, t) #W重みを引数としたloss_W関数。入力と正解データを実引数としたlossの値を返却。数値微分
        grads = {} #勾配のディクショナリ変数。pramsと同じようにそれぞれの勾配が格納される。
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) #loss_Wとself.params['W1']を実引数としたnumerical_gradientの値を代入。
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads #全てのパラメータを配列に格納し終わったらgradsで返す。
    #誤差逆伝播法
    def gradient(self, x, t):
        #forward
        self.loss(x,t) #損失関数を求める
        #backward
        dout = 1 #最終的な出力の微分を１とする
        dout = self.lastLayer.backward(dout) #出力層の誤差
        layers = list(self.layers.values()) #self.layersのvaluesをlist化して代入
        layers.reverse() #順番を逆にする
        for layer in layers: #３回逆伝播
            # print("layer", layer)
            dout = layer.backward(dout)
        #設定
        grads = {} #ディクショナリ変数に再度格納していく
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        # grads['W2'] = self.layers['Affine2'].dW
        # grads['b2'] = self.layers['Affine2'].db
        return grads
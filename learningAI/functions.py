import numpy as np
from numpy.core.numeric import outer

#------- 4章 --------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
#------- 乗算レイヤー -------------------------------------------------------
class MulLayer:
    def __init__(self): #インスタンス変数のxとyの初期化。順伝播時の入力値を保持するため。
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y #入力値を蒸散して返す

        return out

    def backward(self, dout):
        dx = dout * self.y #入力値をひっくり返して乗算→乗算する順伝播においては入力値を保持する必要がある→__init__
        dy = dout * self.x
        return dx, dy #上流から伝わってきた微分(dout)に対して順伝播のひっくり返した値を乗算？？？？？？？？

#りんごの買い物を乗算レイヤーでやってみよう！
# apple = 100
# aapple_num = 2
# tax = 1.1

# #layer
# mul_apple_layer = MulLayer() #MulLayerインスタンスを生成
# mul_tax_layer = MulLayer()

# #forward
# apple_price = mul_apple_layer.forward(apple, aapple_num) #appleとapple_numを実引数としてforwardする
# price = mul_tax_layer.forward(apple_price, tax) #apple_priceとtaxを実引数としてforwardする

# print(price) #priceで返す

#各変数に対する微分
#backward　入力値をひっくり返して乗算
# dprice = 1 #（りんごの値段が1円増えたら、りんごの個数値が１増えたら、消費税数値が１増えたら、）各入力値の数値が１増えたら‥
# dapple_price, dtax = mul_tax_layer.backward(dprice) #dapple_price = 1 * dtax, dtax = 1 * dapple_price
# dapple, dapple_num = mul_apple_layer.backward(dapple_price) #dapple = 1 * dapple_num, dapple_num = 1 * dapple
# print(dapple, dapple_num, dtax) #最終的な値段は‥‥（りんごの値段:2.2円、りんごの個数:110円、消費税:200円増える）

#-------- 加算レイヤー --------------------------------------------
class AddLayer:
    def __init__(self):
        pass #何も行わない
    
    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout): #上流から伝わってきた微分をそのまま流す。
        dx = dout * 1
        dy = dout * 1
        return dx, dy

#りんごとみかんの買い物を加算レイヤーでやってみよう！
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()
#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
#bacward
dprice = 1 # それぞれの入力値の値が１増えたときに
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# print(price)
# print(dapple_num, dapple, dorange, dorange_num, dtax) #110.00000000000001, 2.2, 3.3000000000000003, 165.0, 650

#------------- ReLuレイヤー　----------------------
class ReLu:
    def __init__(self):
        self.mask = None #mask変数はT/FからなるNumpy配列。入力値ｘ<=0ならTrue, それ以外をFalseとします。
    
    def forward(self, x):
        self.mask = (x <=0) #x <=0　の場合、True, それ以外はFalse
        out = x.copy() #outにxの値をわたす。後のｘの変更の影響は受けない。
        out[self.mask] = 0 #self.maskがTrueのときだけoutに０を代入

        return out

    def backward(self, dout): 
        dout[self.mask] = 0 #self.maskがTrueのときだけdoutを０にする。入力値が０ならば逆伝播の値も０となる。そのため入力値ｘ<=0のときはdout=0
        dx = dout

        return dx

#True/False例
# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# print(x)
# mask = (x <= 0)
# print(mask)
"""
[[False  True]
[ True False]]
"""
#---------- sigmoidレイヤー -------------------------
class Sigmoid:
    def __init__(self):
        self.out = None #出力outを順伝播時に保持しておく。逆伝播時にそれを使用。
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.dout) : self.outer

        return dx
#----------- Affineレイヤー(順伝播で行う内積)-------------------------
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x. self.W) + self.b #出力

        return out
        
    def backward(self, dout):
        dx = np.dot(dout, self.W.T) #出力から得られた微分とWの転置したものの内積
        self.dW = np.dot(self.x.T, dout) #転置されたx入力値と微分の内積
        self.db = np.sum(dout, axis=0) #微分を０次元を軸に足していく

        return dx
    #------------- ソフトマックス関数と損失関数 -----------------------------
    #--------------交差エントロピー誤差--------------------------------------
# def cross_entropy_error(y, t): #y=NNの出力、t=教師データ
#     # print("y.ndim", y.ndim)
#     if y.ndim == 1: #yが１次元のとき（データ一つあたりの交差エントロピー誤差を求めるとき）データの形状を整形。
#         t = t.reshape(1, t.size) #バッチデータと次元を合わせる。
#         y = y.reshape(1, y.size)
#         # print(t.shape)
#         # print(y.shape)
#     batch_size = y.shape[0] # バッチサイズはｙの０次元の数。
#     # print("t",t)s
#     # print("y",y)
#     # print("batch_size",batch_size)
#     return -np.sum(t * np.log(y)) / batch_size 

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
            
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#----------------------------------------------------------------------
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class SoftmaxWithLoss:
    def __init__(self): #保持しておきたいもの
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
    
    def backward(self, dout=1):
        batch_size=self.t.shape[0]
        dx = (self.y - self.t) / batch_size #一枚あたりの誤差
        return dx
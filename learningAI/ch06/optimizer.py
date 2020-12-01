import numpy as np

class SGD:#確率的勾配降下（+バッチ化→ミニバッチSGD）
    def __init__(self, lr=0.01):
        self.lr = lr
    #それぞれのパラメータを更新
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        #
        if self.v is None:
            self.v = {}
            #paramsのkeyとvalueを引数として以下を繰り返す
            for key, val in params.items():
                #パラメータと同じ構造のデータをディクショナリ変数として保持
                self.v[key] = np.zeros_like(val)
                # print("self.v['b1']", self.v)

        for key in params.keys():
            #0.9*前回のパラメータの更新量(減衰させる) - 学習率*パラメータの勾配 → 各パラメータ(key)の速さ
            # 前回パラメータを更新した方向の速度を90％もたせながら今回計算した勾配を反映させる
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            #各パラメータに速さを足す
            params[key] += self.v[key]
            # print("--------------------self.v-----------------------------", self.v)
            # print("^^^^^^^^^^^^^^^^^^^^^^^params^^^^^^^^^^^^^^^^^^^^^^^^^", params)

class AdaGrad:
    #初期設定
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {} #ディクショナリ変数を生成
            for key, val in params.items(): 
                self.h[key] = np.zeros_like(val)

        #paramsのkeyごとに更新(重みやバイアス)
        for key in params.keys(): 
            self.h[key] += grads[key] * grads[key]
            #勾配の二乗のnp.sqrtの逆数をかけることで学習係数を調整
            #大きく更新されたパラメータの要素は学習係数が小さくなる
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) 

class Adam: #MomentumとAdaGradの融合

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1 #２つのモメンタムを用意する
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


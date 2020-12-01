import numpy as np
import matplotlib.pylab as plt

def init_network(): #それぞれの重みバイアスの配列をディクショナリ型配列に格納
  network={}
  #key            #value
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def sigmoid(x): #シグモイド関数　中間層
  return 1/(1 + np.exp(-x))

def identity_function(x): #恒等関数（回帰問題でよく使われる）出力層
  return x

def forward(network,x): #順伝播
  W1, W2, W3 = network['W1'], network['W2'], network['W3'] #networkディクショナリの重み、バイアスをそれぞれ代入
  b1, b2, b3 = network['b1'], network['b2'], network['b3'] 

  print("W1, W2, W3")
  print(W1, W2, W3)
  print("b1, b2, b3")
  print(b1, b2, b3)


  a1 = np.dot(x, W1) + b1 #入力xに重み１W1をかけてバイアスを足す。
  z1 = sigmoid(a1) #a1の結果を活性化関数シグモイドにいれて計算
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identity_function(a3) #出力層では活性化関数恒等関数に入れて計算してから出力。

  return y

network = init_network() #init_networkをnetworkに代入する
x = np.array([1.0, 0.5]) #入力値ｘ
y = forward(network, x) #networkとｘを実引数とした順伝播を行う
print(y)



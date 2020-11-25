print("三層nnの実装")
def AND (x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1

# print(AND(1, 1))

import numpy as np
# def Perseptron(x):
#   w = np.array([0.5, 0.5])
#   b = -0.7
#   theta = 0.8
#   print(w*x)
#   print(np.sum(w*x))
#   tmp = np.sum(w*x) + b
#   print( tmp )
#   if tmp <= theta:
#     return 0
#   elif tmp > theta:
#     return 1  

# x = np.array([2,2])
# print(Perseptron(x))



def NAND(x1, x2):  
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = 0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

# print(NAND(1,1))
# print(OR(1,1))

def XOR(x1, x2):
  s1 = NAND(x1,x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y

# print(XOR(1,1))

def step_function(x):
  if x > 0:
    return 1
  else:
    return 0



x = np.array([-1.0, 1.0, 2.0])
# print(x)
y = x > 0
# print(y)
y = y.astype(np.int)
# print(y)

import matplotlib.pylab as plt

def step_function(x):
  return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1) #-5から５を0.1ずつ刻みNumPy配列を生成
y = step_function(x) #xの配列のそれぞれの値に対して実行し、０か１か
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1) #y軸の範囲
# plt.show()

def sigmoid(x):
  return 1/(1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
# print(sigmoid(x))

t = np.array([1.0,])

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.show()

def relu(x):
  return np.maximum(0, x)
 
# x=4
# print(relu(x))

A = np.array([1,2,3,4])
# print(A)

# print(np.ndim(A))
# print(A.shape)
# print(A.shape[0])

# print("B配列の形")
B = np.array([[1,2,3], [2,3,4], [3,4,5], [3,4,5]])
# print(B)
# print(np.ndim(B))
# print(B.shape)
# print(B.shape[1])

A = np.array([[1,2], [2,3]])
# print(A.shape)
B = np.array([[8, 9], [8, 7]])
# print(B.shape)
# print(np.dot(A, B)) #np.dotはNumpy配列を２つ撮って内積を返す。1*8+2*8 1*9+2*7 2*8+3*8 2*9+3*7

# print("練習１")
# A = np.array([[1,2,3], [4,5,6]]) #入力２個
# print(A.shape)
# B= np.array([[1], [4], [5]]) #２つの行列で対応する次元数を揃える。ここで2.3と2.2、2.3と2.3とかだったらエラーが出る。2.3と3.2のどちらか。1*1+2*4+3*5=24, 4*1+5*4+6*5=54。Aの１次元目(3)とBの０次元目の数があっていないといけない。
# #重み３個
# print(B.shape)
# print(np.dot(A,B))
# print(np.dot(A,B).shape) #出力３個

# print("練習２")
# A = np.array([[1,2],[3,4],[5,6]]) #1*7+2*8=23, 3*7+4*8=53, 5*7+6*8=83　入力３個
# print(A.shape)
# B = np.array([7,8]) #重み１個
# print(B.shape)
# print(np.dot(A, B))
# print(np.dot(A,B).shape) #出力１個

# print("重みだけを踏まえたニューラルネットワークを実装してみよう")
# x = np.array([1,2]) #入力２個
# print(x.shape)
# w = np.array([[1,3,5], [2,4,6]]) #入力[1,2]に重み[1,3,5]と[2,4,6]の要素をそれぞれかける。1*1+2*2=5
# print(w.shape)
# y = np.dot(x, w)
# print(y) #出力が３個となる
# print(y.shape)

print("nnの実装")
print("入力層から出力層への伝達と値の調整")

X = np.array([1.0, 0.5]) #2個の出力
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #３つの中間層１
B1 = np.array([0.1, 0.2, 0.3])

# print(W1.shape)
# print(X.shape)
# print(B1.shape)

A1 = np.dot(X, W1) + B1

print("A1")
print(A1)
# print(A1.shape)

Z1 = sigmoid(A1) #中間層でシグモイド関数

print("Z1")
print(Z1) #中間層出力

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) #3つの中間層１から２つの中間層２(１個のニューロンに対して２つの重み)
B2 = np.array([0.1, 0.2])

# print(Z1.shape)
# print(W2.shape)
# print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print("A2")
print(A2)
print("Z2")
print(Z2)

print("中間層から出力層への伝達")
def identity_function(x): #恒等関数（回帰問題でよく使われる）
  return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2,W3) + B3 
print("A3")
print(A3)

Y = identity_function(A3)
print("出力の値")
print(Y)

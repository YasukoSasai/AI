#=========== 三層nnの実装 =============(3.4.3)
def AND (x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1

import numpy as np

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

def XOR(x1, x2):
  s1 = NAND(x1,x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y

def step_function(x):
  if x > 0:
    return 1
  else:
    return 0

def step_function(x):
  return np.array(x > 0, dtype=np.int)

#２クラス分類なのでsigmoid関数
def sigmoid(x):
  return 1/(1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)


# ========= 発表で実行 ===========
print("====== nnの実装 ======")
print("====== 入力層から中間層への伝達(シグモイド関数) ======")
X = np.array([1.0, 0.5]) #2個の入力
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #３つの中間層に対する重み
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

print("====== 中間層から出力層への伝達 ======")
def identity_function(x): #恒等関数（回帰問題でよく使われる）
  return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2,W3) + B3 
print("A3")
print(A3)
print("===== (恒等関数) ======")
Y = identity_function(A3)
print("出力値")
print(Y)

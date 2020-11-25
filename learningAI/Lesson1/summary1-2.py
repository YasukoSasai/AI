#以上のMNISTデータを利用して推論処理を行うNNを作る。（入力層784=28*28、中間層①50、中間層②100、出力層10 "0"~"9")
from __future__ import print_function
import numpy as np
import pickle
import sys, os
os.chdir('/Users/eb604/deep-learning-from-scratch-master/ch03')
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data(): #get_data関数
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, flatten = True, one_hot_label = False) #normalize=入力画像の各ピクセルを255で除算し０〜１の値に正規化, flatten=入力画像を一次元にする, one_hot_label=ラベルをone_hot表現で格納しない。ラベルはわかりやすいように0~9にしておく。
    return x_test,  t_test

def init_network(): #init_network関数
    with open("sample_weight.pkl", 'rb') as f: #sample_weight.pklファイルを読み込みモード(rb)でファイルオブジェクト(f)としてオープンしたら(ディクショナリ型の重みとバイアスが保存されている)
        network = pickle.load(f) #fの読み込み。非直列化し、Pythonオブジェクトに復元する
    return network

# print(init_network())

def predict(network, x): #predict関数
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] #networkのそれぞれの重みを代入
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] #networkのそれぞれのバイアスを代入

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
# print("t", t)
# print("t.shape", t.shape)
# print("x", x)
# print("x.shape", x.shape)
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) #最も高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

#---------------------------------------------------------------
#バッチ処理
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

# print(x.shape) # 10000*784
# print(x[3].shape) #xの0番目の形は一次元の784行 
# print(W1.shape)
# print(W2.shape)
# print(W3.shape)
# print(y.shape)

x, t = get_data()
network = init_network()

batch_size = 100 #100枚分の画像データ(100*784)
accuracy_cnt = 0

for i in range(0, len(x), batch_size): #0からlen(x)-1までの整数のリストを生成し、以下の処理を繰り返す。また、100枚ごと取り出して計算を行う。
    x_batch = x[i:i+batch_size] #iからi+100-1番目のデータ
    y_batch = predict(network, x_batch) #networkとx_batchを実引数としたpredict関数の値
    p = np.argmax(y_batch, axis=1) #y_batchの中から最大値を取る。また、axis=1で100*10の配列の中で一次元目を軸として最大値のインデックスを取り出す。100枚のデータそれぞれが確率の高い番目を出す。
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) #i番目からi+100-1番目のまでのTrueの数を足す。
    # print(accuracy_cnt)

print("Accuracy:" + str(float(accuracy_cnt)/len(x))) #全体のTrueの合計 / xの長さ

print(len(x))

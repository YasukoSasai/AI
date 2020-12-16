# =============== mnistに対して推論処理 ====================
import sys, os
sys.path.append('../')
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# def get_data():
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
"""
normalize=入力画像を０〜１の値に正規化, 
flatten=入力画像を一次元にする, 
one_hot_label=ラベルをone_hot表現で格納しない。ラベルはわかりやすいように0~9にしておく。
"""
    # return x_test, t_test


def init_network(): 
    with open("/Users/eb604/deep-learning-from-scratch-master/ch03/sample_weight.pkl", 'rb') as f:
        """
        学習済みのパラメータが格納されています。
        sample_weight.pklファイルを読み込みモード(rb)で
        ファイルオブジェクト(f)としてオープンする
        (ディクショナリ型の重みとバイアスが保存されている)
        """
        network = pickle.load(f)
    return network


def predict(network, x):
    #pkl得たデータをそれぞれ代入
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3) #確率が出る

    return y

#=========== 実行　＝＝＝＝＝＝＝＝＝＝＝
network = init_network()
#------- 入力するテストデータ ------------------------
print("============ 入力データ =============") #784
print(x_test[0])
#------- 正解は？ ---------------------------
print("============= 正解 ===============") 
print(t_test[0])
#------- 推論結果 --------------
print("============ 予測結果 ============")
print(predict(network, x_test[0])) 

#=========== テストデータ全てでの性能 ==========
accuracy_cnt = 0
for i in range(len(x_test)): #10000枚分繰り返す
    y = predict(network, x_test[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t_test[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))

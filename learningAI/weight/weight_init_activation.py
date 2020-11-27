import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100) #入力値。1000*100配列の乱数(0以上1未満)

node_num = 100 #隠れ層のニューロン数
hidden_layer_size = 5 #隠れ層は５層
activations = {} #アクティベーションの結果を格納。ディクショナリ

for i in range(hidden_layer_size): #5回繰り返す
    if i != 0: #iが０でなければ
        x = activations[i-1] #i-1のactivationsをｘに代入

    # w = np.random.randn(node_num, node_num) * 1 #wに100＊100配列の乱数(0以上1未満)。標準偏差が１のガウス分布
    # w = np.random.randn(node_num, node_num) * 0.01 #wに100＊100配列の乱数(0以上1未満)。標準偏差が0.0１のガウス分布
    # w = np.random.randn(node_num, node_num) / np.sqrt(node_num) #Xavier単純化

    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) #Xavier
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) #He


    z = np.dot(x, w) #ｘとｗの内積結果をｚに代入
    a = sigmoid(z) #ｚのシグモイド関数結果をaに代入。最終的な出力値。
    activations[i] = a #各層の出力値を格納。
    # print(a)

for i, a in activations.items(): #層ごとにヒストグラムを作成
    plt.rcParams["font.size"] = 7
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)/
    plt.hist(a.flatten(), 30, range=(0,1)) #(aを一次元配列にする,  )

plt.show()

# for i, a in activations.items():
#     plt.subplot(1, len(activations), i+1)
#     plt.title(str(i+1) + "-layer")
#     if i != 0: plt.yticks([], [])
#     # plt.xlim(0.1, 1)
#     # plt.ylim(0, 7000)
#     plt.hist(a.flatten(), 30, range=(0,1))
# plt.show()

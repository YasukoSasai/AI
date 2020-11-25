from typing import NewType
import numpy as np
import matplotlib.pylab as plt

# a = np.array([0.3, 2.9, 4.0])

# exp_a = np.exp(a) #指数関数
# print(exp_a)

# sum_exp_a = np.sum(exp_a) #指数関数の和
# print(sum_exp_a)

# y = exp_a / sum_exp_a #出力
# print(y)


def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a/sum_exp_a

  return y

#ソフトマックスにおける問題（オーバーフロー）の対策。大きい値同士で割り算するとおこる。
# a = np.array([1010, 1000, 990])
# print(np.exp(a) / np.sum(np.exp(a)))

# c = np.max(a)
# print(a-c) #aのもとの値からaの最大値を引く

# print(np.exp(a-c)/np.sum(np.exp(a-c)))

#以上を踏まえたソフトマックス関数
def softmax(a):
  c = np.max(a) #前処理
  exp_a = np.exp(a-c)
  sum_exp_a = np.sum(exp_a) #計算
  y = exp_a / sum_exp_a
  return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
# print(y)
# print(np.sum(y)) #ソフトマックス関数の各値は確率なので合計値は１。推論の際はソフトマックス関数がなくても各値の大小関係は変わらないので、必要ないと思われがちですが、学習においてはこの確率値にすることが重要になってきます。


#MNIST
import sys, os
os.chdir('/Users/eb604/deep-learning-from-scratch-master/ch03')
sys.path.append(os.pardir) 
from dataset.mnist import load_mnist #load_mnist関数の呼び出し

(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False) #flatten=入力画像を一次元にするかどうか、normalize=入力画像を０〜１の値に正規化するかどうか（他に、　one_hot_label=ラベルをone_hot表現で格納するかどうか。one_hot表現＝正解ラベル１，それ以外０）
  # pickle=プログラム実行中のオブジェクトをファイルとして格納する
#データの形状
# print(x_train.shape) #訓練データの入力データ
# print(t_train.shape) #訓練データの正解データ
# print(x_test.shape) #テストデータの入力データ
# print(t_test.shape) #テストデータの正解データ

#MNIST画像を表示
from PIL import Image #画像表示にはPILモジュールを使う。

def img_show(img):
  # print("-------- 元の型 ---------")
  # print(np.uint8(img)) #0~255で28行
  # print("np.uint8(img)", np.uint8(img).shape) #28行*28列
  pil_img = Image.fromarray(np.uint8(img)) #np.uint8=８ビット型の符号なしデータ。Numpyとして格納された画像データをPIL用の画像オブジェクトに変換。する必要がある。
  # print("-------- PIL型 ---------")
  # print(pil_img)
  pil_img.show()

(x_train, t_strain), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False) #読み込みしたload_mnist関数によってMNISTデータセットの読み込み。Flatten=Trueなので飲み込んだ画像はNumpy配列に一次元（一列）で格納。→なので後ほど(28,28)に再変形。

img = x_train[0]
label = t_train[0]
# print(img.shape) #784
# print(label) #正解のデータ 5

img = img.reshape(28, 28)
# print(img.shape) #28*28

# img_show(img)




#--------------------------------------------------------------------------------------------------------------------




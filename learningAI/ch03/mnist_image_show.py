#========= どのように画像表示するか ==========(3,6,1)
#MNIST
import sys, os
sys.path.append('../../../deep-learning-from-scratch')
from dataset.mnist import load_mnist #load_mnist関数の呼び出し
import numpy as np

(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False)
"""
  flatten:入力画像を一次元にするかどうか、TrueなのでNumpy配列に一次元（一列）で格納。→なので後ほど(28,28)に再変形。
  normalize:入力画像を０〜１の値に正規化するかどうか
  （one_hot_label:ラベルをone_hot表現で格納するかどうか。one_hot表現＝正解ラベル１，それ以外０）
"""
#データの形状
print("x_train.shape", x_train.shape) #訓練データの入力データ
print("t_train.shape", x_train.shape) #訓練データの正解データ
print("x_test.shape", x_test.shape) #テストデータの入力データ
print("t_test.shape", t_test.shape) #テストデータの正解データ

#MNIST画像を表示
from PIL import Image #画像表示にはPILモジュールを使う。

def img_show(img):
  # print("-------- 元の型 ---------")
  # print("np.uint8(img)", np.uint8(img)) #0~255で28行
  # print("np.uint8(img).shape", np.uint8(img).shape) #28行*28列
  pil_img = Image.fromarray(np.uint8(img)) 
  #np.uint8=８ビット型の符号なしデータ。Numpyとして格納された画像データをPIL用の画像オブジェクトに変換。する必要がある。
  # print("-------- PIL型 ---------")
  # print(pil_img)
  pil_img.show()
#--------------------入力する訓練データ-------------------------
img = x_train[9999]
# print("img.shape", img.shape) #784
#------------------- 正解は？----------------------------
label = t_train[9999]
# print("正解", label) #正解のデータ 5
#-------------------変形してimg.show()に入れて実行 --------------------
img = img.reshape(28, 28)
# print("img.shape2", img.shape) #28*28
img_show(img)
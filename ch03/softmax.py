#==========　ソフトマックス関数　=========(3.５.３)
import numpy as np

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

#Affineからのsoftmax。[0.3, 2.9, 4.0]という出力が出てきたときに最終的に何％となるか。
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y)) #ソフトマックス関数の各値は確率なので合計値は１。推論の際はソフトマックス関数がなくても各値の大小関係は変わらないので、必要ないと思われがちですが、学習においてはこの確率値にすることが重要になってきます。

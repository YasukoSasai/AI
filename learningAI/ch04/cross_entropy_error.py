#3.6の例を引き継いで
import numpy as np
#正解データ(one-hot表現)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#"2"の確率が0.6(softmax関数を使った結果)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

def cross_entropy_error(y, t):
    delta = 1e-7
    #[正解データ×{log(出力値 + 微小な値)}]の合計×マイナス
    #実質正解データが1に対応する出力の自然対数を計算するだけになる。
    return -np.sum(t * np.log(y + delta))

print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))



#===== LOGグラフ(泣) =======
# x = np.arange(0.001, 1.0, 0.1)
# import matplotlib.pyplot as plt
# x = np.arange(0.001, 1.0, 0.1)
# y = np.log(x)
# # 学習による誤差推移
# # print("train_loss_list", train_loss_list)
# plt.plot(y)
# plt.xlabel("iteration")
# plt.ylabel("loss")
# plt.xlim([0.0,1.0])
# plt.ylim([-5,0])
# # plt.show() #しかしここで得られた損失関数はミニバッチに対する損失関数(100枚)
# with np.errstate(invalid='ignore'):
#     x = np.arange(0.01, 1, 0.01)
#     y = np.log(x)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([-5, 0])
#     plt.xlabel('x')
#     plt.ylabel('y', rotation=0)
#     plt.gca().set_aspect('equal')
#     plt.grid()
#     plt.plot(x, y)
#     plt.figure(figsize=(7,5,))

#     plt.show()

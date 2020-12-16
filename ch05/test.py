#========= (誤差逆伝播を使った)学習=============(5.7.4)
#２層ニューラルネットワークのクラス
import sys
sys.path.append('../')
from dataset.mnist import load_mnist
from two_layers_net import TwoLayerNet
from PIL import Image #画像表示にはPILモジュールを使う。
import numpy as np
import matplotlib.pyplot as plt

#============ データを取得(入力データを一次元化、正解ラベルをone_hot_label化) ==================
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print("----- 元データ内容 -----")
print("x_train.shape", x_train.shape) #訓練データの入力データ
print("t_train.shape", t_train.shape) #訓練データの正解データ one_hot_labelにしてるから10
print("x_test.shape", x_test.shape) #テストデータの入力データ
print("t_test.shape", t_test.shape) #テストデータの正解データ
# ================== ディープラーニングのモデルを定義 ==================
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #NNのインスタンス生成

# =======================================================

#====== ハイパーパラメータ ======
iters_num = 10000 #勾配によるパラメータ更新回数(iterarion = 繰り返し)
train_size = x_train.shape[0] #訓練データのサイズはx_trainデータ形状の0次元の数 = 60000枚　x_train.shape = 60000枚 * 784(28*28)ピクセル
batch_size = 100 #60000枚のうち100枚ごと学習を行う
learning_rate = 0.1 #学習率

train_loss_list = [] #学習ごとの損失関数を格納するためのリスト
train_acc_list = [] #学習における正確率
test_acc_list = [] #テストにおける正確率

iter_per_epoch = max(train_size / batch_size, 1) #1エポックあたりの繰り返し数　エポック=訓練データをすべて使い切った回数。60000/100枚 回勾配を行った = １エポック学習を行った。600回。

# 学習フェーズ
print("====== 学習開始 ======")
#何枚の画像で学習したか(画像のシェイプも)
#開始時間取得
for i in range (iters_num): #10000回繰り返し
    #===== ミニバッチの取得 =====
    batch_mask = np.random.choice(train_size, batch_size) #train_size個の中からbatch_size個ランダムでインデックスを取り出す
    x_batch = x_train[batch_mask] #100個の入力画像
    t_batch = t_train[batch_mask] #100個の入力画像に対する正解データ

    if i == 1:
        # print("画像データ", x_batch[0]) #すべて出すと大量になるので一枚だけ
        # print("画像データ", t_batch[0])         
        print("---- １回の学習で使われるバッチデータ ----")
        print("x_batch.shape", x_batch.shape)
        print(str(x_batch.shape[0]) + "枚分、"+ str(x_batch.shape[1]) + "ピクセル/枚")
        print("t_batch.shape", t_batch.shape)
        print(str(t_batch.shape[0]) + "枚分の"+ str(t_batch.shape[1]) + "個の正解ラベル")

    #====== 勾配計算 =====
    #数値微分
    # grad = network.numerical_gradient(x_batch, t_batch)
    #誤差逆伝播法　高速！ 
    grad = network.gradient(x_batch, t_batch)

    #====== パラメータ更新 ======
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    #====== 学習経過の記録 ========
    # loss = network.loss(x_batch, t_batch)
    # train_loss_list.append(loss)
#終了時間取得
#print(終了ー開始)

#1エポックごとにテストデータで認識精度を計算　計算に時間がかかるのでざっくりと。
# if i % iter_per_epoch == 0:
print("====== テスト開始 ======")
# #開始時間取得
#================ 複数枚まとめてテストしたいとき(画像表示の仕方も変更する) ===========
# test_size = x_test.shape[0]
# batch_mask = np.random.choice(test_size, 10) #何枚でテストを行いたいか
# print("batch_mask",batch_mask)
# #train_size枚の中からbatch_size枚ランダムで配列で取り出す
# x_test = x_test[batch_mask] #100個の入力画像
# t_test = t_test[batch_mask] #100個の入力画像に対する正解データ
# test_acc = network.accuracy(x_test, t_test)
# test_acc_list.append(test_acc)
# print(test_acc_list)


#============== １枚テストデータを指定してテストしたいとき ===========
x_test = x_test[[4]] 
t_test = t_test[[4]] 
test_acc = network.accuracy(x_test, t_test)
#終了時間取得
#print(終了ー開始)

# ===========テスト時の精度 ============
print("test acc: "+ str(test_acc))


#============================================

# def ConvertToImg(img):
#     return Image.fromarray(np.uint8(img))
# # MNIST一文字の幅
# chr_w = 28
# # MNIST一文字の高さ
# chr_h = 28
# # 表示する文字数
# num = 20
# # MNISTの文字をPILで１枚の画像に描画する
# canvas = Image.new('RGB', (int(chr_w * num/2), int(chr_h * num/2)), (255, 255, 255))
# # MNISTの文字を読み込んで描画
# i = 0
# #======== 画像表示のために一旦unnormalizedデータを読み込む。(もっといい方法ありますかね) ==========
# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(flatten=True, normalize=False)
# #============ 一枚の画像を出したいとき ===========
# chrImg = ConvertToImg(x_test[0].reshape(chr_w, chr_h))
# canvas.paste(chrImg, (chr_w*i, chr_h))
# #============ 複数ででテストする際の画像表示 ==============
# # for item in batch_mask:
# #     chrImg = ConvertToImg(x_test[item].reshape(chr_w, chr_h))
# #     canvas.paste(chrImg, (chr_w*i, chr_h))
# #     i = i + 1

# canvas.show()
# 表示した画像をJPEGとして保存
# canvas.save('mnist.jpg', 'JPEG', quality=100, optimize=True)


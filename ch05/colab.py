#========= (誤差逆伝播を使った)学習=============(5.7.4)
#２層ニューラルネットワークのクラス
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layers_net import TwoLayerNet
from PIL import Image #画像表示にはPILモジュールを使う。
# ================== ミニバッチ学習の実装 ==================
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#=======================================================
# ================== ディープラーニングのモデルを定義 ==================
network = TwoLayerNet(input_size=784, hidden_size=10, output_size=10) #NNのインスタンス生成
# =======================================================
#====== ハイパーパラメータ ======
iters_num = 10000 #勾配によるパラメータ更新回数(iterarion = 繰り返し)
train_size = x_train.shape[0] #訓練データのサイズはx_trainデータ形状の0次元の数 = 60000枚　x_train.shape = 60000枚 * 784(28*28)ピクセル
batch_size = 1000 #60000枚のうち100枚ごと学習を行う
learning_rate = 0.1 #学習率
train_loss_list = [] #学習ごとの損失関数を格納するためのリスト
train_acc_list = [] #学習における正確率
test_acc_list = [] #テストにおける正確率
#============================
iter_per_epoch = max(train_size / batch_size, 1) #1エポックあたりの繰り返し数　エポック=訓練データをすべて使い切った回数。60000/100枚 回勾配を行った = １エポック学習を行った。
# 学習フェーズ
print("学習開始")
#何枚の画像で学習したか(画像のシェイプも)
#開始時間取得
for i in range (iters_num): #10000回繰り返し
    #===== ミニバッチの取得 =====
    batch_mask = np.random.choice(train_size, batch_size) #train_size枚の中からbatch_size枚ランダムで配列で取り出す
    x_batch = x_train[batch_mask] #100個の入力画像
    t_batch = t_train[batch_mask] #100個の入力画像に対する正解データ
    #====== 勾配計算 =====
    #数値微分
    # grad = network.numerical_gradient(x_batch, t_batch)
    #誤差逆伝播法　高速！ 
    grad = network.gradient(x_batch, t_batch)
    #====== パラメータ更新 ======
    # for key in ('W1', 'b1', 'W2', 'b2'):
    for key in ('W1', 'b1'):
        network.params[key] -= learning_rate * grad[key]
    #====== 学習経過の記録 ========
    # loss = network.loss(x_batch, t_batch)
    # train_loss_list.append(loss)
#終了時間取得
#print(終了ー開始)
#1エポックごとにテストデータで認識精度を計算　計算に時間がかかるのでざっくりと。
# if i % iter_per_epoch == 0:
print("テスト開始")
#開始時間取得
test_size = x_test.shape[0]
batch_mask = np.random.choice(test_size, 10) #train_size枚の中からbatch_size枚ランダムで配列で取り出す
x_test = x_test[batch_mask] #100個の入力画像
t_test = t_test[batch_mask] #100個の入力画像に対する正解データ
test_acc = network.accuracy(x_test, t_test)
test_acc_list.append(test_acc)
#終了時間取得
#print(終了ー開始)
print("test acc: "+ str(test_acc))
#============================================
def ConvertToImg(img):
    return Image.fromarray(np.uint8(img))
# MNIST一文字の幅
chr_w = 28
# MNIST一文字の高さ
chr_h = 28
# 表示する文字数
num = 20
# MNISTの文字をPILで１枚の画像に描画する
canvas = Image.new('RGB', (int(chr_w * num/2), int(chr_h * num/2)), (255, 255, 255))
# MNISTの文字を読み込んで描画
i = 0
(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False)
# for x in range( int(num/2) ):
#     chrImg = ConvertToImg(x_test[i].reshape(chr_w, chr_h))
#     canvas.paste(chrImg, (chr_w*x, chr_h))
#     i = i + 1
for item in batch_mask:
    chrImg = ConvertToImg(x_test[item].reshape(chr_w, chr_h))
    canvas.paste(chrImg, (chr_w*i, chr_h))
    i = i + 1
canvas.show()
# 表示した画像をJPEGとして保存
canvas.save('mnist.jpg', 'JPEG', quality=100, optimize=True)
print(network.params['W1'].shape)
for num in range(10):
  a = []
  for i in range(784):
    a.append(network.params['W1'][i][num])
  #画像の表示
  a = np.reshape(a, (28, 28)) 
  plt.imshow(a)
  plt.show() #使った画像が小さいときはボケて見えるけど今は気にしないで
  plt.gray()
  #画像の保存
  plt.imsave('w1_'+str(num)+'.jpeg', a) #拡張子を.pngとかに変えてもちゃんと保存してくれる。
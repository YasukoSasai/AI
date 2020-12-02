import sys
sys.path.append('../../../deep-learning-from-scratch')
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

#====== 訓練データの中から指定された個数をランダムに取り出す =======
train_size = x_train.shape[0]
batch_size = 10
bacth_mask = np.random.choice(train_size, batch_size)
print(x_train.shape)
x_batch = x_train[bacth_mask]
t_batch = t_train[bacth_mask]
print(np.random.choice(60000, 10))

#===== バッチ対応交差エントロピー誤差 ======
def cross_entropy_error(y,t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

#===== 正解データがラベルのとき =====
def cross_entropy_error(y,t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
    #np.arange(batch_size)は０からbatch_size-1までの配列。y[0,2], y[1,4], y[2,9]...みたいな

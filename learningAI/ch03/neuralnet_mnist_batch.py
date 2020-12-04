# =============== mnistに対して推論処理 ====================
import sys, os
sys.path.append('../')
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# def get_data():
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)

def init_network(): 
    with open("/Users/eb604/deep-learning-from-scratch-master/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3) #確率が出る
    print(y.shape)
    return y


network = init_network()

batch_size = 100# バッチの数
accuracy_cnt = 0

for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # print(predict(network, x_test))
    
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t_test[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))
# print((predict(network, x_test)).shape)


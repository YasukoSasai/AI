# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.optimizer import *

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                epochs=20, mini_batch_size=100,
                optimizer='SGD', optimizer_param={'lr':0.01}, 
                evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1) #1エポックあたりのiteration = 60000 / 100 = 600
        self.max_iter = int(epochs * self.iter_per_epoch) #20エポック * 600回 = 12000回
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        #バッチデータ生成
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        #勾配計算とパラメータ更新
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        #損失関数
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss)) #verboseがTrueだったらtrain_lossの値をプリントする

        
        if self.current_iter % self.iter_per_epoch == 0: #600回ごとに
            #エポック数を更新
            self.current_epoch += 1
            #訓練データ、テストデータを以下に代入
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:#evaluate_sample_num_per_epoch????????????????????????????
                t = self.evaluate_sample_num_per_epoch #tに代入
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t] #t番目までの訓練データを代入
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t] #t番目までのテストデータを代入
                print("-------------", self.evaluate_sample_num_per_epoch)
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample) #sampleデータを実引数としたnetwork.accuracyを実行
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc) #結果をリストに追加
            self.test_acc_list.append(test_acc)
            


            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===") #verboseがTrueならばプリント
        self.current_iter += 1 #current_iterを１足す

    def train(self):
        for i in range(self.max_iter): #12000回まで
            self.train_step() #train_stepを実行

        test_acc = self.network.accuracy(self.x_test, self.t_test) #x_testとt_test(テストデータ)を実引数としたnetwork.accuracyの結果を代入

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


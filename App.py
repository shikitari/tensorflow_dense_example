# coding: utf-8
# import numpy as np
from Data import Data
from TwoDenseOperations import TwoDenseOperations
from TfSession import TfSession
from Plotter import Plotter


class App:
    EPOCH_SIZE = 200
    BATCH_SIZE = 30
    NUM_OF_CLASSES = 5
    NUM_OF_EACH_CLASS_DATA = 100
    NUM_OF_DIMENSIONS = 3

    def __init__(self):
        self.d = None # data manager
        self.o = None # operations manager
        self.t = None # tensorflow session manager

    def init(self):
        """
        データ生成、オペレーションの定義、セッションの準備
        :return:
        """
        self.d = Data(num_of_each_class_data=self.NUM_OF_EACH_CLASS_DATA, num_of_classes=self.NUM_OF_CLASSES,
                      batch_size=self.BATCH_SIZE)

        # オペレーション(計算グラフ)
        self.o = TwoDenseOperations(input_size=self.NUM_OF_DIMENSIONS, hidden_unit_size=5, output_size=self.NUM_OF_CLASSES, learn_rate=1.0)

        # セッション
        self.t = TfSession(self.o)

    def train(self):
        """
        訓練
        :return:
        """
        if self.d is None or self.o is None or self.t is None:
            raise Exception()

        self.print_weight_bias()

        for i in range(self.EPOCH_SIZE):
            self.d.reset_sequence()
            self.d.shuffle()

            if i == 0 or i % 10 == 0:
                feed_dict = self.o.create_feed_dict(self.d.inputs[0:self.BATCH_SIZE], self.d.onehot_labels[0:self.BATCH_SIZE])
                loss_value = self.t.calculate_loss_value(feed_dict)
                print('loss value: {0:0.3f}'.format(loss_value))

            while True:
                inputs, onehot_labels = self.d.next_batch()
                if inputs is None:
                    break

                feed_dict = self.o.create_feed_dict(inputs, onehot_labels)
                self.t.train(feed_dict)

        self.print_weight_bias()

    def predict(self):
        """
        今回は、訓練データとテストデータを分けず。
        :return:
        """
        feed_dict = self.o.create_feed_dict(self.d.inputs_org, self.d.onehot_labels_org)
        return self.t.predict(feed_dict)

    def plot(self, predict):
        """
        :param predict: トレーニングしたモデルで予測した結果。one_hot表現ではなく、確率が入っている。
        :return:
        """
        p = Plotter()
        p.plot_and_show(self.d.inputs_org, predict, self.NUM_OF_EACH_CLASS_DATA, self.NUM_OF_CLASSES)

    def print_weight_bias(self):
        """
        隠れ層の重みとバイアスを表示する。
        :return:
        """
        print('weight: {}'.format(self.t.weight()))
        print('bias: {}'.format(self.t.bias()))


class Main:
    def __init__(self):
        app = App()
        app.init()
        app.train()
        predict = app.predict()
        app.plot(predict)


Main()
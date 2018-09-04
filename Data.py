# coding: utf-8
"""
3Dの渦データを作る。
以下の2Dの渦を3Dの渦に変更。
https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/dataset/spiral.py
"""
import numpy as np


class Data:
    DIMENSIONS = 3
    MAX_RADIUS = 1.0
    NOISE_VALUE = 0.08

    def __init__(self, num_of_each_class_data, num_of_classes, batch_size):
        """
        3Dの渦データを作る
        :param num_of_each_class_data:
        :param num_of_classes:
        :param batch_size:
        """
        self.num_of_each_class_data = num_of_each_class_data
        self.num_of_classes = num_of_classes
        self.inputs = None
        self.inputs_org = None
        self.onehot_labels = None
        self.onehot_labels_org = None
        self.sequence = 0
        self.batch_size = batch_size
        self.create_data()

    def create_data(self, seed=2018):
        np.random.seed(seed)
        num_of_each_class_data = self.num_of_each_class_data
        num_of_classes = self.num_of_classes

        x = np.zeros((num_of_each_class_data * num_of_classes, self.DIMENSIONS), dtype=np.float32)
        t = np.zeros((num_of_each_class_data * num_of_classes, num_of_classes), dtype=np.int)

        for index_class in range(num_of_classes):
            for index_data in range(num_of_each_class_data):
                angle_range = ((2.0 * np.pi) / num_of_classes) + (np.random.randn() * self.NOISE_VALUE)
                index_data_rate = index_data / num_of_each_class_data
                radius = self.MAX_RADIUS * index_data_rate
                theta = (index_class * angle_range) + (angle_range * index_data_rate) + (np.random.randn() * self.NOISE_VALUE)
                phi = self.MAX_RADIUS * index_data_rate * (np.pi / 2.0) + (np.random.randn() * self.NOISE_VALUE)

                index = num_of_each_class_data * index_class + index_data
                pos_x = radius * np.sin(theta)
                pos_y = radius * np.cos(theta)
                pos_z = np.sin(phi)
                x[index] = np.array([pos_x, pos_y, pos_z]).flatten()
                t[index, index_class] = 1

        self.inputs = x
        self.inputs_org = x
        self.onehot_labels = t
        self.onehot_labels_org = t

    def fetch_data_by_class(self, index_of_class):
        s = index_of_class * self.num_of_each_class_data
        e = (index_of_class + 1) * self.num_of_each_class_data
        return self.inputs[s:e, 0], self.inputs[s:e, 1]

    def reset_sequence(self):
        self.sequence = 0

    def shuffle(self):
        i = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[i]
        self.onehot_labels = self.onehot_labels[i]

    def next_batch(self):
        n = len(self.inputs)

        if (self.sequence + 1) * self.batch_size > n:
            return None, None

        s = self.sequence * self.batch_size
        e = (self.sequence + 1) * self.batch_size
        # print('slided by ({0}:{1})'.format(s, e))
        self.sequence += 1
        return self.inputs[s:e], self.onehot_labels[s:e]


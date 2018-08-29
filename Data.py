import numpy as np


class Data:
    def __init__(self, num_of_each_class_data, dimension, num_of_classes, batch_size):
        self.num_of_each_class_data = num_of_each_class_data
        self.dimensions = dimension
        self.num_of_classes = num_of_classes
        self.inputs = None
        self.inputs_org = None
        self.onehot_labels = None
        self.onehot_labels_org = None
        self.labels = None
        self.create_data()
        self.sequence = 0
        self.batch_size = batch_size

    def create_data(self, seed=2018):
        np.random.seed(seed)
        num_of_each_class_data = self.num_of_each_class_data
        dimensions = self.dimensions
        num_of_classes = self.num_of_classes

        x = np.zeros((num_of_each_class_data * num_of_classes, dimensions), dtype=np.float32)
        t = np.zeros((num_of_each_class_data * num_of_classes, num_of_classes), dtype=np.int)

        for j in range(num_of_classes):
            for i in range(num_of_each_class_data):
                rate = i / num_of_each_class_data
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

                ix = num_of_each_class_data * j + i
                x[ix] = np.array([radius * np.sin(theta),
                                  radius * np.cos(theta)]).flatten()
                t[ix, j] = 1

        self.inputs = x
        self.inputs_org = x
        self.onehot_labels = t
        self.onehot_labels_org = t
        # self.labels = np.argmax(t, axis=1)

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

    def next(self):
        n = len(self.inputs)

        if (self.sequence + 1) * self.batch_size > n:
            return None, None

        s = self.sequence * self.batch_size
        e = (self.sequence + 1) * self.batch_size
        # print('slided by ({0}:{1})'.format(s, e))
        self.sequence += 1
        return self.inputs[s:e], self.onehot_labels[s:e]


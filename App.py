import numpy as np
from Data import Data
from TwoDenseOperations import TwoDenseOperations
from Training import Training


class App:
    EPOCH_SIZE = 300
    BATCH_SIZE = 30

    def __init__(self):
        d = Data(num_of_each_class_data=100, dimension=2, num_of_classes=3, batch_size=self.BATCH_SIZE)
        o = TwoDenseOperations()
        t = Training(o)

        self.d = d
        self.o = o
        self.t = t

        self.print_weight_bias()

        for i in range(self.EPOCH_SIZE):
            d.reset_sequence()
            d.shuffle()

            if i == 0 or i % 10 == 0:
                feed_dict = o.create_feed_dict(d.inputs[0:self.BATCH_SIZE], d.onehot_labels[0:self.BATCH_SIZE])
                loss_value = t.calculate_loss_value(feed_dict)
                print('loss value: {0:0.3f}'.format(loss_value))

            while True:
                inputs, onehot_labels = d.next()
                if inputs is None:
                    break

                feed_dict = o.create_feed_dict(inputs, onehot_labels)
                t.train(feed_dict)

        self.print_weight_bias()

    def print_weight_bias(self):
        print('weight: {}'.format(self.t.weight()))
        print('bias: {}'.format(self.t.bias()))

App()

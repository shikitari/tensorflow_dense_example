import tensorflow as tf


class Training:
    def __init__(self, operations):
        self.operations = operations
        self.sess = tf.Session(graph=operations.graph)
        self.sess.run(operations.init)

    def calculate_loss_value(self, feed_dict):
        return self.sess.run(self.operations.loss_value, feed_dict=feed_dict)

    def train(self, feed_dict):
        self.sess.run(self.operations.training, feed_dict=feed_dict)

    def weight(self):
        return self.sess.run(self.operations.dense1_weight)

    def bias(self):
        return self.sess.run(self.operations.dense1_bias)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

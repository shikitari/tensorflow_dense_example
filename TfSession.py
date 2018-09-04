# coding: utf-8
"""
TensorFlowのセッションを管理する
"""
import tensorflow as tf


class TfSession:
    def __init__(self, operations):
        self.operations = operations
        self.sess = tf.Session(graph=operations.graph)
        self.sess.run(operations.init)

    def calculate_loss_value(self, feed_dict):
        return self.sess.run(self.operations.loss_value, feed_dict=feed_dict)

    def train(self, feed_dict):
        self.sess.run(self.operations.train, feed_dict=feed_dict)

    def predict(self, feed_dict):
        return self.sess.run(self.operations.predict, feed_dict=feed_dict)

    def weight(self):
        return self.sess.run(self.operations.dense1_weight)

    def bias(self):
        return self.sess.run(self.operations.dense1_bias)

    def close(self):
        self.sess.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

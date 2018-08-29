import re
import tensorflow as tf


class TwoDenseOperations:
    DENSE1_NAME = "dense1"

    def __init__(self, input_size=2, hidden_unit_size=10, output_size=3, learn_rate=1.0):
        graph = tf.Graph()
        graph.seed = 2018

        with graph.as_default():
            inputs = tf.placeholder(tf.float32, (None, input_size), name='inputs')

            onehot_labels = tf.placeholder(tf.float32, (None, output_size), name='onehot_labels')

            dense1 = tf.layers.dense(inputs=inputs, units=hidden_unit_size, activation=tf.nn.sigmoid,
                                     name=self.DENSE1_NAME)

            logits = tf.layers.dense(inputs=dense1, units=output_size, name='logits')

            loss_value = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

            training = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_value)

            init = tf.global_variables_initializer()

        self.graph = graph
        self.loss_value = loss_value
        self.init = init
        self.training = training
        self.placeholder_names = {
            'inputs': inputs.name,
            'onehot_labels': onehot_labels.name
        }

        self.dense1_weight = None
        self.dense1_bias = None
        self.peep_into_automatically_generated_operations(self.DENSE1_NAME)

    def create_feed_dict(self, inputs, onehot_labels):
        p = self.placeholder_names
        return {p['inputs']: inputs, p['onehot_labels']: onehot_labels}

    def peep_into_automatically_generated_operations(self, target_name):
        for i, v in enumerate(self.graph.get_operations()):
            if re.match(r'Variable(V2)?$', v.type):
                # print("{1:20s}:{0}".format(v.name, v.type))
                pass

            if re.match(r'^' + target_name + r'.+kernel$', v.name):
                self.dense1_weight = v.outputs[0]
            elif re.match(r'^' + target_name + r'.+bias$', v.name):
                self.dense1_bias = v.outputs[0]

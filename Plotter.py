# coding: utf-8
# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
from Data import Data


class Plotter:
    def __init__(self):
        pass

    def plot_and_show(self, data, predict, num_of_each_class_data, num_of_classes, colorize=True):
        """
        :param numpy.ndarray data:
        :param numpy.ndarray predict:
        :param int num_of_each_class_data:
        :param int num_of_classes:
        :param bool colorize:
        :return:
        """
        fig = plt.figure()
        ax = Axes3D(fig)

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        ax.set_xlim((-1.0, 1.0))
        ax.set_ylim((-1.0, 1.0))
        ax.set_zlim((0.0, 1.0))

        # 予測の精度が低い（確率が低い）ものほど色を暗くする
        colors = [[0 for i in range(num_of_each_class_data)] for j in range(num_of_classes)]
        for j in range(num_of_classes):
            for i in range(num_of_each_class_data):
                index = j * num_of_each_class_data + i
                if colorize:
                    colors[j][i] = colorsys.hsv_to_rgb(j / num_of_classes * 0.7, 1, np.max(predict[index]))
                else:
                    colors[j][i] = colorsys.hsv_to_rgb(j / num_of_classes * 0.7, 1, 1)

        for index_classes in range(num_of_classes):
            s = index_classes * num_of_each_class_data
            e = (index_classes + 1) * num_of_each_class_data
            ax.scatter(x[s:e], y[s:e], z[s:e], color=colors[index_classes], depthshade=False, zorder=10)

        plt.show()

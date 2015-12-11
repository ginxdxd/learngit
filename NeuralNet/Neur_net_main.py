# -*- coding: utf-8 -*-
# -*- coding: gbk -*-
#!usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : Neur_net_main.py
    @time : 9/12/15
"""

from data import get_titanic_data
from network import neuralClass
import random
import math
import numpy as np

N_LAYER = 1
STEPS = 5
LAMBDA = 0.0
LEARNING_RATE = 0.0127

def main():
    # training_set, training_label, test_set, valid_set, valid_label = get_titanic_data.load_data()
    training_set = np.array([[0,0], [0,1], [1,0], [1,1]])
    training_label = np.array([0,1,1,0])
    n_item, n_input = training_set.shape
    if len(training_label.shape) == 1:
        n_output = 1
    else:
        n_output = training_label.shape[1]
    neural_net = neuralClass.NeuralNet(n_input, N_LAYER, n_input, n_output, LAMBDA, LEARNING_RATE)

    train(neural_net, training_set, training_label, n_item)
    # result = neural_net.predict(test_set)
    result = neural_net.predict(training_set)
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    print result

def train(neural_net, training_set, label, n_item):  # 向量的迭代器形式是元素的type
    index = range(n_item)
    for i in range(STEPS):
        random.shuffle(index)  # 因为使用随机梯度下降法，需要把样本打乱
        for item in index:
            neural_net.train(training_set[index[item]], [label[index[item]]])
    total_error = 0.0
    weight_sum = neural_net.weight_sum()
    for item in range(n_item):
        total_error += neural_net.calculate_error(training_set[item], [label[item]])
    total_error = (total_error + LAMBDA * weight_sum * 0.5) / n_item
    print 'total error is %.3f' %(total_error)


if __name__ == '__main__':
    main()


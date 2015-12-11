# -*- coding: utf-8 -*-
# -*- coding: gbk -*-
#!usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : neuralClass.py
    @time : 3/11/15
"""
"""
    参考资料：
    1.http://blog.csdn.net/xiaowei_cqu/article/details/9023247
    2.http://m.blog.csdn.net/blog/arsenicer/43604353
    3.《集体智慧编程》 P74-P83
"""

import random
import math
import numpy as np

class NeuralNet:
    def __init__(self, n_inputs, n_layers, n_neurons, n_outputs, lamda, learning_rate,
                 input_layer_bias=None, hidden_layer_bias=None, output_layer_bias=None):

        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.LAMBDA = lamda if lamda else 0.0
        self.LEARNING_RATE = learning_rate if learning_rate else 0.1

        self.input_layer = Layer(n_inputs, input_layer_bias)
        self.output_layer = Layer(n_outputs, output_layer_bias)
        self.hidden_layers = []
        for i in range(self.n_layers):
            self.hidden_layers.append(Layer(n_neurons, hidden_layer_bias))

        self.init_weights_inp_layer()
        self.init_weights_hid_layer()

    def init_weights_inp_layer(self):
        for i in range(len(self.input_layer.neuron)):
            for j in range(self.n_neurons):
                self.input_layer.neuron[i].weights.append(rand(-0.5, 0.5))

    def init_weights_hid_layer(self):
        for l in range(len(self.hidden_layers)-1):  # 不考虑倒数第一层隐藏层的权值初始化，因为那是跟输出层相关
            for i in range(len(self.hidden_layers[l].neuron)):
                for j in range(self.n_neurons):
                    self.hidden_layers[l].neuron[i].weights.append(rand(-0.5, 0.5))

        for i in range(len(self.hidden_layers[-1].neuron)):  # 倒数第一层的隐藏层
            for j in range(self.n_outputs):
                self.hidden_layers[-1].neuron[i].weights.append(rand(-0.5, 0.5))

    def train(self, training_input, training_output):
        self.forward_propagation(training_input)
        self.back_propagation(training_output)
        self.update_weights(self.LAMBDA, self.LEARNING_RATE)

    def forward_propagation(self, inputs):  # 此处inputs为某一行数据
        for i in range(len(self.input_layer.neuron)-1):
            self.input_layer.neuron[i+1].value = inputs[i]

        self.hidden_layers[0].single_layer_forward_pro(self.input_layer)    # 从输入层到第一隐藏层

        for i in range(self.n_layers - 1):                     # 两个隐藏层间
            self.hidden_layers[i+1].single_layer_forward_pro(self.hidden_layers[i])

        self.output_layer.single_layer_forward_pro(self.hidden_layers[-1])  # 倒数第一个隐藏层到输出层

    def back_propagation(self, outputs):
        for i in range(len(outputs)):    # 输出层的delta
            self.output_layer.neuron[i+1].output_layer_back_propagation(outputs[i])

        for ne in range(len(self.hidden_layers[-1].neuron)-1):  # 倒数第一层隐藏层的delta
            self.hidden_layers[-1].neuron[ne+1].layer_back_propagation(self.output_layer)

        for h in range(self.n_layers - 1):  # 其他隐藏层的delta
            for ne in range(len(self.hidden_layers[self.n_layers-2-h].neuron)-1):
                self.hidden_layers[self.n_layers-2-h].neuron[ne+1].layer_back_propagation(self.hidden_layers[self.n_layers-h-2+1])

    def update_weights(self, lamda, learning_rate):
        # 更新最后一层隐藏层的权值
        self.hidden_layers[-1].neuron[0].layer_update_weights_bias(self.output_layer, learning_rate)  # 更新bias神经元权重
        for o in range(len(self.hidden_layers[-1].neuron)-1):
            self.hidden_layers[-1].neuron[o+1].layer_update_weights(self.output_layer, lamda, learning_rate)

        for h in range(self.n_layers-1):  # 对于隐藏层的某一层
            self.hidden_layers[h].neuron[0].layer_update_weights_bias(self.hidden_layers[h+1], learning_rate)  # 更新此层bias神经元权重
            for ne in range(len(self.hidden_layers[h].neuron)-1):  # 更新此层的某个神经元的权值
                self.hidden_layers[h].neuron[ne+1].layer_update_weights(self.hidden_layers[h+1], lamda, learning_rate)

        self.input_layer.neuron[0].layer_update_weights_bias(self.hidden_layers[0], learning_rate)  # 更新bias神经元
        for i in range(len(self.input_layer.neuron)-1):  # 更新输入层的某个神经元的权值
            self.input_layer.neuron[i+1].layer_update_weights(self.hidden_layers[0], lamda, learning_rate)

    def calculate_error(self, training_input, training_output):
        self.forward_propagation(training_input)
        total_error = 0.0
        for o in range(len(training_output)):
            total_error += self.output_layer.neuron[o+1].calculate_single_error(training_output[o])
        return total_error

    def weight_sum(self):  # 权值的平方和，用来做正则化
        weight_sum = 0.0
        for i in range(len(self.input_layer.neuron)-1):
            weight_sum += self.input_layer.neuron[i+1].cal_weight_sum()  # 单个神经元的权值平方和
        for h in range(self.n_layers):
            for ne in range(len(self.hidden_layers[h].neuron)-1):
                weight_sum += self.hidden_layers[h].neuron[ne+1].cal_weight_sum()
        return weight_sum

    def predict(self, test_set):
        result = []
        for i in range(test_set.shape[0]):
            self.forward_propagation(test_set[i])
            for o in range(self.n_outputs):
                result.append(self.output_layer.neuron[o+1].value)
        result = np.array(result)
        return result
#  end class NeuralNet

class Neuron:
    def __init__(self, value=0.0):
        self.input = 0.0  # z
        self.value = value  # a
        self.delta = 0.0  # 误差项
        self.weights = []  # 权值列表

    def output_layer_back_propagation(self, outp):
        self.delta = (outp - self.value) * self.value * (1.0 - self.value)

    def layer_back_propagation(self, layer):
        delta_sum = 0.0
        for l in range(len(layer.neuron) - 1):
            delta_sum += self.weights[l] * layer.neuron[l+1].delta
        self.delta = delta_sum * self.value * (1.0 - self.value)

    def layer_update_weights(self, layer, lamda, learning_rate):
        for w in range(len(layer.neuron)-1):   # 对此神经元的 权值list 的元素 逐个更新
            self.weights[w] -= learning_rate * (layer.neuron[w+1].delta * self.value + lamda * self.weights[w])

    def layer_update_weights_bias(self, layer, learning_rate):
        for w in range(len(layer.neuron)-1):
            self.weights[w] -= learning_rate * layer.neuron[w+1].delta

    def calculate_single_error(self, outp):
        return 0.5 * (self.value - outp) ** 2

    def cal_weight_sum(self):  # 计算单个神经元的权值平方和
        weights_sqr = [x ** 2 for x in self.weights]
        return sum(weights_sqr)

#  end class Neuron

class Layer:
    def __init__(self, num_neuron, bias):
        self.bias = bias if bias else 1.0
        self.neuron = []
        bias_neuron = Neuron(self.bias)   # 首先构造一个bias神经元
        self.neuron.append(bias_neuron)

        for i in range(num_neuron):
            self.neuron.append(Neuron())


    def single_layer_forward_pro(self, layer):
        for h in range(len(self.neuron)-1):  # h for hidden layer
            for i in range(len(layer.neuron)):  # i for input layer
                self.neuron[h+1].input += layer.neuron[i].value * layer.neuron[i].weights[h]
            self.neuron[h+1].value = sigmoid_function(self.neuron[h+1].input)

#  end class Layer

def rand(a, b):
    return (b-a)*random.random() + a
#  end rand

def sigmoid_function(x):
    # return 1.0 / (1 + math.exp(-x))
    return 1.0 / (1 + np.exp(-x))

def d_sigmoid_function(x):
    return sigmoid_function(x) * (1.0 - sigmoid_function(x))
#  end sigmoid_function
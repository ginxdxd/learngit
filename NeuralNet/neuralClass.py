# -*- coding : utf-8 -*-
#!usr/bin/env python
__author__ = 'quanxiandeng'

import random
import math

class NeuralNet:

    def __init__(self, n_inputs, n_layers, n_neurons, n_outputs, \
                 input_layer_bias=None, hidden_layer_bias=None, output_layer_bias=None):

        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        # self.LEARNING_RATE = learning_rate if learning_rate else 0.5

        self.input_layer = Layer(n_inputs, input_layer_bias)
        self.hidden_layers = []
        for i in range(self.n_layers):
            self.hidden_layers.append(Layer(n_neurons, hidden_layer_bias))
        self.output_layer = Layer(n_outputs, output_layer_bias)

        self.init_weights_inp_layer()
        self.init_weights_hid_layer()

    def init_weights_inp_layer(self):
        for i in range(len(self.input_layer.neuron)):
            for j in range(self.n_neurons):
                self.input_layer.neuron[i].weights.append(rand(-0.5, 0.5))

    def init_weights_hid_layer(self):
        for l in range(len(self.hidden_layers)-1):
            for i in range(len(self.hidden_layers[l].neuron)):
                for j in range(self.n_neurons):
                    self.hidden_layers[l].neuron[i].weights.append(rand(-0.5, 0.5))

        for i in range(len(self.hidden_layers[-1].neuron)):
            for j in range(self.n_outputs):
                self.hidden_layers[-1].neuron[i].weights.append(rand(-0.5, 0.5))

    def train(self, training_input, training_output):
        self.forward_propagation(training_input)
        self.back_propagation(training_output)
        self.update_weights()

    def forward_propagation(self, inputs):  # 此处inputs为某一行数据
        for i in range(len(self.input_layer.neuron)-1):
            self.input_layer.neuron[i+1].value = inputs[i]

        self.hidden_layers[1].single_layer_forward_pro(self.input_layer)    # from input layer to the first hidden layer

        for i in range(self.n_layers - 1):                     # between hidden layers
            self.hidden_layers[i+1].single_layer_forward_pro(self.hidden_layers[i])

        self.output_layer.single_layer_forward_pro(self.hidden_layers[-1])  # from the last hidden layer to output layer

    def back_propagation(self, outputs):
        for i in range(len(outputs)):
            self.output_layer.neuron[i+1].output_layer_back_propagation(outputs[i])

        for ne in range(len(self.hidden_layers[-1].neuron)-1):
            self.hidden_layers[-1].neuron[ne+1].layer_back_propagation(self.output_layer)

        for h in range(self.n_layers - 1):
            for ne in range(len(self.hidden_layers[self.n_layers-1-h].neuron)-1):
                self.hidden_layers[self.n_layers-h-1].neuron[ne+1].layer_back_propagation(self.hidden_layers[self.n_layers-h-1+1])

    def update_weights(self):
        for i in range(len(self.input_layer.neuron)-1):
            self.input_layer.neuron[i+1].layer_update_weights(self.hidden_layers[0])

        for h in range(self.n_layers-1):
            for ne in range(len(self.hidden_layers[h].neuron)-1):
                self.hidden_layers[h].neuron[ne+1].layer_update_weights(self.hidden_layer[h+1])

        for o in range(len(self.hidden_layers[-1].neuron)-1):
            self.hidden_layers[-1].neuron[o+1].layer_update_weights(self.output_layer)

    def calculate_error(self, training_input, training_output):
        self.forward_propagation(training_input)
        total_error = 0.0
        for o in range(len(training_output)):
            total_error += self.output_layer.neuron[o+1].calculate_single_error(training_output[o])
        return total_error
#  end class NeuralNet

class Neuron:

    LEARNING_RATE = 0.01
    def __init__(self, value=0.0):  # input, value, weights, delta

        self.input = 0.0  # z
        self.value = value  # a
        self.delta = 0.0
        self.weights = []

    def output_layer_back_propagation(self, outp):
        self.delta = (self.value - outp) * self.value * (1.0 - self.value)

    def layer_back_propagation(self, layer):
        for l in range(len(layer.neuron) - 1):
            self.delta += self.weights[l] * layer.neuron[l+1].delta
        self.delta = self.delta * self.value * (1.0 - self.value)

    def layer_update_weights(self, layer):
        for w in range(len(layer.neuron)-1):
            self.weights[w] -= self.LEARNING_RATE * (layer.neuron[w+1].delta * self.value)

    def calculate_single_error(self, outp):
        return 0.5 * (self.value - outp) ** 2
#  end class Neuron

class Layer:
    def __init__(self, num_neuron, bias):
        self.bias = bias if bias else 1.0
        self.neuron = []
        bias_neuron = Neuron(self.bias)
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
    return 1.0 / (1 + math.exp(-x))
#  end sigmoid_function
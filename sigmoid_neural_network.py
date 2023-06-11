import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def differential_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mean_square_error_half(y_predicted, y_true):
    return (1 / 2) * ((y_predicted - y_true) ** 2)


class Neuron:

    def __init__(self, num_elem):
        self.weights = []
        self.number_of_inputs = num_elem
        self.bias = 1.0
        for i in range(self.number_of_inputs):
            self.weights.append(1.0)
        self.next = None
        self.output = 0.0
        self.de_dout = 0.0

    def feed_forward(self, input_list):
        if len(input_list) != len(self.weights):
            raise ValueError("length of input values incompatible with number of weights")
        feed = 0
        for a_, b_ in zip(self.weights, input_list):
            feed += a_ * b_
        feed += self.bias
        self.output = feed


class NeuralNetwork:
    epochs = 0

    def __init__(self, number_of_hidden_layers, neurons_in_a_layer, number_of_inputs, number_of_epochs=100,
                 learn_rate=0.1):
        if number_of_hidden_layers < 1 or neurons_in_a_layer < 1 or number_of_inputs < 1:
            raise ValueError("Invalid Parameter Values")
        self.layers = number_of_hidden_layers
        self.epochs = number_of_epochs
        self.number_neurons = neurons_in_a_layer
        self.num_inputs = number_of_inputs
        self.neuron_list = []
        self.learn_rate = learn_rate
        self.outputs = []

        # writing the first layer of neurons, each with inputs to match the number of inputs
        self.layer_one = []
        for i in range(self.number_neurons):
            self.layer_one.append(Neuron(number_of_inputs))
        self.neuron_list.append(self.layer_one)

        # writing the remaining layers, each with inputs to match the number of neurons in previous layer
        for _a in range(1, self.layers):
            other_layers = []
            for _b in range(self.number_neurons):
                other_layers.append(Neuron(self.number_neurons))
            self.neuron_list.append(other_layers)

        # adding the output layer with one neuron
        self.neuron_list.append(Neuron(self.number_neurons))

    def feed_forward(self, input_list):
        if len(input_list) != self.num_inputs:
            raise ValueError("number of inputs incompatible with the list of inputs")

        # feed input to the first hidden layer and store outputs
        output_layer_one = []
        for node in self.neuron_list[0]:
            node.feed_forward(input_list)
            output_layer_one.append(sigmoid(node.output))
        self.outputs.append(output_layer_one)

        # feed outputs from previous iteration to the next layer
        for layer in range(1, self.layers):
            output_remaining_layers = []
            for node in self.neuron_list[layer]:
                node.feed_forward(self.outputs[layer - 1])
                output_remaining_layers.append(sigmoid(node.output))
            self.outputs.append(output_remaining_layers)

        # feed outputs from last layer to the output neuron
        self.neuron_list[-1].feed_forward(self.outputs[-1])

    def train_network(self, data, y_true):
        for iteration in range(self.epochs):
            for x, y in zip(data, y_true):

                # check if all the data is valid every loop
                if len(x) != self.num_inputs:
                    raise ValueError("data lacks the number of inputs at", x, " , ", y)

                # feed the network
                # find sigmoid equivalents
                _sigmoid_x = []
                for _data in x:
                    _sigmoid_x.append(sigmoid(_data))
                self.feed_forward(_sigmoid_x)

                # backpropagation
                # starting with output layer
                y_predicted = sigmoid(self.neuron_list[-1].output)
                y_sigmoid = sigmoid(y)
                de_dw = []
                self.neuron_list[-1].de_dout = y_predicted - y_sigmoid
                for out_last in self.outputs[-1]:
                    # calculate derivatives for each weight
                    de_dw.append(self.neuron_list[-1].de_dout * y_predicted * (1 - y_predicted) * out_last)

                for i in range(len(de_dw)):
                    self.neuron_list[-1].weights[i] -= self.learn_rate * de_dw[i]

                # optimising bias
                self.neuron_list[-1].bias -= self.learn_rate * (y_predicted - y_sigmoid) * y_predicted * (
                        1 - y_predicted)

                # continue with last hidden layer and backwards
                for i in range(self.layers - 1, -1, -1):
                    for node, _a_ in zip(self.neuron_list[i], range(self.number_neurons)):
                        if i == 0:
                            number_of_weights = self.num_inputs
                        else:
                            number_of_weights = self.number_neurons

                        # getting output of the node
                        _out = self.outputs[i][_a_]

                        # getting output of the parallel node in the next layer
                        _next_out = (self.neuron_list[-1].output if i == self.layers - 1
                                     else self.neuron_list[i + 1][_a_].output)

                        # getting the weight for the output of this node in the next node
                        _next_weight = (self.neuron_list[-1].weights[_a_] if i == self.layers - 1
                                        else self.neuron_list[i + 1][_a_].weights[_a_])

                        for weight, _b_ in zip(node.weights, range(number_of_weights)):
                            # singling weights one by one
                            # getting output of the parallel node in the previous layer
                            _previous_out = (sigmoid(x[_b_]) if i == 0 else self.outputs[i - 1][_b_])
                            _de_dout = (self.neuron_list[-1].de_dout if i == self.layers - 1
                                        else self.neuron_list[i + 1][_a_].de_dout) * \
                                       _next_out * (1 - _next_out) * _next_weight
                            _de_dw = _de_dout * (_out * (1 - _out)) * _previous_out
                            node.de_dout = _de_dout

                            # optimising weight
                            node.weights[_b_] -= self.learn_rate * _de_dw

                        # optimising bias
                        de_dout = (self.neuron_list[-1].de_dout if i == self.layers - 1
                                   else self.neuron_list[i + 1][-1].de_dout) * _next_out * \
                                  (1 - _next_out) * _next_weight

                        de_db = de_dout * (_out * (1 - _out))
                        node.bias -= self.learn_rate * de_db

                        # one iteration for training for one node complete

    def feed_forward_and_output(self, data):
        _sigmoid_data = []
        for i in data:
            _sigmoid_data.append(sigmoid(i))
        self.feed_forward(_sigmoid_data)
        output = self.neuron_list[-1].output
        print(output)

        return np.log(1 / output - 1)

'''
brain = NeuralNetwork(5, 5, 3)
data_set = [[45, 45, 45], [45, 90, 45], [135, 90, 45], [180, 270, 90], [45, 90, 180], [90, 180, 270], [270, 45, 135], [0, 45, 90], [0, 0, 45], [45, 0, 90], [135, 90, 180], [360, 270, 315], [315, 225, 180], [45, 315, 90], [135, 315, 45], [0, 0, 90], [0, 0, 135], [0, 0, 180], [0, 0, 225], [0, 90, 270], [45, 135, 270], [135, 180, 315], [315, 315, 315], [45, 45, 45], [90, 90, 90], [135, 135, 135], [45, 45, 0]]
processed_data = []
for a, b, c in data_set:
    processed_data.append([float(a) / 360.0, float(b) / 360.0, float(c) / 360.0])
true_y = [5.8383462102392105, 5.967625627438528, 2.9248059795206984, -3.7279395668639266, 6.1299232769728995, -4.1660874004012145, 4.081359608689336, 2.549828984267779, 4.234991659001395, -0.9522376116991167, 3.0871036290550693, 1.3891849701456893, 0.9408358753118744, 1.2831621819731218, 2.4782157823912634, -0.0028815893345761623, 1.5598851338219715, 4.3972893085357665, 0.2917906874111913, 3.81550772986029, 0.4492677750060805, -1.9109285849514825, 6.053935986994301, 5.8383462102392105, -0.21703883079124298, -2.1671844781483944, 3.603354551237815]
processed_y = []
for a in true_y:
    processed_y.append(a / 10)
brain.train_network(processed_data, processed_y)
print(brain.outputs)
print(brain.feed_forward_and_output([135.0 / 360.0, 135.0 / 360.0, 0.0]) * 10)
'''
print(sigmoid(0.2675))


import numpy as np
from scipy.stats import logistic


class Layer:
    def __init__(self):
        self.W = []  # self.W = the incoming weights
        self.b = []  # self.b = the biases
        self.a = []  # self.a = the activations
        self.z = []  # self.z = the outputs
        self.d_W = []  # self.d_W = the gradient of the incoming weights
        self.d_b = []  # self.d_b = the gradient of the biases
        self.d_a = []  # self.d_a = the gradient of the activations
        self.d_z = []  # self.d_z = the gradient of the outputs


class MLP(Layer):  # Multi Layer Perceptron
    def __init__(self, neurons_per_layer):

        super().__init__()

        self.layer = {}

        for i in range(0, len(neurons_per_layer) - 1):
            self.layer[i] = Layer()
            self.layer[i].W = (10 ** (-1)) * np.random.randn(neurons_per_layer[i + 1], neurons_per_layer[i])
            self.layer[i].b = np.zeros((1, neurons_per_layer[i + 1]))
            self.layer[i].a = np.zeros((1, neurons_per_layer[i + 1]))
            self.layer[i].z = np.zeros((1, neurons_per_layer[i + 1]))
            self.layer[i].d_W = np.zeros((neurons_per_layer[i + 1], neurons_per_layer[i]))
            self.layer[i].d_b = np.zeros((1, neurons_per_layer[i + 1]))
            self.layer[i].d_a = np.zeros((1, neurons_per_layer[i + 1]))
            self.layer[i].d_z = np.zeros((1, neurons_per_layer[i + 1]))

        self.losses = []
        self.accuracies = []

    @staticmethod
    def sigmoid(a):
        return np.array(logistic.cdf(a))

    @staticmethod
    def d_sigmoid(a):
        return MLP.sigmoid(a) * (1 - MLP.sigmoid(a))

    def forward(self, x):
        for layer in self.layer.values():
            z = layer.b + x @ layer.W.T
            x = MLP.sigmoid(z)[:]
        y_hat = x

        return y_hat

    @staticmethod
    def loss(y_hat, y):
        return 1 / (2 * y_hat.shape[0]) * np.sum(np.sum((y_hat - y) ** 2))

    @staticmethod
    def accuracy(y_hat, y):
        predict = np.zeros(y_hat.shape)
        for i, values in enumerate(y_hat):
            predict[i][np.argmax(values)] = 1

        acc = (np.sum(predict == y) - y.shape[0]) / (2 * y.shape[0])

        return acc

    def backpropagation(self, x, y, y_hat, learning_rate):
        self.layer[0].a = self.layer[0].b + x @ self.layer[0].W.T
        self.layer[0].z = MLP.sigmoid(self.layer[0].a)
        for i in range(1, len(self.layer)):
            self.layer[i].a = self.layer[i].b + self.layer[i - 1].z @ self.layer[i].W.T
            self.layer[i].z = MLP.sigmoid(self.layer[i].a)

        delta_L = y_hat - y
        last = len(self.layer) - 1
        self.layer[last].d_z = delta_L
        self.layer[last].d_a = delta_L * MLP.d_sigmoid(self.layer[last].a)
        self.layer[last].d_W = self.layer[last].d_a.T @ self.layer[last - 1].z
        self.layer[last].d_b = self.layer[last].d_a

        for i in range(1, len(self.layer) - 1):
            delta_l = self.layer[last - i + 1].d_a @ self.layer[last - i + 1].W
            self.layer[last - i].d_z = delta_l
            self.layer[last - i].d_a = delta_l * MLP.d_sigmoid(self.layer[last - i].a)
            self.layer[last - i].d_W = self.layer[last - i].d_a.T @ self.layer[last - i - 1].z
            self.layer[last - i].d_b = self.layer[last - i].d_a

        delta_l = self.layer[1].d_a @ self.layer[1].W
        self.layer[0].d_z = delta_l
        self.layer[0].d_a = delta_l * MLP.d_sigmoid(self.layer[0].a)
        self.layer[0].d_W = self.layer[0].d_a.T @ x
        self.layer[0].d_b = self.layer[0].d_a

        for layer in self.layer.values():
            layer.W = layer.W - learning_rate * layer.d_W
            layer.b = layer.b - learning_rate * layer.d_b

    def training(self, x, y, learning_rate, num_epochs, verbose=False, print_every_k=1):
        accuracy = []
        loss = []

        for epoch in range(num_epochs):

            shuffle = np.random.permutation(range(x.shape[0]))
            x_shuffled = x[shuffle, :]
            y_shuffled = y[shuffle, :]

            for sample in range(x.shape[0]):
                y_hat = self.forward(x_shuffled[sample, :])
                self.backpropagation(x_shuffled[sample, :].reshape(1, x.shape[1]), y_shuffled[sample, :], y_hat,
                                     learning_rate)

            Y_hat = self.forward(x)
            loss.append(MLP.loss(Y_hat, y))
            accuracy.append(MLP.accuracy(Y_hat, y))

            if verbose and not epoch % print_every_k == 0:
                print('Epoch %d : loss = %.5e, accuracy = %.2f %%' % (epoch, loss[epoch], 100 * accuracy[epoch]))

        self.losses = loss
        self.accuracies = accuracy

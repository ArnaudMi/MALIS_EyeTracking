import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import MLP
from LoadData import load_data

directory = "Data/img"
training_length = 200

X, y = load_data(directory)

print(X.shape, y.shape)

x_train = X[:training_length, :]
y_train = y[:training_length, :]
x_test = X[training_length:, :]
y_test = y[training_length:, :]

# Neural Network (NN) parameters
epochs = 3400
learning_rate = 0.001
verbose = True
print_every_k = 100

# Initialization of the NN
NN1 = MLP([4, 10, 3])
print('TRAINING')
# Training
NN1.training(x_train, y_train, learning_rate, epochs, verbose, print_every_k)
# Compute the training loss and accuracy after having completer the training
y_hat = NN1.forward(x_train)
print('final : loss = %.3e , accuracy = %.2f %%' % (MLP.loss(y_hat, y_train), 100 * MLP.accuracy(y_hat, y_train)))

# Test
print('\nTEST')
y_hat = NN1.forward(x_test)
print('loss = %.3e , accuracy = %.2f %%\n' % (MLP.loss(y_hat, y_test), 100 * MLP.accuracy(y_hat, y_test)))

plt.plot(list(range(epochs)), NN1.losses, c='r', marker='o', ls='--')
plt.title("Training Loss")
plt.xlabel("epochs")
plt.ylabel("loss value")
plt.show()

plt.plot(list(range(epochs)), NN1.accuracies, c='g', marker='o', ls='--')
plt.title("Training accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()

'''
2-layer neural network for XOR
y is expected output
y_hat is the predicted output
'''
import numpy as np

# relu activation function


def Relu(z):
    return z * (z > 0)

# sigmoide activation function


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# Dummy Data
x1 = np.array([1, 0, 1, 0])
x2 = np.array([0, 0, 1, 1])
y = np.array([1, 0, 0, 1])

# initializing random weights and biases to a random number
w11 = np.random.randn()*0.01
w21 = np.random.randn()*0.01
w12 = np.random.randn()*0.01
w22 = np.random.randn()*0.01
w13 = np.random.randn()*0.01
w23 = np.random.randn()*0.01
w14 = np.random.randn()*0.01
w24 = np.random.randn()*0.01
w34 = np.random.randn()*0.01
b1 = np.random.randn()*0.01
b2 = np.random.randn()*0.01
b3 = np.random.randn()*0.01
b4 = np.random.randn()*0.01


epochs = 10000
inputs = len(x1)
lr = 0.1


def forwardpass(a1, a2, a3):
    z4 = w14 * a1 + w24 * a2 + w34 * a3 + b4
    y_hat = sigmoid(z4)
    return y_hat


for epoch in range(epochs):
    total_loss = 0
    for i in range(inputs):
        # implement normal forward pass
        z1 = w11*x1[i] + w21*x2[i] + b1
        z2 = w12*x1[i] + w22*x2[i] + b2
        z3 = w13*x1[i] + w23*x2[i] + b3

        a1 = Relu(z1)
        a2 = Relu(z2)
        a3 = Relu(z3)

        y_hat = forwardpass(a1, a2, a3)

        # compute binary-cross entropy loss
        loss = -(y[i] * np.log(y_hat) + (1-y[i]) * np.log(1-y_hat))

        # implement Back propagation and gradient descent
        # getting the derivative of loss with respect to z4 using chain rule
        dL_dy = (y_hat - y[i])
        dy_dz = y_hat * (1 - y_hat)
        dL_dz4 = dL_dy * dy_dz

        # gradient for output layer weights and bias
        dL_dw14 = dL_dz4 * a1
        dL_dw24 = dL_dz4 * a2
        dL_dw34 = dL_dz4 * a3
        dL_db4 = dL_dz4 + 0

        # backpropagating into hidden layer
        # neuron 1
        dL_da1 = dL_dz4 * w14
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dw11 = dL_dz1 * x1[i]
        dL_dw21 = dL_dz1 * x2[i]
        dL_db1 = dL_dz1
        # neuron 2
        dL_da2 = dL_dz4 * w24
        dL_dz2 = dL_da2 * (z2 > 0)
        dL_dw12 = dL_dz2 * x1[i]
        dL_dw22 = dL_dz2 * x2[i]
        dL_db2 = dL_dz2

        # neuron 3
        dL_da3 = dL_dz4 * w34
        dL_dz3 = dL_da3 * (z3 > 0)
        dL_dw13 = dL_dz3 * x1[i]
        dL_dw23 = dL_dz3 * x2[i]
        dL_db3 = dL_dz3

        # update parameters using gradient descent
        w14 = w14 - lr * dL_dw14
        w24 = w24 - lr * dL_dw24
        w34 = w34 - lr * dL_dw34
        b4 = b4 - lr * dL_db4

        w11 = w11 - lr * dL_dw11
        w21 = w21 - lr * dL_dw21
        b1 = b1 - lr * dL_db1

        w12 = w12 - lr * dL_dw12
        w22 = w22 - lr * dL_dw22
        b2 = b2 - lr * dL_db2

        w13 = w13 - lr * dL_dw13
        w23 = w23 - lr * dL_dw23
        b3 = b3 - lr * dL_db3
        total_loss = total_loss + loss
    average_loss = total_loss/inputs

    # print loss for each iteration
    print(f"epoch {epoch} : loss = {average_loss:.4f} \n")

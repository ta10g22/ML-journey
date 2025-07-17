import numpy as np
import matplotlib.pyplot as plt


# Data y = mx + c
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

n = len(x)
losses = []

# Initializing parameters
w = 0  # weights
b = 0  # biases

lr = 0.01  # learning rate
epochs = 2    # number of iterations


# loss function
def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)


# traning loop

for i in range(epochs):
    # predictions
    y_predicted = w * x + b

    # calculate gradients (by using chain rule )
    dw = (-2/n) * np.sum(x * (y - y_predicted))
    db = (-2/n) * np.sum(y-y_predicted)

    # updated weights & biases
    w = w - dw * lr
    b = b - db * lr

    # track loss
    loss = mse(y, y_predicted)
    losses.append(loss)

    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss:.4f}")

# Plot the data and the line
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, w * x + b, color='red', label='Fitted Line')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.show()

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve')
plt.show()

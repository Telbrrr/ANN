import numpy as np

#activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# weights and biases
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30

w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
inputs = np.array([0.05, 0.10])
targets = np.array([0.01, 0.99])
learning_rate = 0.5


def forward_pass(inputs):
    # Hidden layer
    net_h1 = w1 * inputs[0] + w2 * inputs[1] + b1
    out_h1 = sigmoid(net_h1)

    net_h2 = w3 * inputs[0] + w4 * inputs[1] + b1
    out_h2 = sigmoid(net_h2)

    # Output layer 
    net_o1 = w5 * out_h1 + w6 * out_h2 + b2
    out_o1 = sigmoid(net_o1)

    net_o2 = w7 * out_h1 + w8 * out_h2 + b2
    out_o2 = sigmoid(net_o2)

    return out_h1, out_h2, out_o1, out_o2


def backward_pass(inputs, out_h1, out_h2, out_o1, out_o2, targets):
    global w1, w2, w3, w4, w5, w6, w7, w8, b1, b2


    error_o1 = out_o1 - targets[0]
    error_o2 = out_o2 - targets[1]
    d_out_o1 = error_o1 * sigmoid_derivative(out_o1)
    d_out_o2 = error_o2 * sigmoid_derivative(out_o2)

    # Update weights for output layer
    w5 -= learning_rate * d_out_o1 * out_h1
    w6 -= learning_rate * d_out_o1 * out_h2
    w7 -= learning_rate * d_out_o2 * out_h1
    w8 -= learning_rate * d_out_o2 * out_h2


    error_h1 = (d_out_o1 * w5) + (d_out_o2 * w7)
    error_h2 = (d_out_o1 * w6) + (d_out_o2 * w8)
    d_out_h1 = error_h1 * sigmoid_derivative(out_h1)
    d_out_h2 = error_h2 * sigmoid_derivative(out_h2)

    # Update weights for hidden layer
    w1 -= learning_rate * d_out_h1 * inputs[0]
    w2 -= learning_rate * d_out_h1 * inputs[1]
    w3 -= learning_rate * d_out_h2 * inputs[0]
    w4 -= learning_rate * d_out_h2 * inputs[1]

    # Update biases
    b1 -= learning_rate * (d_out_h1 + d_out_h2)
    b2 -= learning_rate * (d_out_o1 + d_out_o2)

# Calculate total error
def calculate_error(out_o1, out_o2, targets):
    error_o1 = 0.5 * (targets[0] - out_o1) ** 2
    error_o2 = 0.5 * (targets[1] - out_o2) ** 2
    return error_o1 + error_o2

epochs = 10000
for epoch in range(epochs):

    out_h1, out_h2, out_o1, out_o2 = forward_pass(inputs)


    backward_pass(inputs, out_h1, out_h2, out_o1, out_o2, targets)


    if epoch % 1000 == 0:
        total_error = calculate_error(out_o1, out_o2, targets)
        print(f"Epoch {epoch}, Error: {total_error}")

# Final outputs and error
out_h1, out_h2, out_o1, out_o2 = forward_pass(inputs)
total_error = calculate_error(out_o1, out_o2, targets)
print(f"\nFinal Outputs: o1 = {out_o1}, o2 = {out_o2}")
print(f"Final Error: {total_error}")
print(f"Final Weights:\nW1={w1},W2={w2},W3={w3},W4={w4}\nW5={w5},W6={w6},W7={w7},W8={w8} ")
# Understanding Backpropagation in Neural Networks

A beginner-friendly explanation of backpropagation, the chain rule, gradient computation, and weight updates in artificial neural networks.

## Overview

Backpropagation is a fundamental algorithm used to train neural networks. It calculates how much each weight and bias contributes to the overall prediction error by propagating gradients backward through the network.

This repository documents the mathematical foundations of backpropagation, breaking down the process into understandable steps with equations and explanations.

## What You'll Learn

- Forward propagation and weighted sums
- Activation functions and their derivatives
- Loss functions and output error
- The chain rule and computational graphs
- Gradient calculation for weights and biases
- Error propagation through hidden layers
- Gradient descent and parameter updates
- Matrix operations for batch training

## Mathematical Foundations

### 1. Forward Propagation

For an input matrix \(X\) and weight matrix \(W\), the pre-activation values are:

\[
Z = XW
\]

Where:

- \(X\) represents the input examples.
- \(W\) represents the learnable weights.
- \(Z\) represents the pre-activation values.

For a batch of examples:

\[
X_{(m \times n)}W_{(n \times h)}
=
Z_{(m \times h)}
\]

Here, \(m\) is the number of examples, \(n\) is the number of input features, and \(h\) is the number of neurons.

### 2. The Chain Rule

Backpropagation relies on the chain rule to compute derivatives through successive operations:

\[
\frac{dL}{dz}
=
\frac{dL}{da}
\cdot
\frac{da}{dz}
\]

Where \(L\) is the loss, \(a\) is the activation, and \(z\) is the pre-activation value.

### 3. Output Layer Delta

For a sigmoid output neuron with an appropriate differentiable loss:

\[
\delta_{\text{output}}
=
\text{output\_error}
\cdot
\sigma'(z_{\text{output}})
\]

For example, with squared-error loss:

\[
\text{output\_error}=a_{\text{output}}-y
\]

The exact delta formula depends on the loss function being used.

### 4. Weight Gradients

For a batch of \(m\) examples, the gradient for the hidden-to-output weights can be written as:

\[
\nabla W_{\text{hidden,output}}
=
\frac{A_{\text{hidden}}^T
\delta_{\text{output}}}{m}
\]

The weights are then updated using gradient descent:

\[
W \leftarrow W-\eta\nabla W
\]

Where \(\eta\) is the learning rate.

## Backpropagation Implementation

The following function implements backpropagation for a neural network with one hidden layer and a sigmoid activation function.

It calculates the output error, propagates gradients backward through the hidden layer, computes weight and bias gradients, and updates the parameters using gradient descent.

### Code

```python
def backward(X, y, z_hidden, a_hidden, z_output, a_output, learning_rate):
    """
    Compute gradients and update the weights and biases using backpropagation.
    """
    global weights_input_hidden, bias_hidden
    global weights_hidden_output, bias_output

    # Number of training examples
    m = X.shape[0]

    # 1. Output layer error
    output_error = a_output - y

    # 2. Output layer delta
    # Combine the output error with the sigmoid derivative
    output_delta = output_error * sigmoid_derivative(z_output)

    # 3. Gradients for hidden-to-output weights and output bias
    grad_weights_hidden_output = np.dot(a_hidden.T, output_delta) / m
    grad_bias_output = np.mean(output_delta, axis=0, keepdims=True)

    # 4. Propagate the error backward to the hidden layer
    hidden_error = np.dot(output_delta, weights_hidden_output.T)

    # 5. Hidden layer delta
    # Scale the propagated error by the hidden neurons' derivatives
    hidden_delta = hidden_error * sigmoid_derivative(z_hidden)

    # 6. Gradients for input-to-hidden weights and hidden bias
    grad_weights_input_hidden = np.dot(X.T, hidden_delta) / m
    grad_bias_hidden = np.mean(hidden_delta, axis=0, keepdims=True)

    # 7. Update weights and biases using gradient descent
    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_output -= learning_rate * grad_bias_output

    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden
```

### How It Works

1. **Calculate the output error:** Compare the predicted output with the target values.
2. **Calculate the output delta:** Multiply the output error by the derivative of the sigmoid activation function.
3. **Compute output-layer gradients:** Calculate the gradients for the hidden-to-output weights and output bias.
4. **Propagate the error:** Use the transpose of the hidden-to-output weight matrix to distribute the error backward.
5. **Calculate the hidden delta:** Multiply the propagated error by the sigmoid derivative of the hidden layer.
6. **Compute hidden-layer gradients:** Calculate gradients for the input-to-hidden weights and hidden bias.
7. **Update parameters:** Adjust all weights and biases in the direction that reduces the loss.

### Key Equations

**Output layer delta**

\[
\delta_{\text{output}}
=
(a_{\text{output}}-y)\odot\sigma'(z_{\text{output}})
\]

**Hidden layer delta**

\[
\delta_{\text{hidden}}
=
(\delta_{\text{output}}W_{\text{hidden,output}}^T)
\odot\sigma'(z_{\text{hidden}})
\]

**Weight gradients**

\[
\nabla W_{\text{hidden,output}}
=
\frac{A_{\text{hidden}}^T\delta_{\text{output}}}{m}
\]

\[
\nabla W_{\text{input,hidden}}
=
\frac{X^T\delta_{\text{hidden}}}{m}
\]

**Gradient descent update**

\[
W\leftarrow W-\eta\nabla W
\]

Here, \(\odot\) denotes element-wise multiplication, \(m\) is the number of training examples, and \(\eta\) is the learning rate.

### Assumptions

- The network contains one hidden layer and one output layer.
- Both layers use sigmoid activation functions.
- The implementation assumes a batch-compatible array layout.
- The output error `a_output - y` is consistent with the gradient of mean squared error up to a constant factor, depending on the loss definition. The code should use a matching loss function and gradient convention.
- The required weights, biases, NumPy import, and `sigmoid_derivative()` function must be defined elsewhere in the project.

## Repository Contents

- **Paper:** A step-by-step explanation of backpropagation.
- **Equations:** Mathematical derivations of gradients and weight updates.
- **Examples:** Illustrations of the calculations involved in neural network training.

The contents may expand as the paper and supporting examples evolve.

## Who Is This For?

This resource is intended for:

- Students learning artificial neural networks.
- Beginners exploring machine learning mathematics.
- Developers implementing neural networks from scratch.
- Anyone seeking an intuitive understanding of gradient-based learning.

Basic familiarity with derivatives, matrices, and neural network terminology is helpful.

## Scope

This is an educational resource explaining established backpropagation concepts. It is not intended to introduce a novel training algorithm or claim new research findings.

## Contributing

Suggestions, corrections, and improvements are welcome. If you find a mathematical error or have an idea that could make an explanation clearer, feel free to open an issue or submit a pull request.

## License

Unless otherwise specified, the repository's content is provided for educational purposes. A formal reuse license can be added when the repository owner selects one.

## Learn Along With the Video Series

New to backpropagation or neural networks? Check out the accompanying YouTube playlist for video explanations that complement this repository.

**[Watch the Backpropagation & Neural Networks Playlist on YouTube](https://www.youtube.com/playlist?list=PLPx5hA5usyrk)**

The playlist is a great starting point if you'd prefer to understand the concepts visually before diving into the mathematical derivations and Python implementation in this repository.

---

**Author:** Tirth S. Pandya

If you find this resource useful, consider starring the repository and sharing it with other learners.
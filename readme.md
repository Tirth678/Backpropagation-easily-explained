# Understanding Backpropagation in Neural Networks

A beginner-friendly guide to understanding backpropagation, the chain
rule, gradient computation, and weight updates in artificial neural
networks.

## Overview

Backpropagation is an algorithm used to train neural networks. It
computes how the loss changes with respect to the network's weights and
biases by propagating gradients backward from the output layer toward
the input layer.

This repository explains the mathematics step by step and includes a
Python implementation of backpropagation for a neural network with one
hidden layer.

## Learn Along With the Video Series

New to backpropagation or neural networks? Watch the accompanying video
playlist for explanations that complement the equations and code in this
repository.

**[Watch the Backpropagation & Neural Networks Playlist on
YouTube](https://www.youtube.com/playlist?list=PLPx5hA5usyrk)**

## What You'll Learn

-   Forward propagation and weighted sums
-   Activation functions and their derivatives
-   The chain rule
-   Output-layer and hidden-layer error signals
-   Weight and bias gradients
-   Matrix operations for batch training
-   Gradient descent and parameter updates

## Mathematical Foundations

### 1. Forward Propagation

For an input matrix (X) and a weight matrix (W), the pre-activation
values are:

$$
Z = XW
$$

Where:

-   \(X\) represents the input examples.
-   \(W\) represents the learnable weights.
-   \(Z\) represents the pre-activation values.

For a batch of examples:

$$
X_{(m \times n)} W_{(n \times h)} = Z_{(m \times h)}
$$

Here, (m) is the number of examples, (n) is the number of input
features, and (h) is the number of neurons. The dimensions align because
the number of columns in (X) equals the number of rows in (W).

### 2. The Chain Rule

Backpropagation uses the chain rule to calculate derivatives through
successive operations:

$$
\frac{dL}{dz} = \frac{dL}{da}\cdot\frac{da}{dz}
$$

Here, (L) is the loss, (a) is an activation, and (z) is the
corresponding pre-activation value.

### 3. Output-Layer Delta

For a sigmoid output neuron using a squared-error loss, the output error is:

$$
\text{output\_error} = a_{\text{output}} - y
$$

The output-layer delta is then:

$$
\delta_{\text{output}} = \text{output\_error}\cdot\sigma'(z_{\text{output}})
$$

The exact delta depends on the loss function. For sigmoid activation
with binary cross-entropy loss, the output delta typically simplifies to
(a\_{`\text{output}`{=tex}} - y), without multiplying by the sigmoid
derivative again.

### 4. Weight Gradients

For a batch containing (m) examples, the gradient for the
hidden-to-output weights is:

$$
\nabla W_{\text{hidden,output}} =
\frac{A_{\text{hidden}}^T\delta_{\text{output}}}{m}
$$

The gradient for the input-to-hidden weights is:

$$
\nabla W_{\text{input,hidden}} =
\frac{X^T\delta_{\text{hidden}}}{m}
$$

The weights are updated using gradient descent:

$$
W \leftarrow W - \eta\nabla W
$$

Here, (`\eta`{=tex}) is the learning rate, (T) denotes matrix transpose,
and the delta terms contain the layer-wise error signals.

## Backpropagation Implementation

The function below computes gradients and updates the weights and biases
for a neural network with one hidden layer. It assumes that the weights,
biases, NumPy import, and `sigmoid_derivative()` function have already
been defined.

``` python
def backward(X, y, z_hidden, a_hidden, z_output, a_output, learning_rate):
    """
    Compute gradients and update weights and biases using backpropagation.
    """
    global weights_input_hidden, bias_hidden
    global weights_hidden_output, bias_output

    # Number of training examples
    m = X.shape[0]

    # 1. Calculate the output error
    output_error = a_output - y

    # 2. Calculate the output-layer delta
    output_delta = output_error * sigmoid_derivative(z_output)

    # 3. Calculate gradients for hidden-to-output weights and output bias
    grad_weights_hidden_output = np.dot(a_hidden.T, output_delta) / m
    grad_bias_output = np.mean(output_delta, axis=0, keepdims=True)

    # 4. Propagate the error backward to the hidden layer
    hidden_error = np.dot(output_delta, weights_hidden_output.T)

    # 5. Calculate the hidden-layer delta
    hidden_delta = hidden_error * sigmoid_derivative(z_hidden)

    # 6. Calculate gradients for input-to-hidden weights and hidden bias
    grad_weights_input_hidden = np.dot(X.T, hidden_delta) / m
    grad_bias_hidden = np.mean(hidden_delta, axis=0, keepdims=True)

    # 7. Update weights and biases using gradient descent
    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_output -= learning_rate * grad_bias_output

    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden
```

### How the Code Works

1.  **Calculate the output error:** Compare the predicted output with
    the target value.
2.  **Calculate the output delta:** Scale the output error by the
    derivative of the sigmoid activation.
3.  **Compute output-layer gradients:** Use the hidden-layer activations
    and output delta to calculate weight and bias gradients.
4.  **Propagate the error backward:** Multiply the output delta by the
    transpose of the hidden-to-output weight matrix.
5.  **Calculate the hidden delta:** Scale the propagated error by the
    derivative of the hidden-layer sigmoid activation.
6.  **Compute hidden-layer gradients:** Use the input matrix and hidden
    delta to calculate weight and bias gradients.
7.  **Update parameters:** Apply gradient descent to the weights and
    biases.

### Notes About the Implementation

-   The code uses NumPy-style matrix operations and expects
    batch-compatible array shapes.
-   The division by (m) averages the weight gradients across the
    training batch.
-   The bias gradients are averaged across examples using `np.mean`.
-   The output delta shown in the code is appropriate for a sigmoid
    output with a squared-error-style loss convention. Ensure your
    training loss and derivative conventions match.
-   The weights and biases are updated in place, so they must be
    initialized before calling `backward()`.

## Who Is This For?

This repository is intended for:

-   Students learning artificial neural networks.
-   Beginners exploring the mathematics behind machine learning.
-   Developers implementing neural networks from scratch.
-   Learners who want to connect mathematical equations to Python code.

Basic familiarity with derivatives, matrices, and neural network
terminology is helpful.

## Scope

This is an educational explanation of established backpropagation
concepts. It does not claim to introduce a new algorithm or novel
research finding.

## Contributing

Suggestions, corrections, and improvements are welcome. If you find a
mathematical error or have an idea that could make an explanation
clearer, feel free to open an issue or submit a pull request.

## License

A license has not yet been specified. Add a license file if you want to
clearly define how others may reuse or distribute this repository's
content.

------------------------------------------------------------------------

**Author:** Tirth S. Pandya

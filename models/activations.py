```python
import numpy as np

class Activation:
    """
    Base class for activation functions.
    All activation functions should inherit from this class and implement
    the forward and backward methods.
    """
    def forward(self, x):
        """
        Performs the forward pass of the activation function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array after applying the activation function.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Performs the backward pass (computes the gradient of the loss with
        respect to the input of the activation function).

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect
                                      to the output of this activation function.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input
                        of this activation function.
        """
        raise NotImplementedError

class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    f(x) = max(0, x)
    """
    def __init__(self):
        self.input = None # Stores the input for use in the backward pass

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        # Derivative is 1 for x > 0, 0 for x <= 0.
        # We handle x = 0 by commonly assigning 0 gradient.
        return grad_output * (self.input > 0)

class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    f(x) = x for x >= 0
    f(x) = alpha * x for x < 0
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(self.alpha * x, x)

    def backward(self, grad_output):
        # Derivative is 1 for x >= 0, alpha for x < 0.
        dx = np.ones_like(self.input)
        dx[self.input < 0] = self.alpha
        return grad_output * dx

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    """
    def __init__(self):
        self.output = None # Stores the output for use in the backward pass

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.output = y
        return y

    def backward(self, grad_output):
        # Derivative of sigmoid: y * (1 - y)
        return self.output * (1 - self.output) * grad_output

class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) activation function.
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    def __init__(self):
        self.output = None # Stores the output for use in the backward pass

    def forward(self, x):
        y = np.tanh(x)
        self.output = y
        return y

    def backward(self, grad_output):
        # Derivative of tanh: 1 - y^2
        return (1 - self.output**2) * grad_output

class Softmax(Activation):
    """
    Softmax activation function.
    Applies the exponential function to each element and then normalizes
    by the sum of all exponentials. Used primarily in the output layer
    of a classifier to produce probability distributions.
    """
    def __init__(self, axis=-1):
        self.output = None
        self.axis = axis # The axis along which softmax is computed (e.g., -1 for last dimension)

    def forward(self, x):
        # Subtract max for numerical stability to prevent overflow
        exp_x = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        y = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        self.output = y
        return y

    def backward(self, grad_output):
        # Softmax backward pass for a batch of inputs (N, C) where N is batch size, C is number of classes.
        # The gradient dL/dx_j for a single sample is y_j * (dL/dy_j - sum_k(dL/dy_k * y_k))
        y = self.output

        # Calculate sum_k(dL/dy_k * y_k) for each sample
        sum_grad_output_y = np.sum(grad_output * y, axis=self.axis, keepdims=True)

        # Compute dL/dx
        grad_input = y * (grad_output - sum_grad_output_y)

        return grad_input

class ELU(Activation):
    """
    Exponential Linear Unit (ELU) activation function.
    f(x) = x for x >= 0
    f(x) = alpha * (exp(x) - 1) for x < 0
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        self.input = x
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, grad_output):
        # Derivative is 1 for x >= 0
        # Derivative is alpha * exp(x) for x < 0
        dx = np.ones_like(self.input)
        neg_indices = self.input < 0
        dx[neg_indices] = self.alpha * np.exp(self.input[neg_indices])
        return grad_output * dx

class SELU(Activation):
    """
    Scaled Exponential Linear Unit (SELU) activation function.
    f(x) = scale * (x for x >= 0)
    f(x) = scale * (alpha * (exp(x) - 1) for x < 0)
    where alpha and scale are fixed constants designed for self-normalizing properties.
    Commonly: alpha = 1.6732632423543772, scale = 1.0507009873554804
    """
    def __init__(self, alpha=1.6732632423543772, scale=1.0507009873554804):
        self.alpha = alpha
        self.scale = scale
        self.input = None

    def forward(self, x):
        self.input = x
        return self.scale * np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, grad_output):
        # Derivative: scale * 1 for x >= 0
        # Derivative: scale * alpha * exp(x) for x < 0
        dx = np.ones_like(self.input)
        neg_indices = self.input < 0
        dx[neg_indices] = self.alpha * np.exp(self.input[neg_indices])
        return self.scale * grad_output * dx
```
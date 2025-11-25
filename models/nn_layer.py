import numpy as np

class NNLayer:
    """
    Implements a basic fully connected neural network layer from scratch.
    This layer performs a linear transformation: output = input @ weights + bias.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initializes the neural network layer with random weights and zero biases.

        Args:
            input_dim (int): The number of input features (neurons in the previous layer).
            output_dim (int): The number of output features (neurons in this layer).
        """
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize weights with a small random distribution (e.g., He or Glorot/Xavier initialization)
        # Using a simple normal distribution scaled by sqrt(2 / input_dim)
        # for a rough He-like initialization often works well with ReLU activations.
        # For a pure linear layer, std dev of 1/sqrt(input_dim) (Xavier) might be more appropriate.
        std_dev = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim) * std_dev

        # Initialize biases to zeros
        self.bias = np.zeros((1, output_dim))

        # Variables to store input and gradients for backward pass
        self.input_tensor = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass calculation for the layer.

        Args:
            input_tensor (np.ndarray): The input data for this layer,
                                      expected shape (batch_size, input_dim).

        Returns:
            np.ndarray: The output of the layer, shape (batch_size, output_dim).
        """
        if not isinstance(input_tensor, np.ndarray):
            raise TypeError("Input tensor must be a numpy array.")
        if input_tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input tensor last dimension ({input_tensor.shape[-1]}) "
                f"must match layer input_dim ({self.input_dim})."
            )

        self.input_tensor = input_tensor
        output = np.dot(input_tensor, self.weights) + self.bias
        return output

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs the backward pass calculation for the layer,
        computes gradients, and updates weights and biases.

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to the output
                                      of this layer, shape (batch_size, output_dim).
            learning_rate (float): The learning rate to use for updating weights and biases.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input of this layer,
                        shape (batch_size, input_dim).
        """
        if self.input_tensor is None:
            raise RuntimeError("Forward pass must be called before backward pass.")
        if not isinstance(grad_output, np.ndarray):
            raise TypeError("grad_output must be a numpy array.")
        if grad_output.shape[-1] != self.output_dim:
            raise ValueError(
                f"grad_output last dimension ({grad_output.shape[-1]}) "
                f"must match layer output_dim ({self.output_dim})."
            )
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")

        # Gradient with respect to weights (d_L/d_W = X.T @ d_L/d_Y)
        # self.input_tensor shape: (batch_size, input_dim)
        # grad_output shape: (batch_size, output_dim)
        # d_weights shape: (input_dim, output_dim)
        self.d_weights = np.dot(self.input_tensor.T, grad_output)

        # Gradient with respect to bias (d_L/d_B = sum(d_L/d_Y, axis=0))
        # Summing along the batch dimension to get gradients for each bias term
        # d_bias shape: (1, output_dim)
        self.d_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Gradient with respect to input (d_L/d_X = d_L/d_Y @ W.T)
        # grad_input shape: (batch_size, input_dim)
        grad_input = np.dot(grad_output, self.weights.T)

        # Update weights and biases using gradient descent
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

        return grad_input

    def get_weights(self) -> np.ndarray:
        """Returns the current weights of the layer."""
        return self.weights

    def get_bias(self) -> np.ndarray:
        """Returns the current biases of the layer."""
        return self.bias

    def get_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the last computed gradients for weights and biases."""
        if self.d_weights is None or self.d_bias is None:
            raise RuntimeError("Backward pass has not been called yet.")
        return self.d_weights, self.d_bias
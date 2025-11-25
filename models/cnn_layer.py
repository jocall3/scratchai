```python
import numpy as np

class CNNLayer:
    """
    Implements a basic convolutional neural network layer from scratch.
    This implementation uses 'valid' padding, meaning no padding is applied,
    and the output feature map dimensions will be smaller than the input.
    """
    def __init__(self, input_shape, num_filters, kernel_size, stride=1):
        """
        Initializes the convolutional layer's parameters (weights and biases).

        Args:
            input_shape (tuple): A tuple (H_in, W_in, C_in) representing the
                                 height, width, and number of channels of the input feature map.
            num_filters (int): The number of output filters (feature maps)
                                this layer will produce.
            kernel_size (tuple): A tuple (K_H, K_W) representing the height and
                                 width of the convolutional kernel (filter).
            stride (int): The step size for sliding the kernel across the input.
                          Defaults to 1.
        
        Raises:
            ValueError: If input parameters are invalid or lead to an invalid output shape.
        """
        if not (isinstance(input_shape, tuple) and len(input_shape) == 3 and 
                all(isinstance(i, int) and i > 0 for i in input_shape)):
            raise ValueError("input_shape must be a tuple of 3 positive integers (H_in, W_in, C_in).")
        if not (isinstance(num_filters, int) and num_filters > 0):
            raise ValueError("num_filters must be a positive integer.")
        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2 and 
                all(isinstance(k, int) and k > 0 for k in kernel_size)):
            raise ValueError("kernel_size must be a tuple of 2 positive integers (K_H, K_W).")
        if not (isinstance(stride, int) and stride > 0):
            raise ValueError("stride must be a positive integer.")

        self.H_in, self.W_in, self.C_in = input_shape
        self.num_filters = num_filters
        self.K_H, self.K_W = kernel_size
        self.stride = stride

        # He initialization for weights, suitable for ReLU activations
        # fan_in is the number of input connections for each neuron (filter element)
        fan_in = self.K_H * self.K_W * self.C_in
        std_dev = np.sqrt(2.0 / fan_in)
        self.weights = np.random.randn(self.K_H, self.K_W, self.C_in, self.num_filters) * std_dev

        # Biases initialized to zeros
        self.biases = np.zeros((self.num_filters,))

        # Placeholder for storing input during forward pass, used in backpropagation
        self.input = None

        # Calculate output dimensions for 'valid' padding
        self.H_out = (self.H_in - self.K_H) // self.stride + 1
        self.W_out = (self.W_in - self.K_W) // self.stride + 1

        if self.H_out <= 0 or self.W_out <= 0:
            raise ValueError(
                f"Invalid kernel_size ({self.K_H}, {self.K_W}) or stride ({self.stride}) "
                f"for input_shape ({self.H_in}, {self.W_in}, {self.C_in}). "
                f"Output dimensions ({self.H_out}, {self.W_out}) must be positive. "
                f"Ensure (H_in - K_H) >= 0 and (W_in - K_W) >= 0 for 'valid' padding."
            )

    def forward(self, X):
        """
        Performs the forward pass (convolution operation) of the layer.

        Args:
            X (np.ndarray): Input data of shape (batch_size, H_in, W_in, C_in).

        Returns:
            np.ndarray: Output feature map of shape (batch_size, H_out, W_out, num_filters).
        
        Raises:
            ValueError: If the input data's spatial or channel dimensions do not match
                        the layer's expected input_shape.
        """
        if X.shape[1:] != (self.H_in, self.W_in, self.C_in):
            raise ValueError(
                f"Input data shape {X.shape[1:]} does not match expected input_shape "
                f"({self.H_in}, {self.W_in}, {self.C_in})."
            )

        self.input = X  # Store input for use in the backward pass
        batch_size = X.shape[0]

        # Initialize the output feature map
        output = np.zeros((batch_size, self.H_out, self.W_out, self.num_filters))

        # Iterate over each image in the batch
        for b in range(batch_size):
            # Iterate over each output spatial position
            for h in range(self.H_out):
                for w in range(self.W_out):
                    # Calculate the starting and ending indices for the current input patch
                    h_start = h * self.stride
                    h_end = h_start + self.K_H
                    w_start = w * self.stride
                    w_end = w_start + self.K_W

                    # Extract the input patch
                    # Shape: (K_H, K_W, C_in)
                    patch = X[b, h_start:h_end, w_start:w_end, :]

                    # Perform convolution for each filter
                    for f in range(self.num_filters):
                        # Element-wise multiplication of the patch with the current filter's weights
                        # followed by a sum over all spatial and channel dimensions.
                        # Then add the corresponding bias.
                        output[b, h, w, f] = np.sum(patch * self.weights[:, :, :, f]) + self.biases[f]
        return output

    def backward(self, dZ):
        """
        Performs the backward pass of the convolutional layer to compute gradients
        with respect to weights, biases, and input.

        Args:
            dZ (np.ndarray): Gradient of the loss with respect to the output of this layer.
                             Shape: (batch_size, H_out, W_out, num_filters).

        Returns:
            tuple: (dW, db, dX)
                dW (np.ndarray): Gradient of the loss with respect to the weights.
                                 Shape: (K_H, K_W, C_in, num_filters).
                db (np.ndarray): Gradient of the loss with respect to the biases.
                                 Shape: (num_filters,).
                dX (np.ndarray): Gradient of the loss with respect to the input X.
                                 Shape: (batch_size, H_in, W_in, C_in).
        """
        batch_size = dZ.shape[0]

        # Initialize gradients for weights and input
        dW = np.zeros_like(self.weights)
        dX = np.zeros_like(self.input)

        # Gradient for biases can be computed by summing dZ across batch, height, and width
        db = np.sum(dZ, axis=(0, 1, 2))

        # Iterate over each image in the batch
        for b in range(batch_size):
            # Iterate over each output spatial position
            for h in range(self.H_out):
                for w in range(self.W_out):
                    # Calculate the starting and ending indices for the current input patch
                    h_start = h * self.stride
                    h_end = h_start + self.K_H
                    w_start = w * self.stride
                    w_end = w_start + self.K_W

                    # Extract the input patch used during the forward pass
                    # Shape: (K_H, K_W, C_in)
                    patch = self.input[b, h_start:h_end, w_start:w_end, :]

                    # Iterate over each filter
                    for f in range(self.num_filters):
                        # The gradient flowing into this specific output cell (b,h,w,f)
                        grad_output = dZ[b, h, w, f]

                        # Gradient for weights:
                        # Each weight contributing to this output cell is scaled by the corresponding input value
                        # and the gradient from the output. Summing these over all output cells gives dW.
                        dW[:, :, :, f] += patch * grad_output

                        # Gradient for input (dX):
                        # Each input pixel contributes to multiple output cells.
                        # Here, we add the gradient propagated back from this output cell (b,h,w,f)
                        # to the corresponding input patch, weighted by the filter's weights.
                        dX[b, h_start:h_end, w_start:w_end, :] += self.weights[:, :, :, f] * grad_output
        return dW, db, dX
```
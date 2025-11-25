import numpy as np
from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """
    Abstract base class for all neural network layers.

    All concrete layer implementations must inherit from this class and
    implement its abstract methods.
    """

    def __init__(self):
        """
        Initializes the base layer.
        Sets default mode to training.
        """
        self._mode = 'train'

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass computation of the layer.

        Args:
            input_data (np.ndarray): The input tensor to the layer.

        Returns:
            np.ndarray: The output tensor of the layer.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass computation of the layer.
        Computes gradients with respect to inputs and updates internal parameters.

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to
                                      the output of this layer.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input
                        of this layer, passed to the previous layer.
        """
        pass

    @abstractmethod
    def parameters(self) -> list:
        """
        Returns a list of trainable parameters (e.g., weights, biases) for this layer.
        Each parameter should typically be an object or a tuple containing
        the parameter value and its corresponding gradient.

        Returns:
            list: A list of parameter objects (or tuples like (value, gradient)).
        """
        pass

    @abstractmethod
    def zero_grad(self):
        """
        Resets the gradients of all trainable parameters in the layer to zero.
        """
        pass

    def train(self):
        """
        Sets the layer to training mode.
        Some layers (e.g., BatchNorm, Dropout) behave differently in train vs. eval mode.
        """
        self._mode = 'train'

    def eval(self):
        """
        Sets the layer to evaluation (inference) mode.
        Some layers (e.g., BatchNorm, Dropout) behave differently in train vs. eval mode.
        """
        self._mode = 'eval'

    @property
    def training(self) -> bool:
        """
        Returns True if the layer is in training mode, False otherwise.
        """
        return self._mode == 'train'

    @property
    def evaluation(self) -> bool:
        """
        Returns True if the layer is in evaluation mode, False otherwise.
        """
        return self._mode == 'eval'
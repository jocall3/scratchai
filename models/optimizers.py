```python
import math

class Optimizer:
    """
    Base class for optimization algorithms.

    Provides common utility methods for element-wise operations on lists of numbers,
    emulating basic vector operations without external libraries.
    It is assumed that parameters and gradients are provided as flattened lists
    of numerical values.
    """
    def __init__(self, learning_rate):
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """
        Updates the model parameters in-place based on computed gradients.

        Subclasses must implement this method.

        Args:
            params (list): A mutable list of numerical values representing model parameters.
                           These values are updated in-place by the optimizer.
            grads (list): A list of numerical values representing gradients corresponding to `params`.
                          Must have the same length as `params`.
        """
        raise NotImplementedError("Subclasses must implement the 'update' method.")

    def _zero_like(self, target_list):
        """
        Creates a list of zeros with the same length as the target list.
        """
        return [0.0] * len(target_list)

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum.

    Parameters are updated based on their gradients, potentially incorporating
    a fraction of the previous update vector to accelerate convergence.
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between 0.0 and 1.0 (inclusive).")
        self.momentum = momentum
        self.velocity = None  # Stores momentum buffers, initialized on first update

    def update(self, params, grads):
        if len(params) != len(grads):
            raise ValueError(f"Parameters and gradients must have the same length. Got {len(params)} and {len(grads)}.")

        if self.velocity is None:
            self.velocity = self._zero_like(params)

        for i in range(len(params)):
            # Calculate new velocity: v_new = momentum * v_old + learning_rate * grad
            self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * grads[i]
            
            # Update parameter in-place: param_new = param_old - v_new
            params[i] -= self.velocity[i]
        
        return params

class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Adam uses adaptive learning rates for each parameter, computed from
    estimates of first and second moments of the gradients.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("Beta1 must be between 0.0 and 1.0 (exclusive of 1.0).")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("Beta2 must be between 0.0 and 1.0 (exclusive of 1.0).")
        if not (epsilon > 0.0):
            raise ValueError("Epsilon must be greater than 0.")

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimates (mean)
        self.v = None  # Second moment estimates (uncentered variance)
        self.t = 0     # Time step / iteration counter

    def update(self, params, grads):
        if len(params) != len(grads):
            raise ValueError(f"Parameters and gradients must have the same length. Got {len(params)} and {len(grads)}.")

        self.t += 1  # Increment time step on each update call
        
        if self.m is None:
            self.m = self._zero_like(params)
            self.v = self._zero_like(params)

        # Pre-calculate bias correction factors for efficiency
        beta1_power_t = self.beta1 ** self.t
        beta2_power_t = self.beta2 ** self.t
        
        for i in range(len(params)):
            # Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            # Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            # Compute bias-corrected first moment estimate: m_hat = m_t / (1 - beta1^t)
            m_hat = self.m[i] / (1 - beta1_power_t)
            # Compute bias-corrected second moment estimate: v_hat = v_t / (1 - beta2^t)
            v_hat = self.v[i] / (1 - beta2_power_t)

            # Update parameters: param_new = param_old - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            params[i] -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
        
        return params

class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    RMSprop maintains a moving average of the square of gradients,
    and divides the learning rate by the root of this moving average.
    """
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-6):
        super().__init__(learning_rate)
        if not (0.0 <= rho < 1.0):
            raise ValueError("Rho (decay rate) must be between 0.0 and 1.0 (exclusive of 1.0).")
        if not (epsilon > 0.0):
            raise ValueError("Epsilon must be greater than 0.")

        self.rho = rho
        self.epsilon = epsilon
        self.s = None  # Stores moving average of squared gradients

    def update(self, params, grads):
        if len(params) != len(grads):
            raise ValueError(f"Parameters and gradients must have the same length. Got {len(params)} and {len(grads)}.")

        if self.s is None:
            self.s = self._zero_like(params)

        for i in range(len(params)):
            # Update moving average of squared gradients: s_t = rho * s_{t-1} + (1 - rho) * g_t^2
            self.s[i] = self.rho * self.s[i] + (1 - self.rho) * (grads[i] ** 2)
            
            # Update parameters: param_new = param_old - learning_rate * g_t / (sqrt(s_t) + epsilon)
            params[i] -= self.learning_rate * grads[i] / (math.sqrt(self.s[i]) + self.epsilon)
            
        return params
```
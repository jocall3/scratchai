# ai_logic/trainer.py

# --- Basic Math Operations (from scratch) ---
# These functions simulate basic array/matrix operations using Python lists of lists.
# They handle 1D vectors (lists) and 2D matrices (lists of lists).

def _transpose(matrix):
    """Transposes a matrix (list of lists) or converts a 1D list to a column vector."""
    if not matrix:
        return []
    if not isinstance(matrix[0], list): # If it's a 1D list (row vector)
        return [[x] for x in matrix] # Convert to column vector
    
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed

def _matrix_multiply(matrix1, matrix2):
    """Multiplies two matrices or a scalar by a matrix."""
    if not matrix1 or not matrix2:
        return []

    # Handle scalar multiplication
    if isinstance(matrix1, (int, float)):
        return [[matrix1 * x for x in row] for row in matrix2] if isinstance(matrix2[0], list) else [matrix1 * x for x in matrix2]
    if isinstance(matrix2, (int, float)):
        return [[matrix2 * x for x in row] for row in matrix1] if isinstance(matrix1[0], list) else [matrix2 * x for x in matrix1]

    # Convert 1D vectors to 2D matrices for consistent multiplication
    m1_is_1d = not isinstance(matrix1[0], list)
    m2_is_1d = not isinstance(matrix2[0], list)
    
    m1_rows = len(matrix1)
    m1_cols = len(matrix1[0]) if not m1_is_1d else 1
    m2_rows = len(matrix2) if not m2_is_1d else 1
    m2_cols = len(matrix2[0]) if not m2_is_1d else len(matrix2)

    # Reshape 1D vectors for matrix multiplication if necessary (e.g., dot product)
    # This assumes matrix1 is (R1, C1) and matrix2 is (R2, C2)
    # If matrix1 is (N,) and matrix2 is (N, M), it becomes (1, N) @ (N, M)
    # If matrix1 is (M, N) and matrix2 is (N,), it becomes (M, N) @ (N, 1)
    
    if m1_is_1d: # treat as a row vector (1, N)
        matrix1_reshaped = [matrix1]
        m1_rows = 1
        m1_cols = len(matrix1)
    else:
        matrix1_reshaped = matrix1

    if m2_is_1d: # treat as a column vector (N, 1)
        matrix2_reshaped = [[x] for x in matrix2]
        m2_rows = len(matrix2)
        m2_cols = 1
    else:
        matrix2_reshaped = matrix2

    if m1_cols != m2_rows:
        raise ValueError(f"Matrix dimensions mismatch for multiplication: {m1_rows}x{m1_cols} vs {m2_rows}x{m2_cols}")

    result = [[0 for _ in range(m2_cols)] for _ in range(m1_rows)]
    for i in range(m1_rows):
        for j in range(m2_cols):
            for k in range(m1_cols):
                result[i][j] += matrix1_reshaped[i][k] * matrix2_reshaped[k][j]
    return result

def _matrix_add(matrix1, matrix2):
    """Adds two matrices element-wise or a scalar to a matrix."""
    if not matrix1: return matrix2
    if not matrix2: return matrix1

    # Handle scalar addition
    if isinstance(matrix1, (int, float)):
        if isinstance(matrix2[0], list): return [[matrix1 + x for x in row] for row in matrix2]
        return [matrix1 + x for x in matrix2]
    if isinstance(matrix2, (int, float)):
        if isinstance(matrix1[0], list): return [[matrix2 + x for x in row] for row in matrix1]
        return [matrix2 + x for x in matrix1]

    # Standard matrix addition
    m1_is_1d = not isinstance(matrix1[0], list)
    m2_is_1d = not isinstance(matrix2[0], list)

    if m1_is_1d != m2_is_1d or len(matrix1) != len(matrix2) or (not m1_is_1d and len(matrix1[0]) != len(matrix2[0])):
        raise ValueError("Matrix dimensions must match for addition.")
    
    if m1_is_1d: # 1D vectors
        return [matrix1[i] + matrix2[i] for i in range(len(matrix1))]
    else: # 2D matrices
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = matrix1[i][j] + matrix2[i][j]
        return result

def _matrix_subtract(matrix1, matrix2):
    """Subtracts two matrices element-wise or a scalar from/to a matrix."""
    if not matrix1: return _scalar_multiply(matrix2, -1) # Assuming matrix2 exists
    if not matrix2: return matrix1

    # Handle scalar subtraction
    if isinstance(matrix1, (int, float)):
        if isinstance(matrix2[0], list): return [[matrix1 - x for x in row] for row in matrix2]
        return [matrix1 - x for x in matrix2]
    if isinstance(matrix2, (int, float)):
        if isinstance(matrix1[0], list): return [[x - matrix2 for x in row] for row in matrix1]
        return [x - matrix2 for x in matrix1]

    # Standard matrix subtraction
    m1_is_1d = not isinstance(matrix1[0], list)
    m2_is_1d = not isinstance(matrix2[0], list)

    if m1_is_1d != m2_is_1d or len(matrix1) != len(matrix2) or (not m1_is_1d and len(matrix1[0]) != len(matrix2[0])):
        raise ValueError("Matrix dimensions must match for subtraction.")
    
    if m1_is_1d: # 1D vectors
        return [matrix1[i] - matrix2[i] for i in range(len(matrix1))]
    else: # 2D matrices
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = matrix1[i][j] - matrix2[i][j]
        return result

def _scalar_multiply(matrix, scalar):
    """Multiplies a matrix (or 1D list) by a scalar."""
    if not matrix: return []
    if isinstance(matrix[0], list):
        return [[val * scalar for val in row] for row in matrix]
    else: # 1D vector
        return [val * scalar for val in matrix]

def _elementwise_multiply(matrix1, matrix2):
    """Performs element-wise multiplication (Hadamard product) of two matrices."""
    if not matrix1 or not matrix2:
        return []

    # Handle scalar cases for element-wise (Hadamard product)
    if isinstance(matrix1, (int, float)):
        if isinstance(matrix2[0], list): return [[matrix1 * x for x in row] for row in matrix2]
        return [matrix1 * x for x in matrix2]
    if isinstance(matrix2, (int, float)):
        if isinstance(matrix1[0], list): return [[matrix2 * x for x in row] for row in matrix1]
        return [matrix2 * x for x in matrix1]
        
    m1_is_1d = not isinstance(matrix1[0], list)
    m2_is_1d = not isinstance(matrix2[0], list)

    if m1_is_1d != m2_is_1d or len(matrix1) != len(matrix2) or (not m1_is_1d and len(matrix1[0]) != len(matrix2[0])):
        raise ValueError("Matrix dimensions must match for element-wise multiplication.")
    
    if m1_is_1d: # 1D vectors
        return [matrix1[i] * matrix2[i] for i in range(len(matrix1))]
    else: # 2D matrices
        rows = len(matrix1)
        cols = len(matrix1[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = matrix1[i][j] * matrix2[i][j]
        return result

# --- Activation Functions and their Derivatives ---

import math

def _relu(matrix):
    """Applies the ReLU activation function element-wise."""
    if not matrix: return []
    if isinstance(matrix[0], list):
        return [[max(0, x) for x in row] for row in matrix]
    else: # 1D vector
        return [max(0, x) for x in matrix]

def _relu_derivative(matrix):
    """Calculates the derivative of the ReLU function element-wise."""
    if not matrix: return []
    if isinstance(matrix[0], list):
        return [[1 if x > 0 else 0 for x in row] for row in matrix]
    else: # 1D vector
        return [1 if x > 0 else 0 for x in matrix]

def _softmax(matrix):
    """Applies the Softmax activation function to each row (representing a sample)."""
    if not matrix: return []
    if not isinstance(matrix[0], list): # Single vector
        exp_values = [math.exp(x) for x in matrix]
        sum_exp = sum(exp_values)
        return [x / sum_exp for x in exp_values] if sum_exp != 0 else [0.0] * len(matrix)
    else: # Batch of vectors
        result = []
        for row in matrix:
            exp_values = [math.exp(x) for x in row]
            sum_exp = sum(exp_values)
            result.append([x / sum_exp for x in exp_values] if sum_exp != 0 else [0.0] * len(row))
        return result

# --- Placeholder Components for a self-contained Trainer ---
# These classes simulate the interfaces expected by the Trainer,
# allowing `trainer.py` to be run independently.
# In a full project, these would be imported from other modules (e.g., nn.py, optimizers.py).

class Parameter:
    """A simple wrapper for a learnable parameter (weight or bias) and its gradient."""
    def __init__(self, data):
        self.data = data # This is a list or list of lists
        if isinstance(data[0], list):
            self.grad = [[0.0 for _ in row] for row in data]
        else:
            self.grad = [0.0 for _ in data]

    def zero_grad(self):
        if isinstance(self.data[0], list):
            self.grad = [[0.0 for _ in row] for row in self.data]
        else:
            self.grad = [0.0 for _ in self.data]

    def __repr__(self):
        return f"Parameter(data_shape={len(self.data)}x{len(self.data[0]) if isinstance(self.data[0], list) else 1})"

class Layer:
    """Base class for all neural network layers."""
    def __init__(self):
        self.output = None
        self.input = None
        self.parameters = [] # List of Parameter objects (weights, biases)

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_parameters(self):
        return self.parameters

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

class Linear(Layer):
    """A simple fully connected (dense) layer."""
    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialize weights with small random values
        # Weights: (input_size, output_size)
        # Biases: (output_size,)
        self.weights = Parameter([[((x % 100) / 500.0 - 0.1) for x in range(output_size)] for _ in range(input_size)]) # Small random init
        self.biases = Parameter([((x % 100) / 500.0 - 0.1) for x in range(output_size)]) # Small random init
        self.parameters = [self.weights, self.biases]

    def forward(self, input_data):
        self.input = input_data # Store input for backward pass
        
        # Ensure input_data is treated as matrix for multiplication (batch_size, input_size)
        is_single_sample = not isinstance(input_data[0], list)
        if is_single_sample:
            input_data_matrix = [input_data]
        else:
            input_data_matrix = input_data

        # Perform Wx + b
        # Wx results in (batch_size, output_size)
        wx = _matrix_multiply(input_data_matrix, self.weights.data)
        
        # Add biases: biases are (output_size,) and need to be broadcasted to all rows of wx
        self.output = [_matrix_add(row, self.biases.data) for row in wx]
        
        # If input was a single sample, return single sample output
        if is_single_sample:
            return self.output[0]
        return self.output

    def backward(self, grad_output):
        # grad_output is dL/dY (gradient from the next layer/loss)
        # dL/dW = X_T @ dL/dY
        # dL/db = sum(dL/dY, axis=0) (sum over batch dimension)
        # dL/dX = dL/dY @ W_T

        is_single_sample = not isinstance(self.input[0], list)
        if is_single_sample:
            input_matrix = [self.input]
            grad_output_matrix = [grad_output]
        else:
            input_matrix = self.input
            grad_output_matrix = grad_output

        input_T = _transpose(input_matrix)
        
        # dL/dW accumulation
        # self.weights.grad and the new grad need to be added
        grad_w = _matrix_multiply(input_T, grad_output_matrix)
        self.weights.grad = _matrix_add(self.weights.grad, grad_w)

        # dL/db accumulation (sum over batch for biases)
        # grad_output_matrix is (batch_size, output_size)
        bias_grad_sum = [0.0 for _ in range(len(grad_output_matrix[0]))]
        for row in grad_output_matrix:
            for i, val in enumerate(row):
                bias_grad_sum[i] += val
        
        self.biases.grad = _matrix_add(self.biases.grad, bias_grad_sum)

        # dL/dX to pass to the previous layer
        weights_T = _transpose(self.weights.data)
        grad_input = _matrix_multiply(grad_output_matrix, weights_T)

        if is_single_sample:
            return grad_input[0]
        return grad_input

class ReLU(Layer):
    """ReLU activation layer."""
    def forward(self, input_data):
        self.input = input_data
        self.output = _relu(input_data)
        return self.output

    def backward(self, grad_output):
        relu_derivative_matrix = _relu_derivative(self.input)
        grad_input = _elementwise_multiply(grad_output, relu_derivative_matrix)
        return grad_input

class Softmax(Layer):
    """Softmax activation layer (typically for output layer in classification)."""
    def forward(self, input_data):
        self.input = input_data
        self.output = _softmax(input_data)
        return self.output

    def backward(self, grad_output):
        # When Softmax is combined with CrossEntropyLoss, the gradient w.r.t.
        # the pre-softmax logits (dL/dZ) is often directly computed by the loss function.
        # Thus, this Softmax layer acts as a passthrough for the gradient
        # if used with CategoricalCrossEntropyLoss as implemented here.
        return grad_output


class NeuralNetwork: # This would be `nn.py`'s Model class
    """A simple feedforward neural network."""
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_parameters(self):
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.get_parameters())
        return all_params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

class MSELoss: # This would be `loss_functions.py`'s MSELoss
    """Mean Squared Error Loss."""
    def forward(self, predictions, targets):
        if not predictions or not targets: return 0.0
        
        # Ensure predictions and targets are lists of lists for consistent iteration
        is_single_sample = not isinstance(predictions[0], list)
        if is_single_sample:
            predictions = [predictions]
            targets = [targets]
            
        diff_sq_sum = 0.0
        num_elements = 0
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                diff = predictions[i][j] - targets[i][j]
                diff_sq_sum += diff * diff
                num_elements += 1
        return diff_sq_sum / num_elements if num_elements > 0 else 0.0

    def backward(self, predictions, targets):
        # dL/dY = 2 * (Y - T) / N, where N is total elements in batch
        if not predictions or not targets: return []

        is_single_sample = not isinstance(predictions[0], list)
        if is_single_sample:
            predictions = [predictions]
            targets = [targets]
            
        num_elements_in_batch = len(predictions) * len(predictions[0])
        
        grad = []
        for i in range(len(predictions)):
            row_grad = []
            for j in range(len(predictions[i])):
                g = 2 * (predictions[i][j] - targets[i][j]) / num_elements_in_batch
                row_grad.append(g)
            grad.append(row_grad)

        if is_single_sample:
            return grad[0]
        return grad


class CategoricalCrossEntropyLoss: # For classification
    """Categorical Cross-Entropy Loss (often used with Softmax output)."""
    def forward(self, predictions, targets):
        # predictions are probabilities (after softmax), targets are one-hot encoded
        # L = -sum(target_i * log(prediction_i))
        epsilon = 1e-10 # To prevent log(0)
        if not predictions or not targets: return 0.0
        
        is_single_sample = not isinstance(predictions[0], list)
        if is_single_sample:
            predictions = [predictions]
            targets = [targets]

        loss_sum = 0.0
        num_samples = len(predictions)
        for i in range(num_samples):
            for j in range(len(predictions[i])):
                if targets[i][j] == 1: # Only care about the probability of the true class
                    loss_sum += -targets[i][j] * math.log(max(predictions[i][j], epsilon))
        return loss_sum / num_samples if num_samples > 0 else 0.0

    def backward(self, predictions, targets):
        # For Softmax + CrossEntropy, dL/dZ (gradient w.r.t. pre-softmax logits Z)
        # is simply (predictions - targets) / num_samples.
        if not predictions or not targets: return []

        is_single_sample = not isinstance(predictions[0], list)
        if is_single_sample:
            predictions = [predictions]
            targets = [targets]

        grad = []
        num_samples = len(predictions)
        for i in range(num_samples):
            row_grad = []
            for j in range(len(predictions[i])):
                g = (predictions[i][j] - targets[i][j]) / num_samples
                row_grad.append(g)
            grad.append(row_grad)
        
        if is_single_sample:
            return grad[0]
        return grad


class SGD: # This would be `optimizers.py`'s SGD
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, parameters):
        """
        Updates parameters using their accumulated gradients.
        `parameters` is a list of Parameter objects (each holding data and grad).
        """
        for param in parameters:
            if not param.grad: # Skip if no gradient computed (e.g., if it's not a learnable param)
                continue
            
            # Param.data and param.grad are lists of lists (matrix) or list (vector)
            if isinstance(param.data[0], list): # Matrix
                updated_data = []
                for i in range(len(param.data)):
                    row = []
                    for j in range(len(param.data[i])):
                        row.append(param.data[i][j] - self.learning_rate * param.grad[i][j])
                    updated_data.append(row)
                param.data = updated_data
            else: # Vector
                param.data = [param.data[i] - self.learning_rate * param.grad[i] for i in range(len(param.data))]

class AccuracyMetric: # This would be `metrics.py`'s Accuracy
    """Calculates classification accuracy."""
    def compute(self, predictions, targets):
        if not predictions or not targets: return 0.0

        is_single_sample = not isinstance(predictions[0], list)
        if is_single_sample:
            predictions = [predictions]
            targets = [targets]

        correct_predictions = 0
        total_samples = 0
        for i in range(len(predictions)):
            # Find the index of the max probability (predicted class)
            predicted_class_idx = 0
            max_pred_val = -float('inf')
            for j in range(len(predictions[i])):
                if predictions[i][j] > max_pred_val:
                    max_pred_val = predictions[i][j]
                    predicted_class_idx = j
            
            # Find the index of the true class (assuming one-hot targets)
            true_class_idx = 0
            for j in range(len(targets[i])):
                if targets[i][j] == 1:
                    true_class_idx = j
                    break # Assuming one-hot, there's only one '1'

            if predicted_class_idx == true_class_idx:
                correct_predictions += 1
            total_samples += 1
        
        return correct_predictions / total_samples if total_samples > 0 else 0.0


# --- Trainer Class ---

class Trainer:
    """
    Handles the training loop and evaluation process for all AI models.
    Supports arbitrary models, optimizers, loss functions, and metrics
    as long as they adhere to the expected interfaces (forward/backward/step/compute).
    """
    def __init__(self, model, optimizer, loss_fn, metrics=None):
        """
        Initializes the Trainer.
        Args:
            model: An instance of a custom model (e.g., NeuralNetwork, CNN, GNN).
                   Must have `forward()`, `backward()`, `get_parameters()`, `zero_grad()` methods.
            optimizer: An instance of a custom optimizer (e.g., SGD).
                       Must have a `step()` method that takes model parameters.
            loss_fn: An instance of a custom loss function (e.g., MSELoss, CrossEntropyLoss).
                     Must have `forward()` and `backward()` methods.
            metrics: An optional list of custom metric instances (e.g., AccuracyMetric).
                     Each metric must have a `compute()` method.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics is not None else []
        self.history = {'train_loss': [], 'eval_loss': [], 'eval_metrics': {}}
        for metric in self.metrics:
            self.history['eval_metrics'][metric.__class__.__name__] = []

    def _batch_data(self, data, batch_size):
        """Generator to yield batches of data."""
        # data is expected to be a list of tuples: [(features, labels), ...]
        num_samples = len(data)
        for i in range(0, num_samples, batch_size):
            batch = data[i:min(i + batch_size, num_samples)]
            # Unzip features and labels from the batch
            batch_features = [sample[0] for sample in batch]
            batch_labels = [sample[1] for sample in batch]
            yield batch_features, batch_labels

    def train_epoch(self, train_data, batch_size):
        """
        Performs one full training epoch.
        Args:
            train_data: List of (features, labels) tuples for training.
            batch_size: Number of samples per batch.
        Returns:
            Average loss for the epoch.
        """
        total_loss = 0.0
        num_batches = 0

        for batch_features, batch_labels in self._batch_data(train_data, batch_size):
            self.model.zero_grad() # Clear gradients from previous step

            # Forward pass
            predictions = self.model.forward(batch_features)

            # Calculate loss
            loss = self.loss_fn.forward(predictions, batch_labels)
            total_loss += loss

            # Backward pass (compute gradients)
            grad_loss = self.loss_fn.backward(predictions, batch_labels)
            self.model.backward(grad_loss)

            # Update model parameters
            self.optimizer.step(self.model.get_parameters())
            
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, eval_data, batch_size):
        """
        Evaluates the model on the given evaluation data.
        Args:
            eval_data: List of (features, labels) tuples for evaluation.
            batch_size: Number of samples per batch.
        Returns:
            Tuple of (average_loss, dict_of_average_metrics).
        """
        total_loss = 0.0
        num_batches = 0
        metric_sums = {metric.__class__.__name__: 0.0 for metric in self.metrics}
        
        # Set model to evaluation mode (if your model had dropout/batchnorm)
        # For our simple model, no special eval mode needed.

        for batch_features, batch_labels in self._batch_data(eval_data, batch_size):
            # Forward pass (no gradient computation needed for evaluation)
            predictions = self.model.forward(batch_features)
            
            # Calculate loss
            loss = self.loss_fn.forward(predictions, batch_labels)
            total_loss += loss

            # Calculate metrics
            for metric in self.metrics:
                metric_value = metric.compute(predictions, batch_labels)
                metric_sums[metric.__class__.__name__] += metric_value
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {name: s / num_batches for name, s in metric_sums.items()} if num_batches > 0 else {name: 0.0 for name in metric_sums}

        return avg_loss, avg_metrics

    def train(self, train_data, eval_data, epochs, batch_size, verbose=True):
        """
        Main training loop.
        Args:
            train_data: List of (features, labels) tuples for training.
            eval_data: List of (features, labels) tuples for evaluation.
            epochs: Number of epochs to train for.
            batch_size: Number of samples per batch.
            verbose: If True, print training progress.
        """
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_data, batch_size)
            eval_loss, eval_metrics = self.evaluate(eval_data, batch_size)

            self.history['train_loss'].append(train_loss)
            self.history['eval_loss'].append(eval_loss)
            for name, value in eval_metrics.items():
                self.history['eval_metrics'][name].append(value)

            if verbose:
                metric_str = ", ".join([f"{name}: {value:.4f}" for name, value in eval_metrics.items()])
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, {metric_str}")

        print("Training complete.")


if __name__ == '__main__':
    # --- Example Usage for a self-contained test ---
    print("Running self-contained Trainer example (XOR Problem)...")

    # 1. Generate some dummy data (binary classification)
    # 2 features, 2 classes (one-hot encoded)
    # Dataset will be (input_vector, one_hot_label_vector)
    
    # Simple XOR-like data
    train_data = [
        ([0.0, 0.0], [1.0, 0.0]), # Class 0
        ([0.0, 1.0], [0.0, 1.0]), # Class 1
        ([1.0, 0.0], [0.0, 1.0]), # Class 1
        ([1.0, 1.0], [1.0, 0.0]), # Class 0
    ] * 200 # Make it a bit larger to simulate a dataset

    eval_data = [
        ([0.1, 0.1], [1.0, 0.0]),
        ([0.9, 0.1], [0.0, 1.0]),
        ([0.1, 0.9], [0.0, 1.0]),
        ([0.9, 0.9], [1.0, 0.0]),
    ] * 50

    # 2. Define the model architecture (a simple MLP)
    # Input: 2 features
    # Hidden Layer: 4 neurons, ReLU activation
    # Output Layer: 2 neurons (for 2 classes), Softmax activation
    model = NeuralNetwork(layers=[
        Linear(input_size=2, output_size=4),
        ReLU(),
        Linear(input_size=4, output_size=2),
        Softmax() # Softmax typically goes after the last linear layer
    ])

    # 3. Define optimizer, loss function, and metrics
    optimizer = SGD(learning_rate=0.1)
    loss_fn = CategoricalCrossEntropyLoss()
    metrics = [AccuracyMetric()]

    # 4. Initialize the Trainer
    trainer = Trainer(model, optimizer, loss_fn, metrics)

    # 5. Train the model
    epochs = 100
    batch_size = 4
    trainer.train(train_data, eval_data, epochs, batch_size)

    # 6. Post-training evaluation on a few samples
    print("\n--- Final Evaluation on Test Samples ---")
    test_samples = [
        ([0.0, 0.0], [1.0, 0.0]), # Expected Class 0
        ([0.0, 1.0], [0.0, 1.0]), # Expected Class 1
        ([1.0, 0.0], [0.0, 1.0]), # Expected Class 1
        ([1.0, 1.0], [1.0, 0.0]), # Expected Class 0
        ([0.5, 0.5], [1.0, 0.0]), # Ambiguous, should lean towards a class
    ]
    
    for inputs, true_label_one_hot in test_samples:
        prediction_probs = model.forward([inputs])[0] # Get probabilities for a single sample
        
        # Determine predicted class index
        predicted_class_idx = 0
        max_prob = -1.0
        for i, prob in enumerate(prediction_probs):
            if prob > max_prob:
                max_prob = prob
                predicted_class_idx = i

        # Determine true class index
        true_class_idx = true_label_one_hot.index(1.0)
        
        print(f"Input: {inputs}, True Label: Class {true_class_idx}, "
              f"Predicted Probs: [{prediction_probs[0]:.2f}, {prediction_probs[1]:.2f}], "
              f"Predicted Class: Class {predicted_class_idx}")

    print("\nTraining History (First 5 and Last 5 Epochs):")
    for i in range(len(trainer.history['train_loss'])):
        epoch_str = f"Epoch {i+1}: Train Loss={trainer.history['train_loss'][i]:.4f}, Eval Loss={trainer.history['eval_loss'][i]:.4f}"
        for metric_name, values in trainer.history['eval_metrics'].items():
            epoch_str += f", Eval {metric_name}={values[i]:.4f}"
        
        if i < 5 or i >= epochs - 5:
            print(epoch_str)
        elif i == 5 and epochs > 10: # Only print ellipsis if there are skipped epochs
            print("...")
class GNNLayer:
    """
    Implements a basic Graph Neural Network layer from scratch,
    including forward and backward passes.
    """

    def __init__(self, input_dim, output_dim, activation='relu', seed=None):
        """
        Initializes the GNN layer.

        Args:
            input_dim (int): Dimension of input node features.
            output_dim (int): Dimension of output node embeddings.
            activation (str): Activation function to use ('relu' supported).
            seed (int, optional): Seed for weight initialization for reproducibility.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation

        # Initialize weights and biases
        self.weights = self._initialize_weights(input_dim, output_dim, seed)
        # Bias is a row vector broadcasted across nodes
        self.bias = self._initialize_weights(1, output_dim, seed + 1 if seed is not None else None)[0]

        # Store gradients for weights and bias
        self.grad_weights = [[0.0 for _ in range(output_dim)] for _ in range(input_dim)]
        self.grad_bias = [0.0 for _ in range(output_dim)]

        # Store intermediate values for backward pass
        self.last_input_features = None
        self.last_transformed_features = None
        self.last_aggregated_features = None
        self.last_output_before_activation = None
        self.last_adjacency_matrix = None

    def _random_uniform(self, low, high, seed=None):
        """
        Basic Linear Congruential Generator (LCG) for random numbers.
        Used for weight initialization without external dependencies.
        """
        # Parameters for LCG from Wikipedia/Numerical Recipes
        a = 1664525
        c = 1013904223
        m = 2**32  # Modulus

        if seed is None:
            # Fallback to system time-based seed if not provided.
            # Requires 'time' standard library module.
            import time
            seed = int(time.time() * 1000) % m

        current_seed = seed
        # Warm-up LCG to improve randomness of initial samples
        for _ in range(10):
            current_seed = (a * current_seed + c) % m

        # Generator function to yield random numbers
        while True:
            current_seed = (a * current_seed + c) % m
            yield low + (current_seed / m) * (high - low)

    def _initialize_weights(self, rows, cols, seed=None):
        """
        Initializes weights using a simple Glorot/Xavier uniform heuristic.
        """
        # Heuristic for uniform initialization: sqrt(6 / (fan_in + fan_out))
        # For ReLU, He initialization (sqrt(2 / fan_in)) is common.
        # We use a combined approach for simplicity with custom PRNG.
        limit = (6 / (self.input_dim + self.output_dim))**0.5
        
        rand_gen = self._random_uniform(-limit, limit, seed)
        return [[next(rand_gen) for _ in range(cols)] for _ in range(rows)]

    def _matrix_multiply(self, A, B):
        """
        Performs matrix multiplication C = A @ B.
        A: (rows_A x cols_A)
        B: (rows_B x cols_B)
        Result: (rows_A x cols_B)
        """
        if not A or not B:
            raise ValueError("Matrices cannot be empty.")
        if not A[0] or not B[0]:
            raise ValueError("Matrices cannot have empty rows/columns.")

        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])

        if cols_A != rows_B:
            raise ValueError(
                f"Cannot multiply matrices: A has {cols_A} columns, B has {rows_B} rows. "
                "Number of columns in A must match number of rows in B."
            )

        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def _matrix_add_bias(self, matrix, bias_vector):
        """
        Adds a bias vector to each row of a matrix (broadcasting).
        matrix: (rows x cols)
        bias_vector: (cols,)
        Result: (rows x cols)
        """
        rows = len(matrix)
        cols = len(matrix[0])
        if cols != len(bias_vector):
            raise ValueError("Bias vector dimension must match matrix column dimension.")
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = matrix[i][j] + bias_vector[j]
        return result
    
    def _elementwise_multiply(self, A, B):
        """
        Performs element-wise multiplication C = A * B.
        A: (rows x cols)
        B: (rows x cols)
        Result: (rows x cols)
        """
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ValueError("Matrices must have the same dimensions for element-wise multiplication.")
        
        rows = len(A)
        cols = len(A[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] * B[i][j]
        return result

    def _transpose(self, matrix):
        """
        Transposes a matrix.
        matrix: (rows x cols)
        Result: (cols x rows)
        """
        if not matrix:
            return []
        if not matrix[0]:
            return [[] for _ in range(len(matrix))] # Handle empty inner lists gracefully
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        result = [[0.0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                result[j][i] = matrix[i][j]
        return result
        
    def _relu(self, matrix):
        """
        Applies the Rectified Linear Unit (ReLU) activation function element-wise.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = max(0.0, matrix[i][j])
        return result

    def _relu_derivative(self, matrix):
        """
        Computes the derivative of the ReLU function element-wise.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = 1.0 if matrix[i][j] > 0 else 0.0
        return result

    def forward(self, node_features, adjacency_matrix):
        """
        Performs the forward pass of the GNN layer.

        Args:
            node_features (list of list): Node features matrix (N_nodes x F_in).
            adjacency_matrix (list of list): Adjacency matrix (N_nodes x N_nodes).
                                             Should typically be normalized (e.g., D^-1 A or D^-0.5 A D^-0.5)
                                             and include self-loops (A + I) before passing.
        Returns:
            list of list: Output node embeddings (N_nodes x F_out).
        """
        # Store inputs for backward pass
        self.last_input_features = node_features
        self.last_adjacency_matrix = adjacency_matrix

        # 1. Feature Transformation: H_transformed = X @ W
        # Each node's features are linearly transformed.
        transformed_features = self._matrix_multiply(node_features, self.weights)
        self.last_transformed_features = transformed_features

        # 2. Aggregation: H_aggregated = A @ H_transformed
        # Each node aggregates transformed features from its neighbors (and itself, if A includes self-loops).
        aggregated_features = self._matrix_multiply(adjacency_matrix, transformed_features)
        self.last_aggregated_features = aggregated_features

        # 3. Add Bias: H_out_before_act = H_aggregated + B
        output_before_activation = self._matrix_add_bias(aggregated_features, self.bias)
        self.last_output_before_activation = output_before_activation

        # 4. Activation function
        if self.activation_name == 'relu':
            output = self._relu(output_before_activation)
        else:
            # Default to linear activation if no specific activation is implemented
            output = output_before_activation
        
        return output

    def backward(self, grad_output):
        """
        Performs the backward pass to compute gradients for weights, bias,
        and the gradient with respect to the input features of this layer.

        Args:
            grad_output (list of list): Gradient of the loss with respect to the output
                                        of this layer (N_nodes x F_out).
        
        Returns:
            list of list: Gradient of the loss with respect to the input features
                          of this layer (N_nodes x F_in).
        """
        # Derivative of activation function
        if self.activation_name == 'relu':
            grad_activation = self._relu_derivative(self.last_output_before_activation)
            grad_intermediate = self._elementwise_multiply(grad_output, grad_activation)
        else:
            grad_intermediate = grad_output # For linear activation, derivative is 1

        # Gradient w.r.t. bias (dL/dB)
        # Sum grad_intermediate along the node dimension (axis 0)
        num_nodes = len(grad_intermediate)
        output_dim = len(grad_intermediate[0])
        self.grad_bias = [0.0] * output_dim
        for j in range(output_dim):
            for i in range(num_nodes):
                self.grad_bias[j] += grad_intermediate[i][j]

        # Gradient w.r.t. aggregated_features (dL/d(A @ H_transformed))
        # Since output_before_activation = aggregated_features + bias, dL/d(aggregated_features) = dL/d(output_before_activation)
        grad_aggregated_features = grad_intermediate

        # Gradient w.r.t. transformed_features (dL/dH_transformed)
        # H_aggregated = A @ H_transformed  => dL/dH_transformed = A_T @ dL/dH_aggregated
        adj_T = self._transpose(self.last_adjacency_matrix)
        grad_transformed_features = self._matrix_multiply(adj_T, grad_aggregated_features)

        # Gradient w.r.t. weights (dL/dW)
        # H_transformed = X @ W => dL/dW = X_T @ dL/dH_transformed
        input_features_T = self._transpose(self.last_input_features)
        self.grad_weights = self._matrix_multiply(input_features_T, grad_transformed_features)

        # Gradient w.r.t. input_features (dL/dX)
        # H_transformed = X @ W => dL/dX = dL/dH_transformed @ W_T
        weights_T = self._transpose(self.weights)
        grad_input = self._matrix_multiply(grad_transformed_features, weights_T)

        return grad_input
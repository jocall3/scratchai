import random
import math

# --- Basic Math Operations (no numpy or external libraries) ---

def _create_matrix(rows, cols, value=0.0):
    """Creates a matrix (list of lists) initialized with a given value."""
    return [[value for _ in range(cols)] for _ in range(rows)]

def _matmul(A, B):
    """Performs matrix multiplication C = A * B.
    A: (rows_a x cols_a)
    B: (rows_b x cols_b)
    Result: (rows_a x cols_b)
    """
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])

    if cols_a != rows_b:
        raise ValueError(f"Matrix dimensions mismatch for matmul: A({rows_a}x{cols_a}) B({rows_b}x{cols_b})")

    C = _create_matrix(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            sum_val = 0.0
            for k in range(cols_a): # or rows_b
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
    return C

def _add(A, B):
    """Performs element-wise matrix addition A + B."""
    rows = len(A)
    cols = len(A[0])
    C = _create_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] + B[i][j]
    return C

def _vector_add(v1, v2):
    """Performs element-wise vector addition v1 + v2."""
    if len(v1) != len(v2):
        raise ValueError("Vector lengths must match for addition.")
    return [v1[i] + v2[i] for i in range(len(v1))]

def _vector_matrix_multiply(vector, matrix):
    """Performs vector-matrix multiplication (row_vector @ matrix).
    vector: 1D list (1 x N)
    matrix: list of lists (N x M)
    Result: 1D list (1 x M)
    """
    N = len(vector)
    rows_mat = len(matrix)
    cols_mat = len(matrix[0])

    if N != rows_mat:
        raise ValueError(f"Vector length ({N}) must match matrix rows ({rows_mat}) for vector-matrix multiplication.")

    result_vec = [0.0] * cols_mat
    for j in range(cols_mat):
        sum_val = 0.0
        for i in range(N):
            sum_val += vector[i] * matrix[i][j]
        result_vec[j] = sum_val
    return result_vec

def _sigmoid(x):
    """Applies the sigmoid activation function element-wise to a list or scalar."""
    if isinstance(x, list):
        return [1.0 / (1.0 + math.exp(-val)) for val in x]
    return 1.0 / (1.0 + math.exp(-x))

def _tanh(x):
    """Applies the tanh activation function element-wise to a list or scalar."""
    if isinstance(x, list):
        return [math.tanh(val) for val in x]
    return math.tanh(x)

def _softmax(vector):
    """Applies the softmax function to a 1D list (vector) to get probabilities."""
    if not vector:
        return []
    # For numerical stability, subtract the maximum value from all elements
    max_val = max(vector)
    exp_values = [math.exp(val - max_val) for val in vector]
    sum_exp_values = sum(exp_values)
    # Handle case where sum_exp_values might be zero or very small due to underflow
    if sum_exp_values == 0:
        return [0.0] * len(vector)
    return [val / sum_exp_values for val in exp_values]

def _initialize_weights(rows, cols):
    """Initializes weights using a simplified Xavier uniform initialization."""
    limit = math.sqrt(6 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def _initialize_bias(size):
    """Initializes bias vector with zeros."""
    return [0.0 for _ in range(size)]

# --- Neural Network Layers (from scratch) ---

class EmbeddingLayer:
    """
    Converts input token indices into dense vector representations.
    """
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Weights are vocab_size x embedding_dim, each row is an embedding vector
        self.weights = _initialize_weights(vocab_size, embedding_dim)

    def forward(self, input_index):
        """
        Performs a lookup for the embedding vector corresponding to the input index.
        Args:
            input_index (int): An integer representing the token ID.
        Returns:
            list: The embedding vector (1D list).
        """
        if not (0 <= input_index < self.vocab_size):
            raise ValueError(f"Input index {input_index} out of vocab range [0, {self.vocab_size-1}]")
        return self.weights[input_index]

class DenseLayer:
    """
    A fully connected layer (linear transformation + optional activation).
    Output = Activation(Input @ Weights + Bias)
    """
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.weights = _initialize_weights(input_dim, output_dim)
        self.bias = _initialize_bias(output_dim)

    def forward(self, input_vector):
        """
        Performs the forward pass for the dense layer.
        Args:
            input_vector (list): A 1D list representing the input features.
        Returns:
            list: The output vector after transformation and activation.
        """
        if len(input_vector) != self.input_dim:
            raise ValueError(f"Input vector dimension mismatch. Expected {self.input_dim}, got {len(input_vector)}")

        # Perform W*x + b (input_vector @ self.weights + self.bias)
        output_vec_no_bias = _vector_matrix_multiply(input_vector, self.weights)
        output_vec = _vector_add(output_vec_no_bias, self.bias)

        # Apply activation function
        if self.activation == 'relu':
            return [max(0.0, x) for x in output_vec]
        elif self.activation == 'sigmoid':
            return _sigmoid(output_vec)
        elif self.activation == 'tanh':
            return _tanh(output_vec)
        elif self.activation is None:
            return output_vec
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

class RNNCustomCell:
    """
    A very basic, vanilla Recurrent Neural Network (RNN) cell.
    It computes the next hidden state based on the current input and previous hidden state.
    next_hidden = tanh(W_ih * input + b_ih + W_hh * prev_hidden + b_hh)
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weights for input to hidden transformation
        self.W_ih = _initialize_weights(input_dim, hidden_dim)
        self.b_ih = _initialize_bias(hidden_dim)

        # Weights for hidden to hidden transformation
        self.W_hh = _initialize_weights(hidden_dim, hidden_dim)
        self.b_hh = _initialize_bias(hidden_dim)

    def forward(self, input_vector, prev_hidden_state):
        """
        Performs the forward pass for one step of the RNN cell.
        Args:
            input_vector (list): The input at the current timestep (e.g., word embedding).
            prev_hidden_state (list): The hidden state from the previous timestep.
        Returns:
            list: The new hidden state for the current timestep.
        """
        if len(input_vector) != self.input_dim:
            raise ValueError(f"Input vector dimension mismatch. Expected {self.input_dim}, got {len(input_vector)}")
        if len(prev_hidden_state) != self.hidden_dim:
            raise ValueError(f"Previous hidden state dimension mismatch. Expected {self.hidden_dim}, got {len(prev_hidden_state)}")

        # Input to hidden transformation
        i_h = _vector_matrix_multiply(input_vector, self.W_ih)
        i_h = _vector_add(i_h, self.b_ih)

        # Hidden to hidden transformation
        h_h = _vector_matrix_multiply(prev_hidden_state, self.W_hh)
        h_h = _vector_add(h_h, self.b_hh)

        # Combine and apply activation (tanh is common for RNN)
        next_hidden_state_pre_activation = _vector_add(i_h, h_h)
        next_hidden_state = _tanh(next_hidden_state_pre_activation)

        return next_hidden_state

# --- Chatbot Model Integration ---

class ChatbotModel:
    """
    Integrates various neural network layers to form the core model for a chatbot.
    This model uses an EmbeddingLayer, a simple RNNCustomCell, and a DenseLayer
    with Softmax for output prediction.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Initialize the layers
        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)
        self.rnn_cell = RNNCustomCell(embedding_dim, hidden_dim)
        # The output layer projects the hidden state to the vocabulary size
        # Softmax will be applied explicitly after this layer.
        self.output_layer = DenseLayer(hidden_dim, vocab_size, activation=None)

    def initialize_hidden_state(self):
        """
        Initializes the hidden state to a vector of zeros.
        Returns:
            list: A 1D list of zeros of size hidden_dim.
        """
        return [0.0] * self.hidden_dim

    def forward(self, input_token_index, prev_hidden_state):
        """
        Performs a single forward pass through the chatbot model for one token.
        Args:
            input_token_index (int): The index of the current input token.
            prev_hidden_state (list): The hidden state from the previous timestep.
        Returns:
            tuple:
                - list: A 1D list of probabilities over the vocabulary for the next token.
                - list: The new hidden state computed by the RNN cell.
        """
        # 1. Get the embedding vector for the current input token
        embedded_input = self.embedding_layer.forward(input_token_index)

        # 2. Pass the embedded input and previous hidden state through the RNN cell
        #    to compute the new hidden state.
        next_hidden_state = self.rnn_cell.forward(embedded_input, prev_hidden_state)

        # 3. Project the new hidden state to the size of the vocabulary to get logits.
        logits = self.output_layer.forward(next_hidden_state)

        # 4. Apply softmax to the logits to convert them into a probability distribution
        #    over the entire vocabulary.
        output_probabilities = _softmax(logits)

        return output_probabilities, next_hidden_state

    def predict_next_token(self, input_token_index, prev_hidden_state):
        """
        A convenience method for inference, simply calls the forward pass.
        Args:
            input_token_index (int): The index of the current input token.
            prev_hidden_state (list): The hidden state from the previous timestep.
        Returns:
            tuple:
                - list: A 1D list of probabilities over the vocabulary for the next token.
                - list: The new hidden state computed by the RNN cell.
        """
        return self.forward(input_token_index, prev_hidden_state)
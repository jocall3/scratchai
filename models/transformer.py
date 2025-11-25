import math

class MatrixOperations:
    """
    A collection of static methods for basic matrix and vector operations.
    Implemented from scratch to adhere to the "no dependencies" constraint.
    """

    @staticmethod
    def matmul(A, B):
        """
        Performs matrix multiplication C = A @ B.
        A: list of lists (rowsA, colsA)
        B: list of lists (rowsB, colsB)
        Result: list of lists (rowsA, colsB)
        """
        if not A or not B:
            return []
        if not A[0] or not B[0]:
            return []

        rowsA = len(A)
        colsA = len(A[0])
        rowsB = len(B)
        colsB = len(B[0])

        if colsA != rowsB:
            raise ValueError(f"Matrix dimensions mismatch for matmul: A columns ({colsA}) != B rows ({rowsB})")

        C = [[0 for _ in range(colsB)] for _ in range(rowsA)]

        for i in range(rowsA):
            for j in range(colsB):
                for k in range(colsA):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    @staticmethod
    def add(A, B):
        """
        Performs element-wise addition C = A + B.
        A, B: list of lists (rows, cols)
        Result: list of lists (rows, cols)
        """
        if not A or not B:
            return []
        rowsA = len(A)
        colsA = len(A[0])
        rowsB = len(B)
        colsB = len(B[0])

        if rowsA != rowsB or colsA != colsB:
            raise ValueError("Matrix dimensions mismatch for element-wise add")

        C = [[0 for _ in range(colsA)] for _ in range(rowsA)]
        for i in range(rowsA):
            for j in range(colsA):
                C[i][j] = A[i][j] + B[i][j]
        return C

    @staticmethod
    def scalar_multiply(A, scalar):
        """
        Performs element-wise multiplication by a scalar.
        A: list of lists (rows, cols)
        scalar: a number
        Result: list of lists (rows, cols)
        """
        if not A:
            return []
        rows = len(A)
        cols = len(A[0])
        C = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                C[i][j] = A[i][j] * scalar
        return C

    @staticmethod
    def transpose(A):
        """
        Transposes a matrix.
        A: list of lists (rows, cols)
        Result: list of lists (cols, rows)
        """
        if not A:
            return []
        rows = len(A)
        cols = len(A[0])
        C = [[0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                C[j][i] = A[i][j]
        return C

    @staticmethod
    def softmax(vector_or_matrix):
        """
        Applies the softmax function.
        vector_or_matrix: list of numbers (vector) or list of lists (matrix)
        """
        if not vector_or_matrix:
            return []
        if isinstance(vector_or_matrix[0], list): # Matrix (batch of vectors)
            results = []
            for row in vector_or_matrix:
                exp_row = [math.exp(x) for x in row]
                sum_exp_row = sum(exp_row)
                if sum_exp_row == 0: # Avoid division by zero
                    results.append([0.0 for _ in exp_row])
                else:
                    results.append([x / sum_exp_row for x in exp_row])
            return results
        else: # Vector
            exp_vector = [math.exp(x) for x in vector_or_matrix]
            sum_exp_vector = sum(exp_vector)
            if sum_exp_vector == 0: # Avoid division by zero
                return [0.0 for _ in exp_vector]
            return [x / sum_exp_vector for x in exp_vector]

    @staticmethod
    def relu(matrix):
        """
        Applies the ReLU activation function element-wise.
        matrix: list of lists (rows, cols)
        """
        if not matrix:
            return []
        rows = len(matrix)
        cols = len(matrix[0])
        C = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                C[i][j] = max(0, matrix[i][j])
        return C

    @staticmethod
    def mean(data):
        """Calculates the mean of a list of numbers."""
        if not data:
            return 0.0
        return sum(data) / len(data)

    @staticmethod
    def variance(data, mean_val):
        """Calculates the population variance of a list of numbers."""
        if not data:
            return 0.0
        return sum([(x - mean_val)**2 for x in data]) / len(data)

    @staticmethod
    def multiply(A, B):
        """
        Performs element-wise multiplication C = A * B.
        A, B: list of lists (rows, cols)
        Result: list of lists (rows, cols)
        """
        if not A or not B:
            return []
        rowsA = len(A)
        colsA = len(A[0])
        rowsB = len(B)
        colsB = len(B[0])

        if rowsA != rowsB or colsA != colsB:
            raise ValueError("Matrix dimensions mismatch for element-wise multiply")

        C = [[0 for _ in range(colsA)] for _ in range(rowsA)]
        for i in range(rowsA):
            for j in range(colsA):
                C[i][j] = A[i][j] * B[i][j]
        return C

    @staticmethod
    def concatenate(matrices, axis=1):
        """
        Concatenates a list of 2D matrices along a specified axis.
        matrices: list of list of lists (2D matrices)
        axis: 0 for row-wise, 1 for column-wise concatenation.
        """
        if not matrices:
            return []
        if axis == 1: # Concatenate columns (horizontally)
            result = []
            num_rows = len(matrices[0])
            for i in range(num_rows): # Iterate over rows
                row = []
                for mat in matrices:
                    if len(mat) != num_rows:
                        raise ValueError("Matrices must have the same number of rows for column-wise concatenation.")
                    row.extend(mat[i])
                result.append(row)
            return result
        elif axis == 0: # Concatenate rows (vertically)
            result = []
            num_cols = len(matrices[0][0])
            for mat in matrices:
                if len(mat) > 0 and len(mat[0]) != num_cols:
                    raise ValueError("Matrices must have the same number of columns for row-wise concatenation.")
                result.extend(mat)
            return result
        else:
            raise ValueError("Unsupported axis for concatenation. Only 0 or 1.")

    @staticmethod
    def get_slice(matrix, row_start, row_end, col_start, col_end):
        """
        Extracts a slice from a matrix.
        matrix: list of lists
        row_start, row_end: slice for rows (exclusive end)
        col_start, col_end: slice for columns (exclusive end)
        """
        if not matrix:
            return []
        
        sliced_rows = matrix[row_start:row_end]
        if not sliced_rows:
            return []
        
        result = []
        for row in sliced_rows:
            result.append(row[col_start:col_end])
        return result


class LinearLayer:
    """
    A basic linear transformation layer (y = x @ W.T + b).
    Initializes weights and biases with simple deterministic values.
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize weights (output_dim, input_dim) and biases (1, output_dim)
        # Using a deterministic pattern for weights/biases to avoid randomness for 'from scratch' constraint.
        # In a real scenario, these would be initialized randomly (e.g., Kaiming, Xavier).
        self.weights = [[(i * input_dim + j) * 0.01 for j in range(input_dim)] for i in range(output_dim)]
        self.bias = [[i * 0.01 for i in range(output_dim)]]

    def forward(self, x):
        """
        Performs the forward pass of the linear layer.
        x: list of lists (batch_size, input_dim)
        Result: list of lists (batch_size, output_dim)
        """
        weights_T = MatrixOperations.transpose(self.weights)
        matmul_result = MatrixOperations.matmul(x, weights_T)

        # Add bias: self.bias (1, output_dim) is broadcasted to each row of matmul_result.
        output = []
        for row in matmul_result:
            # Add returns list of lists, so we take the first (and only) row.
            output_row = MatrixOperations.add([row], self.bias)[0]
            output.append(output_row)
        return output


class LayerNorm:
    """
    A basic Layer Normalization implementation.
    gamma and beta are learnable parameters.
    """
    def __init__(self, features_dim, epsilon=1e-5):
        self.features_dim = features_dim
        self.epsilon = epsilon
        # gamma (scale) initialized to ones, beta (shift) initialized to zeros.
        self.gamma = [[1.0 for _ in range(features_dim)]] # shape (1, features_dim)
        self.beta = [[0.0 for _ in range(features_dim)]]  # shape (1, features_dim)

    def forward(self, x):
        """
        Performs the forward pass of Layer Normalization.
        x: list of lists (batch_size, features_dim)
        Result: list of lists (batch_size, features_dim)
        """
        normalized_x = []
        for row in x:
            mean_val = MatrixOperations.mean(row)
            variance_val = MatrixOperations.variance(row, mean_val)
            std_dev = math.sqrt(variance_val + self.epsilon)

            norm_row = [(val - mean_val) / std_dev for val in row]
            
            # Apply gamma and beta (broadcasting element-wise multiplication and addition)
            scaled_row = MatrixOperations.add(
                MatrixOperations.multiply([norm_row], self.gamma),
                self.beta
            )[0] # Take the first (and only) row from the result of adding 1xN matrices
            normalized_x.append(scaled_row)
        return normalized_x


class MultiHeadSelfAttention:
    """
    Multi-Head Self-Attention mechanism.
    Simplified to process a single sequence (batch_size = 1 implicitly).
    """
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Linear layers for Q, K, V projections for all heads combined
        self.wq = LinearLayer(embed_dim, embed_dim)
        self.wk = LinearLayer(embed_dim, embed_dim)
        self.wv = LinearLayer(embed_dim, embed_dim)

        self.wo = LinearLayer(embed_dim, embed_dim) # Output linear layer after concatenation

    def _scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Performs scaled dot-product attention for a single head.
        Q, K, V: list of lists (seq_len, head_dim)
        mask: Optional, list of lists (seq_len, seq_len) to mask future tokens or padding.
              Non-zero values in mask will cause a large negative number to be added to scores.
        """
        # Calculate attention scores: Q @ K.T
        # K.T will be (head_dim, seq_len)
        scores = MatrixOperations.matmul(Q, MatrixOperations.transpose(K))
        
        # Scale scores by sqrt(head_dim)
        dk = self.head_dim
        scaled_scores = MatrixOperations.scalar_multiply(scores, 1.0 / math.sqrt(dk))

        # Apply mask if provided
        if mask is not None:
            # Assume mask is (seq_len, seq_len)
            for i in range(len(scaled_scores)):
                for j in range(len(scaled_scores[0])):
                    if mask[i][j] != 0: # If mask indicates to hide this position
                        scaled_scores[i][j] += -1e9 # Add large negative to zero out after softmax

        attention_weights = MatrixOperations.softmax(scaled_scores)

        # Apply attention weights to V: attention_weights @ V
        output = MatrixOperations.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x, mask=None):
        """
        Performs the forward pass of multi-head self-attention.
        x: list of lists (seq_len, embed_dim)
        mask: Optional, list of lists for attention masking.
        Result: list of lists (seq_len, embed_dim), list of attention weights (for each head)
        """
        seq_len = len(x)

        # Project input to Q, K, V spaces
        # All Q, K, V are initially (seq_len, embed_dim)
        Q = self.wq.forward(x)
        K = self.wk.forward(x)
        V = self.wv.forward(x)

        # Split Q, K, V into multiple heads
        Q_heads, K_heads, V_heads = [], [], []
        for h in range(self.num_heads):
            # Each head takes a slice of the embed_dim dimension
            q_head = MatrixOperations.get_slice(Q, 0, seq_len, h * self.head_dim, (h + 1) * self.head_dim)
            k_head = MatrixOperations.get_slice(K, 0, seq_len, h * self.head_dim, (h + 1) * self.head_dim)
            v_head = MatrixOperations.get_slice(V, 0, seq_len, h * self.head_dim, (h + 1) * self.head_dim)
            Q_heads.append(q_head)
            K_heads.append(k_head)
            V_heads.append(v_head)
        
        # Apply scaled dot-product attention for each head
        head_outputs = []
        attention_weight_list = [] # Stores weights for all heads
        for h in range(self.num_heads):
            output_h, attn_w_h = self._scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], mask)
            head_outputs.append(output_h)
            attention_weight_list.append(attn_w_h)

        # Concatenate head outputs along the embed_dim dimension
        concatenated_output = MatrixOperations.concatenate(head_outputs, axis=1)

        # Final linear layer projection
        output = self.wo.forward(concatenated_output)
        return output, attention_weight_list


class FeedForward:
    """
    A simple position-wise feed-forward network with two linear layers and ReLU activation.
    """
    def __init__(self, embed_dim, ff_dim):
        self.linear1 = LinearLayer(embed_dim, ff_dim)
        self.linear2 = LinearLayer(ff_dim, embed_dim)

    def forward(self, x):
        """
        Performs the forward pass of the feed-forward network.
        x: list of lists (seq_len, embed_dim)
        Result: list of lists (seq_len, embed_dim)
        """
        x = self.linear1.forward(x)
        x = MatrixOperations.relu(x)
        x = self.linear2.forward(x)
        return x


class PositionalEncoding:
    """
    Generates sinusoidal positional encodings to inject position information into embeddings.
    """
    def __init__(self, embed_dim, max_seq_len):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.positional_encoding_matrix = self._generate_positional_encoding()

    def _generate_positional_encoding(self):
        """
        Generates the static positional encoding matrix.
        """
        pe = [[0.0 for _ in range(self.embed_dim)] for _ in range(self.max_seq_len)]
        for pos in range(self.max_seq_len):
            for i in range(self.embed_dim):
                denominator = 10000 ** ((2 * (i // 2)) / self.embed_dim)
                if i % 2 == 0: # Even indices use sine
                    pe[pos][i] = math.sin(pos / denominator)
                else: # Odd indices use cosine
                    pe[pos][i] = math.cos(pos / denominator)
        return pe

    def forward(self, seq_len):
        """
        Returns the positional encodings for a given sequence length.
        seq_len: integer, current sequence length.
        Result: list of lists (seq_len, embed_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        return MatrixOperations.get_slice(self.positional_encoding_matrix, 0, seq_len, 0, self.embed_dim)


class EncoderBlock:
    """
    A single Transformer Encoder Block, consisting of:
    1. Multi-Head Self-Attention
    2. Add & Norm (Residual connection + Layer Normalization)
    3. Position-wise Feed-Forward Network
    4. Add & Norm
    Uses pre-normalization for residual connections.
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)

        self.layernorm1 = LayerNorm(embed_dim)
        self.layernorm2 = LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        Performs the forward pass of an encoder block.
        x: list of lists (seq_len, embed_dim)
        mask: Optional, list of lists for self-attention masking.
        Result: list of lists (seq_len, embed_dim)
        """
        # Multi-Head Self-Attention with pre-normalization and residual connection
        norm_x1 = self.layernorm1.forward(x)
        attn_output, _ = self.mha.forward(norm_x1, mask)
        x = MatrixOperations.add(x, attn_output) # Residual connection

        # Feed-Forward Network with pre-normalization and residual connection
        norm_x2 = self.layernorm2.forward(x)
        ffn_output = self.ffn.forward(norm_x2)
        x = MatrixOperations.add(x, ffn_output) # Residual connection
        return x


class TransformerEncoder:
    """
    A full Transformer Encoder model.
    Stacks multiple Encoder Blocks.
    Includes token embedding and positional encoding.
    """
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, max_seq_len, vocab_size):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Token embedding layer. Represented as a matrix where each row is an embedding for a token ID.
        # Initialized deterministically. In a real scenario, these would be learned.
        self.token_embeddings = [[(i * embed_dim + j) * 0.01 for j in range(embed_dim)] for i in range(vocab_size)]

        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)

        self.encoder_blocks = []
        for _ in range(num_layers):
            self.encoder_blocks.append(EncoderBlock(embed_dim, num_heads, ff_dim))

        self.final_layernorm = LayerNorm(embed_dim)

    def forward(self, input_tokens, mask=None):
        """
        Performs the forward pass of the Transformer Encoder.
        input_tokens: list of integers, representing token IDs (seq_len)
        mask: Optional, list of lists for self-attention masking within encoder blocks.
        Result: list of lists (seq_len, embed_dim), the final encoder output.
        """
        seq_len = len(input_tokens)

        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Get token embeddings for the input sequence
        # token_embed: (seq_len, embed_dim)
        token_embed = [self.token_embeddings[token_id] for token_id in input_tokens]

        # Get positional encodings for the current sequence length
        pos_embed = self.positional_encoding.forward(seq_len)

        # Add token and positional embeddings
        x = MatrixOperations.add(token_embed, pos_embed)

        # Pass through all encoder blocks
        for block in self.encoder_blocks:
            x = block.forward(x, mask)
        
        # Apply final layer normalization
        x = self.final_layernorm.forward(x)

        return x
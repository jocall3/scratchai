import math

# --- Vector Operations ---

def vector_add(v1, v2):
    """Adds two vectors element-wise.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        list: A new vector representing the sum of v1 and v2.

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension for addition.")
    return [v1[i] + v2[i] for i in range(len(v1))]

def vector_sub(v1, v2):
    """Subtracts the second vector from the first vector element-wise.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        list: A new vector representing the difference v1 - v2.

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension for subtraction.")
    return [v1[i] - v2[i] for i in range(len(v1))]

def scalar_mul(scalar, vec):
    """Multiplies a vector by a scalar.

    Args:
        scalar (float or int): The scalar value.
        vec (list): The vector.

    Returns:
        list: A new vector with each element multiplied by the scalar.
    """
    return [scalar * x for x in vec]

def dot_product(v1, v2):
    """Calculates the dot product of two vectors.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        float: The dot product of v1 and v2.

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension for dot product.")
    return sum(v1[i] * v2[i] for i in range(len(v1)))

# --- Matrix Utility Function ---

def get_matrix_dims(mat):
    """Returns the dimensions (rows, cols) of a matrix.

    Args:
        mat (list of list): The matrix.

    Returns:
        tuple: (rows, cols) or (0, 0) if the matrix is empty.
    """
    if not mat:
        return 0, 0
    if not isinstance(mat[0], list):
        raise TypeError("Input is not a matrix (list of lists).")
    return len(mat), len(mat[0])

# --- Matrix Operations ---

def matrix_add(m1, m2):
    """Adds two matrices element-wise.

    Args:
        m1 (list of list): The first matrix.
        m2 (list of list): The second matrix.

    Returns:
        list of list: A new matrix representing the sum of m1 and m2.

    Raises:
        ValueError: If matrices have different dimensions.
    """
    rows1, cols1 = get_matrix_dims(m1)
    rows2, cols2 = get_matrix_dims(m2)

    if rows1 != rows2 or cols1 != cols2:
        raise ValueError("Matrices must have the same dimensions for addition.")

    result = [[0 for _ in range(cols1)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols1):
            result[i][j] = m1[i][j] + m2[i][j]
    return result

def matrix_sub(m1, m2):
    """Subtracts the second matrix from the first matrix element-wise.

    Args:
        m1 (list of list): The first matrix.
        m2 (list of list): The second matrix.

    Returns:
        list of list: A new matrix representing the difference m1 - m2.

    Raises:
        ValueError: If matrices have different dimensions.
    """
    rows1, cols1 = get_matrix_dims(m1)
    rows2, cols2 = get_matrix_dims(m2)

    if rows1 != rows2 or cols1 != cols2:
        raise ValueError("Matrices must have the same dimensions for subtraction.")

    result = [[0 for _ in range(cols1)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols1):
            result[i][j] = m1[i][j] - m2[i][j]
    return result

def matrix_scalar_mul(scalar, mat):
    """Multiplies a matrix by a scalar.

    Args:
        scalar (float or int): The scalar value.
        mat (list of list): The matrix.

    Returns:
        list of list: A new matrix with each element multiplied by the scalar.
    """
    rows, cols = get_matrix_dims(mat)
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = scalar * mat[i][j]
    return result

def matrix_transpose(mat):
    """Transposes a matrix.

    Args:
        mat (list of list): The matrix to transpose.

    Returns:
        list of list: The transposed matrix.
    """
    rows, cols = get_matrix_dims(mat)
    if rows == 0 and cols == 0:
        return []
    result = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = mat[i][j]
    return result

def matrix_mul(m1, m2):
    """Multiplies two matrices.

    Args:
        m1 (list of list): The first matrix (rows1 x cols1).
        m2 (list of list): The second matrix (rows2 x cols2).

    Returns:
        list of list: The product matrix (rows1 x cols2).

    Raises:
        ValueError: If inner dimensions do not match (cols1 != rows2).
    """
    rows1, cols1 = get_matrix_dims(m1)
    rows2, cols2 = get_matrix_dims(m2)

    if cols1 != rows2:
        raise ValueError(f"Matrix dimensions incompatible for multiplication: ({rows1}x{cols1}) @ ({rows2}x{cols2})")

    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):  # Summation over the inner dimension
                result[i][j] += m1[i][k] * m2[k][j]
    return result

# --- Element-wise Operations (for vectors or matrices) ---

def elementwise_mul(a, b):
    """Performs element-wise multiplication on two lists/vectors or matrices.

    Args:
        a (list or list of list): The first operand.
        b (list or list of list): The second operand.

    Returns:
        list or list of list: The result of element-wise multiplication.

    Raises:
        TypeError: If inputs are not lists or lists of lists.
        ValueError: If dimensions do not match.
    """
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeError("Inputs must be lists or lists of lists (matrices).")

    if not a or not b:
        if len(a) != len(b):
            raise ValueError("Empty lists must have matching emptiness.")
        return []

    is_matrix_a = isinstance(a[0], list)
    is_matrix_b = isinstance(b[0], list)

    if is_matrix_a != is_matrix_b:
        raise TypeError("Both inputs must be either vectors or matrices, not a mix.")

    if is_matrix_a:  # Both are matrices
        rows_a, cols_a = get_matrix_dims(a)
        rows_b, cols_b = get_matrix_dims(b)
        if rows_a != rows_b or cols_a != cols_b:
            raise ValueError("Matrices must have the same dimensions for element-wise multiplication.")
        result = [[0 for _ in range(cols_a)] for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_a):
                result[i][j] = a[i][j] * b[i][j]
        return result
    else:  # Both are vectors
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension for element-wise multiplication.")
        return [a[i] * b[i] for i in range(len(a))]

def elementwise_div(a, b):
    """Performs element-wise division on two lists/vectors or matrices.

    Args:
        a (list or list of list): The dividend operand.
        b (list or list of list): The divisor operand.

    Returns:
        list or list of list: The result of element-wise division.

    Raises:
        TypeError: If inputs are not lists or lists of lists.
        ValueError: If dimensions do not match.
        ZeroDivisionError: If any element in `b` is zero.
    """
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeError("Inputs must be lists or lists of lists (matrices).")

    if not a or not b:
        if len(a) != len(b):
            raise ValueError("Empty lists must have matching emptiness.")
        return []

    is_matrix_a = isinstance(a[0], list)
    is_matrix_b = isinstance(b[0], list)

    if is_matrix_a != is_matrix_b:
        raise TypeError("Both inputs must be either vectors or matrices, not a mix.")

    if is_matrix_a:  # Both are matrices
        rows_a, cols_a = get_matrix_dims(a)
        rows_b, cols_b = get_matrix_dims(b)
        if rows_a != rows_b or cols_a != cols_b:
            raise ValueError("Matrices must have the same dimensions for element-wise division.")
        result = [[0 for _ in range(cols_a)] for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_a):
                if b[i][j] == 0:
                    raise ZeroDivisionError("Cannot divide by zero.")
                result[i][j] = a[i][j] / b[i][j]
        return result
    else:  # Both are vectors
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension for element-wise division.")
        result = []
        for i in range(len(a)):
            if b[i] == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            result.append(a[i] / b[i])
        return result

# --- Activation Functions ---

def _apply_scalar_func(func, x):
    """Helper to apply a scalar function to scalar, vector, or matrix."""
    if isinstance(x, (int, float)):
        return func(x)
    elif isinstance(x, list):
        if not x: return []
        if isinstance(x[0], list):  # Matrix
            rows, cols = get_matrix_dims(x)
            result = [[0 for _ in range(cols)] for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    result[i][j] = func(x[i][j])
            return result
        else:  # Vector
            return [func(val) for val in x]
    else:
        raise TypeError("Input must be a scalar, list, or list of lists.")

def sigmoid(x):
    """Sigmoid activation function. Can handle scalar, vector, or matrix."""
    return _apply_scalar_func(lambda val: 1 / (1 + math.exp(-val)), x)

def sigmoid_derivative(x):
    """Derivative of the sigmoid function. Can handle scalar, vector, or matrix."""
    s = sigmoid(x)
    return _apply_scalar_func(lambda val: val * (1 - val), s)

def relu(x):
    """ReLU activation function. Can handle scalar, vector, or matrix."""
    return _apply_scalar_func(lambda val: max(0, val), x)

def relu_derivative(x):
    """Derivative of the ReLU function. Can handle scalar, vector, or matrix."""
    return _apply_scalar_func(lambda val: 1 if val > 0 else 0, x)

def tanh(x):
    """Tanh activation function. Can handle scalar, vector, or matrix."""
    return _apply_scalar_func(math.tanh, x)

def tanh_derivative(x):
    """Derivative of the Tanh function. Can handle scalar, vector, or matrix."""
    t = tanh(x)
    return _apply_scalar_func(lambda val: 1 - val**2, t)

def softmax(vec):
    """Softmax activation function for a vector.

    Args:
        vec (list): A list of numbers (vector).

    Returns:
        list: A list of probabilities summing to 1.

    Raises:
        TypeError: If input is not a list of numbers.
    """
    if not isinstance(vec, list) or not all(isinstance(x, (int, float)) for x in vec):
        raise TypeError("Input for softmax must be a list of numbers (vector).")
    
    if not vec:
        return []

    # Shift values for numerical stability (optional, but good practice)
    # max_val = max(vec) if vec else 0
    # exp_values = [math.exp(val - max_val) for val in vec]
    
    exp_values = [math.exp(val) for val in vec]
    sum_exp_values = sum(exp_values)
    
    if sum_exp_values == 0: # Should ideally not happen with math.exp unless vec contains -inf
        return [0.0] * len(vec)

    return [val / sum_exp_values for val in exp_values]

# --- Loss Functions ---

def mean_squared_error(y_true, y_pred):
    """Calculates Mean Squared Error (MSE).

    Args:
        y_true (list): The true target values.
        y_pred (list): The predicted values.

    Returns:
        float: The mean squared error.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        return 0.0 # Or raise an error for empty input
    
    error = 0
    for i in range(len(y_true)):
        error += (y_true[i] - y_pred[i])**2
    return error / len(y_true)

def mse_derivative(y_true, y_pred):
    """Calculates the derivative of Mean Squared Error with respect to y_pred.

    Args:
        y_true (list): The true target values.
        y_pred (list): The predicted values.

    Returns:
        list: The derivative of MSE for each prediction.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        return []
    
    n = len(y_true)
    return [2 * (y_pred[i] - y_true[i]) / n for i in range(n)]

def cross_entropy_loss(y_true, y_pred):
    """Calculates Cross-Entropy Loss for multi-class classification.

    Args:
        y_true (list): The true one-hot encoded labels.
        y_pred (list): The predicted probabilities (output of softmax).

    Returns:
        float: The cross-entropy loss.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        return 0.0

    loss = 0
    # Small epsilon for numerical stability to prevent log(0)
    epsilon = 1e-10 
    for i in range(len(y_true)):
        # Clamp predictions to avoid log(0) or log(1) if predictions are exactly 0 or 1
        pred_val = max(epsilon, min(1 - epsilon, y_pred[i])) 
        loss += -y_true[i] * math.log(pred_val)
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    """Calculates the derivative of Cross-Entropy Loss with respect to y_pred.

    Args:
        y_true (list): The true one-hot encoded labels.
        y_pred (list): The predicted probabilities (output of softmax).

    Returns:
        list: The derivative of cross-entropy loss for each prediction.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        return []

    deriv = []
    # Small epsilon for numerical stability to avoid division by zero
    epsilon = 1e-10 
    for i in range(len(y_true)):
        # Clamp predictions
        pred_val = max(epsilon, min(1 - epsilon, y_pred[i])) 
        deriv.append(-y_true[i] / pred_val)
    return deriv
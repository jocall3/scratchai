import random
import math

class AgentModel:
    """
    Defines a simple feed-forward neural network model for an AI agent's
    decision-making logic, implemented from scratch without external deep
    learning libraries.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes the neural network with specified layer sizes.

        Args:
            input_size (int): The number of features in the input state.
            hidden_sizes (list of int): A list of integers, where each integer
                                        represents the number of neurons in a
                                        hidden layer. Can be an empty list for
                                        a direct input-output model.
            output_size (int): The number of possible actions or output values.
        """
        if not isinstance(hidden_sizes, list):
            raise ValueError("hidden_sizes must be a list of integers.")
        if not all(isinstance(size, int) and size > 0 for size in [input_size] + hidden_sizes + [output_size]):
            raise ValueError("All layer sizes must be positive integers.")

        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer transition
        for i in range(len(self.layer_sizes) - 1):
            rows = self.layer_sizes[i]
            cols = self.layer_sizes[i+1]
            
            # Simple initialization scale inspired by Kaiming/He initialization for ReLU
            # This helps in preventing vanishing/exploding gradients in deeper networks.
            # For a basic scratch implementation, a small random range is also common.
            scale = math.sqrt(2.0 / rows) 

            # Initialize weights with random values
            weight_matrix = []
            for _ in range(rows):
                weight_matrix.append([random.uniform(-scale, scale) for _ in range(cols)])
            self.weights.append(weight_matrix)

            # Initialize biases to zeros
            bias_vector = [0.0 for _ in range(cols)]
            self.biases.append(bias_vector)

    def _relu(self, x):
        """
        Applies the Rectified Linear Unit (ReLU) activation function.
        f(x) = max(0, x)
        Args:
            x (list of float): Input vector.
        Returns:
            list of float: Output vector after applying ReLU.
        """
        return [max(0.0, val) for val in x]

    def _softmax(self, x):
        """
        Applies the Softmax activation function to convert a vector of raw
        scores into a probability distribution.
        
        Args:
            x (list of float): Input vector of scores.
        Returns:
            list of float: Output vector of probabilities.
        """
        if not x:
            return []

        # Subtract max_val for numerical stability to prevent overflow with exp()
        max_val = max(x)
        exp_x = [math.exp(val - max_val) for val in x]
        sum_exp_x = sum(exp_x)

        if sum_exp_x == 0:
            # Handle case where all exp_x are effectively zero (e.g., very large negative inputs)
            # In such a rare case, return a uniform distribution.
            return [1.0 / len(x) for _ in x]
        
        return [val / sum_exp_x for val in exp_x]

    def _linear_transform(self, input_vector, weight_matrix, bias_vector):
        """
        Performs a linear transformation: output = (input_vector @ weight_matrix) + bias_vector.
        This is a vector-matrix multiplication followed by vector addition.

        Args:
            input_vector (list of float): A 1D list representing the input.
            weight_matrix (list of list of float): A 2D list representing the weight matrix.
                                                  Dimensions: (input_size x output_size).
            bias_vector (list of float): A 1D list representing the bias vector.
                                         Dimensions: (output_size).
        Returns:
            list of float: The transformed output vector.
        """
        input_size = len(input_vector)
        matrix_rows = len(weight_matrix)
        matrix_cols = len(weight_matrix[0]) if weight_matrix else 0
        bias_size = len(bias_vector)

        if input_size != matrix_rows:
            raise ValueError(f"Input vector size ({input_size}) must match weight matrix rows ({matrix_rows}).")
        if matrix_cols != bias_size:
            raise ValueError(f"Weight matrix columns ({matrix_cols}) must match bias vector size ({bias_size}).")
        
        output_vector = [0.0] * matrix_cols

        # Perform vector-matrix multiplication
        # output_j = sum(input_i * weight_ij) for all i
        for j in range(matrix_cols): # Iterate through columns of weight_matrix (output neurons)
            for i in range(input_size): # Iterate through rows of weight_matrix (input neurons)
                output_vector[j] += input_vector[i] * weight_matrix[i][j]
            # Add bias after summing all products for the current output neuron
            output_vector[j] += bias_vector[j] 
            
        return output_vector

    def forward(self, input_data):
        """
        Performs a forward pass through the neural network.

        Args:
            input_data (list of float): A 1D list representing the current state
                                        or observation of the agent.
        Returns:
            list of float: A 1D list representing the output of the network,
                           typically probabilities for actions if it's a policy
                           network, or action values.
        """
        if len(input_data) != self.input_size:
            raise ValueError(f"Input data size ({len(input_data)}) does not "
                             f"match model's input size ({self.input_size}).")

        current_layer_output = input_data

        # Process through hidden layers
        # The loop runs for all layers EXCEPT the last one (output layer)
        for i in range(len(self.weights) - 1):
            current_layer_output = self._linear_transform(
                current_layer_output,
                self.weights[i],
                self.biases[i]
            )
            current_layer_output = self._relu(current_layer_output)

        # Process the final output layer
        # Assumes the output layer typically uses softmax for action probabilities
        # or a linear activation for value prediction. Here, we use softmax.
        final_layer_output = self._linear_transform(
            current_layer_output,
            self.weights[-1],  # Get the weights for the last layer
            self.biases[-1]    # Get the biases for the last layer
        )
        
        return self._softmax(final_layer_output)

# Example Usage (optional, for testing purposes)
if __name__ == "__main__":
    print("Initializing AgentModel...")
    
    # Model with 3 input features, two hidden layers (64 and 32 neurons), and 4 output actions
    agent_model_complex = AgentModel(input_size=3, hidden_sizes=[64, 32], output_size=4)
    print(f"Complex Model Input Size: {agent_model_complex.input_size}")
    print(f"Complex Model Output Size: {agent_model_complex.output_size}")
    print(f"Complex Model Layer Sizes: {agent_model_complex.layer_sizes}")
    print(f"Number of weight matrices: {len(agent_model_complex.weights)}")
    print(f"Number of bias vectors: {len(agent_model_complex.biases)}")

    # Example input state
    sample_input = [0.1, 0.5, -0.2]
    print(f"\nSample input: {sample_input}")

    # Perform a forward pass
    output_probabilities_complex = agent_model_complex.forward(sample_input)
    print(f"Complex Model Output Probabilities: {output_probabilities_complex}")
    print(f"Sum of probabilities (should be ~1.0): {sum(output_probabilities_complex)}")

    print("\n----------------------------------")

    # Model with 2 input features, no hidden layers, and 3 output actions
    print("Initializing simpler AgentModel (no hidden layers)...")
    agent_model_simple = AgentModel(input_size=2, hidden_sizes=[], output_size=3)
    print(f"Simple Model Input Size: {agent_model_simple.input_size}")
    print(f"Simple Model Output Size: {agent_model_simple.output_size}")
    print(f"Simple Model Layer Sizes: {agent_model_simple.layer_sizes}")
    print(f"Number of weight matrices: {len(agent_model_simple.weights)}")
    print(f"Number of bias vectors: {len(agent_model_simple.biases)}")

    sample_input_simple = [1.0, 0.0]
    print(f"\nSample input: {sample_input_simple}")

    output_probabilities_simple = agent_model_simple.forward(sample_input_simple)
    print(f"Simple Model Output Probabilities: {output_probabilities_simple}")
    print(f"Sum of probabilities (should be ~1.0): {sum(output_probabilities_simple)}")

    print("\nTesting error handling:")
    try:
        AgentModel(input_size=0, hidden_sizes=[10], output_size=2)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        agent_model_simple.forward([1.0]) # Incorrect input size
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nAgentModel functionality demonstrated.")
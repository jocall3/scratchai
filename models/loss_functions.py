```python
import math

class LossFunction:
    """
    Base class for all loss functions.
    Defines the interface for calculating the loss (forward pass)
    and its gradient with respect to predictions (backward pass).
    """
    def forward(self, predictions, targets):
        """
        Computes the loss value.

        Args:
            predictions (list of lists of floats): Model outputs for a batch.
                                                  Each inner list represents a sample's output vector.
            targets (list of lists of floats): True labels for a batch.
                                               Each inner list represents a sample's target vector.

        Returns:
            float: The scalar loss value averaged over the batch.
        """
        raise NotImplementedError

    def backward(self, predictions, targets):
        """
        Computes the gradient of the loss with respect to predictions.
        This gradient is used for backpropagation.

        Args:
            predictions (list of lists of floats): Model outputs for a batch.
            targets (list of lists of floats): True labels for a batch.

        Returns:
            list of lists of floats: The gradient of the loss with respect to predictions,
                                     with the same shape as `predictions`.
        """
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss function.
    Commonly used for regression tasks.
    L = 1/N * sum((y_i - y_hat_i)^2)
    """
    def forward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return 0.0

        total_squared_error = 0.0
        
        # Validate input shapes
        if not all(isinstance(p, list) and isinstance(t, list) for p, t in zip(predictions, targets)):
            raise TypeError("Predictions and targets must be lists of lists of numbers.")
        
        num_output_dimensions = len(predictions[0])
        if not all(len(p) == num_output_dimensions and len(t) == num_output_dimensions for p, t in zip(predictions, targets)):
            raise ValueError("All prediction and target vectors must have the same dimension.")
            
        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            for j in range(num_output_dimensions):
                error = target_vec[j] - pred_vec[j]
                total_squared_error += error * error
        
        total_elements = batch_size * num_output_dimensions
        return total_squared_error / total_elements if total_elements > 0 else 0.0

    def backward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return []

        gradients = []
        
        num_output_dimensions = len(predictions[0])
        total_elements = batch_size * num_output_dimensions

        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            grad_vec = []
            for j in range(num_output_dimensions):
                # dL/d_pred_ij = 2 * (pred_ij - target_ij) / total_elements
                grad_val = 2.0 * (pred_vec[j] - target_vec[j]) / total_elements
                grad_vec.append(grad_val)
            gradients.append(grad_vec)
        return gradients

class MeanAbsoluteError(LossFunction):
    """
    Mean Absolute Error (MAE) loss function.
    Commonly used for regression tasks, less sensitive to outliers than MSE.
    L = 1/N * sum(|y_i - y_hat_i|)
    """
    def forward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return 0.0

        total_absolute_error = 0.0
        
        # Validate input shapes
        if not all(isinstance(p, list) and isinstance(t, list) for p, t in zip(predictions, targets)):
            raise TypeError("Predictions and targets must be lists of lists of numbers.")
        
        num_output_dimensions = len(predictions[0])
        if not all(len(p) == num_output_dimensions and len(t) == num_output_dimensions for p, t in zip(predictions, targets)):
            raise ValueError("All prediction and target vectors must have the same dimension.")
            
        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            for j in range(num_output_dimensions):
                error = target_vec[j] - pred_vec[j]
                total_absolute_error += abs(error)
        
        total_elements = batch_size * num_output_dimensions
        return total_absolute_error / total_elements if total_elements > 0 else 0.0

    def backward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return []

        gradients = []
        
        num_output_dimensions = len(predictions[0])
        total_elements = batch_size * num_output_dimensions

        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            grad_vec = []
            for j in range(num_output_dimensions):
                diff = pred_vec[j] - target_vec[j]
                if diff > 0:
                    grad_val = 1.0 / total_elements
                elif diff < 0:
                    grad_val = -1.0 / total_elements
                else:
                    # Derivative of |x| at x=0 is undefined.
                    # We typically use 0 in ML frameworks for stability.
                    grad_val = 0.0
                grad_vec.append(grad_val)
            gradients.append(grad_vec)
        return gradients

class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross-Entropy (BCE) loss function.
    Used for binary classification tasks, often with sigmoid activation.
    L = -1/N * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
    Expects predictions to be probabilities (between 0 and 1).
    """
    def forward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return 0.0

        total_loss = 0.0
        epsilon = 1e-10 # For numerical stability, prevent log(0)
        
        # Validate input shapes
        if not all(isinstance(p, list) and isinstance(t, list) for p, t in zip(predictions, targets)):
            raise TypeError("Predictions and targets must be lists of lists of numbers.")
        
        num_output_dimensions = len(predictions[0])
        if not all(len(p) == num_output_dimensions and len(t) == num_output_dimensions for p, t in zip(predictions, targets)):
            raise ValueError("All prediction and target vectors must have the same dimension.")
            
        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            for j in range(num_output_dimensions):
                p = max(epsilon, min(1.0 - epsilon, pred_vec[j])) # Clip predictions to (epsilon, 1-epsilon)
                y = target_vec[j]
                
                loss_val = - (y * math.log(p) + (1 - y) * math.log(1 - p))
                total_loss += loss_val
        
        total_elements = batch_size * num_output_dimensions
        return total_loss / total_elements if total_elements > 0 else 0.0

    def backward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return []

        gradients = []
        epsilon = 1e-10
        
        num_output_dimensions = len(predictions[0])
        total_elements = batch_size * num_output_dimensions

        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            grad_vec = []
            for j in range(num_output_dimensions):
                p = max(epsilon, min(1.0 - epsilon, pred_vec[j])) # Clip predictions
                y = target_vec[j]
                
                # dL/dp_ij = (p_ij - y_ij) / (p_ij * (1 - p_ij)) / total_elements
                grad_val = (p - y) / (p * (1 - p)) / total_elements
                grad_vec.append(grad_val)
            gradients.append(grad_vec)
        return gradients

class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross-Entropy (CCE) loss function.
    Used for multi-class classification, typically with softmax activation.
    L = -1/N * sum(sum(y_ic * log(p_ic)))
    Expects predictions to be probabilities (sum to 1 for each sample)
    and targets to be one-hot encoded.
    """
    def forward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return 0.0

        total_loss = 0.0
        epsilon = 1e-10 # For numerical stability, prevent log(0)
        
        # Validate input shapes
        if not all(isinstance(p, list) and isinstance(t, list) for p, t in zip(predictions, targets)):
            raise TypeError("Predictions and targets must be lists of lists of numbers.")
        
        num_output_dimensions = len(predictions[0])
        if not all(len(p) == num_output_dimensions and len(t) == num_output_dimensions for p, t in zip(predictions, targets)):
            raise ValueError("All prediction and target vectors must have the same dimension.")
            
        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            sample_loss = 0.0
            found_true_class = False
            for j in range(num_output_dimensions):
                p = max(epsilon, min(1.0 - epsilon, pred_vec[j])) # Clip predictions
                y = target_vec[j]
                
                if y == 1: # Only the true class contributes to the loss
                    sample_loss += -y * math.log(p)
                    found_true_class = True
                elif y != 0:
                    raise ValueError(f"Targets for CategoricalCrossEntropy must be one-hot encoded (0 or 1), but found {y}.")
            
            if not found_true_class and num_output_dimensions > 0:
                raise ValueError("Target vector for sample has no true class (no '1' found in one-hot encoding).")

            total_loss += sample_loss
        
        # Loss is typically averaged over the batch size only
        return total_loss / batch_size if batch_size > 0 else 0.0

    def backward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return []

        gradients = []
        epsilon = 1e-10
        
        num_output_dimensions = len(predictions[0])

        for i in range(batch_size):
            pred_vec = predictions[i]
            target_vec = targets[i]
            
            grad_vec = []
            for j in range(num_output_dimensions):
                p = max(epsilon, min(1.0 - epsilon, pred_vec[j])) # Clip predictions
                y = target_vec[j]
                
                # dL/dp_j = -y_j / p_j
                # The total gradient for CCE needs to consider the constraint sum(p_j)=1.
                # However, if used directly on softmax outputs, the combined gradient is simpler.
                # This backward computes dL/d(softmax_output)
                grad_val = -y / p if y == 1 else 0.0
                grad_vec.append(grad_val)
            
            # Average over batch
            gradients.append([g / batch_size for g in grad_vec])
        return gradients

# Helper functions for SoftmaxCrossEntropyWithLogits for numerical stability
def _softmax(logits_vec):
    """Computes softmax probabilities for a single sample's logits."""
    if not logits_vec:
        return []

    # Subtract max_logit for numerical stability to prevent overflow with large exp()
    max_logit = logits_vec[0]
    for logit in logits_vec:
        if logit > max_logit:
            max_logit = logit
    
    exp_logits = [math.exp(logit - max_logit) for logit in logits_vec]
    sum_exp_logits = sum(exp_logits)
    
    if sum_exp_logits == 0: 
        # Extremely small logits, all exp_logits rounded to 0. 
        # Return a uniform distribution as a fallback.
        return [1.0 / len(logits_vec)] * len(logits_vec) 
        
    return [exp_logit / sum_exp_logits for exp_logit in exp_logits]

def _log_softmax(logits_vec):
    """Computes log-softmax probabilities for a single sample's logits."""
    if not logits_vec:
        return []

    # Use log-sum-exp trick for numerical stability
    max_logit = logits_vec[0]
    for logit in logits_vec:
        if logit > max_logit:
            max_logit = logit
    
    sum_exp = 0.0
    for logit in logits_vec:
        sum_exp += math.exp(logit - max_logit)
    
    # math.log(0) is an error. Handle if sum_exp becomes zero (all exp_logits are extremely small).
    # This implies all probabilities are effectively zero, leading to infinite loss.
    if sum_exp == 0:
        return [-float('inf')] * len(logits_vec) # Each log_prob is -infinity
        
    log_sum_exp = max_logit + math.log(sum_exp)
    
    return [logit - log_sum_exp for logit in logits_vec]

class SoftmaxCrossEntropyWithLogits(LossFunction):
    """
    Combines Softmax activation and Cross-Entropy loss for multi-class classification.
    Takes raw logits (unnormalized scores) as input, providing numerical stability.
    L = -1/N * sum(sum(y_ic * log(softmax(z_ic))))
    The gradient w.r.t. logits z_ic simplifies to (softmax(z_ic) - y_ic).
    Targets must be one-hot encoded.
    """
    def forward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return 0.0

        total_loss = 0.0
        
        # Validate input shapes
        if not all(isinstance(p, list) and isinstance(t, list) for p, t in zip(predictions, targets)):
            raise TypeError("Predictions (logits) and targets must be lists of lists of numbers.")
        
        num_output_dimensions = len(predictions[0])
        if not all(len(p) == num_output_dimensions and len(t) == num_output_dimensions for p, t in zip(predictions, targets)):
            raise ValueError("All logits and target vectors must have the same dimension.")
            
        for i in range(batch_size):
            logits_vec = predictions[i] # These are raw logits
            target_vec = targets[i]     # These are one-hot encoded
            
            log_probs = _log_softmax(logits_vec)
            
            sample_loss = 0.0
            found_true_class = False
            for j in range(num_output_dimensions):
                if target_vec[j] == 1:
                    sample_loss += -log_probs[j]
                    found_true_class = True
                elif target_vec[j] != 0:
                    raise ValueError(f"Targets for SoftmaxCrossEntropyWithLogits must be one-hot encoded (0 or 1), but found {target_vec[j]}.")
            
            if not found_true_class and num_output_dimensions > 0:
                raise ValueError("Target vector for sample has no true class (no '1' found in one-hot encoding).")

            total_loss += sample_loss
        
        # Average over batch
        return total_loss / batch_size if batch_size > 0 else 0.0

    def backward(self, predictions, targets):
        batch_size = len(predictions)
        if batch_size == 0:
            return []

        gradients = []
        
        num_output_dimensions = len(predictions[0])

        for i in range(batch_size):
            logits_vec = predictions[i]
            target_vec = targets[i]
            
            softmax_output = _softmax(logits_vec) # Compute softmax probabilities from logits
            
            grad_vec = []
            for j in range(num_output_dimensions):
                # dL/d_logit_j = (softmax_output_j - target_j)
                # The gradient is averaged over the batch
                grad_val = (softmax_output[j] - target_vec[j]) / batch_size 
                grad_vec.append(grad_val)
            gradients.append(grad_vec)
        return gradients
```
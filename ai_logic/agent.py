class SimpleNeuralNetwork:
    """
    A very basic placeholder for a neural network, built completely from scratch
    without external dependencies like NumPy.
    It simulates having input, output, and very rudimentary 'predict' and 'train' methods.
    Actual neural network logic (weights, biases, activation functions, backpropagation)
    would be implemented in a dedicated 'nn' or 'models' directory.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # For this placeholder, we just acknowledge the existence of weights and biases.
        # In a real scenario, these would be initialized with random values.
        # Representing them as nested lists for conceptual understanding.
        # Hidden layer (conceptual, a simple feedforward might just have input->output)
        self.weights_hidden = [[0.1 for _ in range(input_size)] for _ in range(input_size)] # Dummy connections
        self.biases_hidden = [0.0 for _ in range(input_size)]
        
        # Output layer
        self.weights_output = [[0.1 for _ in range(input_size)] for _ in range(output_size)]
        self.biases_output = [0.0 for _ in range(output_size)]

    def _relu(self, x):
        """Simple Rectified Linear Unit activation function."""
        return max(0, x)

    def _softmax(self, x):
        """
        A highly simplified softmax-like function for output 'probabilities'.
        Without a proper math.exp, this is illustrative only.
        It normalizes positive values to sum to 1.
        """
        if not x:
            return []
        
        # Filter for positive values to simulate 'activation'
        positive_values = [val for val in x if val > 0]
        total = sum(positive_values)
        
        if total == 0:
            # If all are zero or negative, return uniform distribution conceptually
            return [1.0 / len(x)] * len(x)
        
        return [val / total if val > 0 else 0.0 for val in x]

    def predict(self, input_data):
        """
        Performs a conceptual forward pass through the 'network'.
        It generates dummy output scores based on input data and dummy weights.
        """
        # Ensure input data matches expected size (pad or truncate)
        processed_input = (list(input_data) + [0.0] * self.input_size)[:self.input_size]

        # Conceptual hidden layer calculation (skipped for extreme simplicity, directly to output)
        # In a real NN, this would be `dot_product(input, weights_hidden) + biases_hidden -> activation`

        # Simple weighted sum for output (illustrative only)
        output_scores = [0.0] * self.output_size
        for i in range(self.output_size):
            layer_sum = 0.0
            for j in range(self.input_size):
                layer_sum += processed_input[j] * self.weights_output[i][j] # Using dummy weights
            output_scores[i] = layer_sum + self.biases_output[i]
            # Apply a simple 'activation'
            output_scores[i] = self._relu(output_scores[i])
        
        # Apply a simple 'softmax' to get conceptual probabilities/normalized scores
        return self._softmax(output_scores)

    def train(self, inputs, targets):
        """
        A placeholder for the training method.
        In a real NN, this would involve backpropagation, loss functions,
        and actual weight updates implemented manually.
        """
        # This function simply exists to show where training would be invoked.
        # No actual weight updates or learning logic is implemented here.
        # The full implementation would be in a dedicated `nn.py` file.
        pass


class Memory:
    """
    A basic replay buffer for storing experiences (observation, action, reward, next_observation, done).
    Implemented as a simple list with a maximum size, acting as a FIFO (First-In, First-Out) queue.
    """
    def __init__(self, max_size=1000):
        self.experiences = []
        self.max_size = max_size

    def add_experience(self, observation, action, reward, next_observation, done):
        """Adds a single experience tuple to the memory."""
        if len(self.experiences) >= self.max_size:
            self.experiences.pop(0) # Remove the oldest experience
        self.experiences.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done
        })

    def sample(self, batch_size):
        """
        Samples a batch of experiences from memory.
        For simplicity, this implementation returns the most recent 'batch_size' experiences.
        A true replay buffer would use random sampling for better generalization.
        """
        if not self.experiences:
            return []
        return self.experiences[-batch_size:] # Return up to batch_size elements from the end

    def clear(self):
        """Clears all experiences from memory."""
        self.experiences = []


class PerceptionModule:
    """
    Translates raw environment state into a structured observation that the agent can understand.
    This is a highly simplified version, suitable for a 'from scratch' project without dependencies.
    """
    def __init__(self):
        pass

    def process(self, raw_environment_state):
        """
        Processes the raw environment state into an agent-understandable observation.
        The output is a list of numerical features, capped at a specific size (e.g., 10)
        to match the expected input of a SimpleNeuralNetwork.
        """
        processed_obs = []
        if isinstance(raw_environment_state, dict):
            for k, v in raw_environment_state.items():
                if isinstance(v, (int, float)):
                    processed_obs.append(v)
                elif isinstance(v, str):
                    # Convert first few characters of a string to their ASCII values
                    processed_obs.extend([ord(c) for c in v if c.isalnum()][:3])
        elif isinstance(raw_environment_state, str):
            # Convert first few alphanumeric characters of a string to their ASCII values
            processed_obs.extend([ord(c) for c in raw_environment_state if c.isalnum()][:10])
        elif isinstance(raw_environment_state, (list, tuple)):
            for item in raw_environment_state:
                if isinstance(item, (int, float)):
                    processed_obs.append(item)
                elif isinstance(item, str):
                    processed_obs.extend([ord(c) for c in item if c.isalnum()][:3])
        elif isinstance(raw_environment_state, (int, float)):
            processed_obs.append(raw_environment_state)
        
        # Cap the observation size to match the default SimpleNeuralNetwork input_size
        return processed_obs[:10]


class ActionModule:
    """
    Translates an agent's abstract action (e.g., 'move_forward') into an
    executable command format for the environment.
    This is a simplified passthrough with a predefined mapping.
    """
    def __init__(self):
        # A mapping of abstract agent actions to specific environment commands.
        self.action_mapping = {
            "move_forward": "ENV_CMD_FORWARD",
            "turn_left": "ENV_CMD_ROTATE_LEFT",
            "turn_right": "ENV_CMD_ROTATE_RIGHT",
            "interact": "ENV_CMD_ACTIVATE",
            "wait": "ENV_CMD_STANDBY",
            "no_op": "ENV_CMD_NO_OPERATION" # Default/safety action
        }

    def translate(self, agent_action):
        """
        Translates the agent's chosen abstract action into an environment-specific command.
        Returns "ENV_CMD_NO_OPERATION" if the action is not recognized.
        """
        return self.action_mapping.get(agent_action, "ENV_CMD_NO_OPERATION")


class Goal:
    """
    Represents an agent's objective.
    A goal has a description, criteria for achievement, and a reward function.
    """
    def __init__(self, description, target_state_criteria=None, reward_function=None):
        self.description = description
        # target_state_criteria can be a dict, string, or a callable function
        self.target_state_criteria = target_state_criteria
        # A function: (current_observation, action, next_observation, current_raw_env_state) -> reward
        self.reward_function = reward_function

    def is_achieved(self, current_raw_environment_state):
        """
        Checks if the goal is achieved given the current raw environment state.
        Supports different types of criteria for simplicity.
        """
        if callable(self.target_state_criteria):
            return self.target_state_criteria(current_raw_environment_state)
        elif isinstance(self.target_state_criteria, dict):
            # Check if all key-value pairs in criteria dict are present in raw_environment_state
            if isinstance(current_raw_environment_state, dict):
                return all(item in current_raw_environment_state.items() for item in self.target_state_criteria.items())
            return False
        elif isinstance(self.target_state_criteria, str):
            # Check if the target string is a substring of the raw_environment_state string
            if isinstance(current_raw_environment_state, str):
                return self.target_state_criteria in current_raw_environment_state
            return False
        return False # No specific criteria or unsupported criteria type

    def get_reward(self, current_observation, action, next_observation, current_raw_environment_state):
        """
        Calculates the immediate reward for the agent based on progress towards this goal.
        """
        if self.reward_function:
            return self.reward_function(current_observation, action, next_observation, current_raw_environment_state)
        
        # Default simple reward: 1 if goal is achieved, 0 otherwise
        return 1 if self.is_achieved(current_raw_environment_state) else 0


class Agent:
    """
    Encapsulates the high-level logic for an AI agent's goals, actions,
    and interaction with its environment through perception and action modules.
    It includes a decision model (e.g., a simple NN) and a memory for learning.
    """
    def __init__(self, name="SimpleAgent", goals=None, perception_module=None, action_module=None,
                 decision_model=None, memory=None):
        self.name = name
        self.goals = list(goals) if goals is not None else []
        self.current_goal = None
        if self.goals:
            self.current_goal = self.goals[0] # Agent starts working on the first goal

        self.perception_module = perception_module if perception_module else PerceptionModule()
        self.action_module = action_module if action_module else ActionModule()
        self.memory = memory if memory else Memory()

        # The decision model maps observations to actions.
        self.decision_model = decision_model
        if self.decision_model is None:
            # Default to a very simple NN placeholder.
            # Input size (10) should match the PerceptionModule's output.
            # Output size (6) corresponds to the number of generic actions defined in ActionModule.
            self.decision_model = SimpleNeuralNetwork(input_size=10, output_size=len(self.action_module.action_mapping))

        # Agent's internal model of the world and current context.
        self.internal_state = {} 
        # List of abstract actions the agent can choose from, corresponding to ActionModule.
        self.possible_actions = list(self.action_module.action_mapping.keys())

    def add_goal(self, goal):
        """Adds a new goal to the agent's list of objectives."""
        self.goals.append(goal)
        if not self.current_goal: # If no current goal, set this as the first one
            self.current_goal = goal

    def perceive(self, raw_environment_state):
        """
        Receives raw environment state and processes it into an internal observation.
        Stores the raw state for potential future use (e.g., goal evaluation).
        """
        observation = self.perception_module.process(raw_environment_state)
        self.internal_state['last_observation'] = observation
        self.internal_state['last_raw_environment_state'] = raw_environment_state
        return observation

    def decide_action(self, observation):
        """
        Decides the next abstract action based on the current observation and internal state/goals.
        Uses the decision model (e.g., SimpleNeuralNetwork) to select an action.
        """
        if not observation:
            return "no_op" # Default action if no observation

        action_scores = self.decision_model.predict(observation)

        # Select the action with the highest score.
        if action_scores:
            max_score = -float('inf') 
            chosen_action_index = 0
            for i, score in enumerate(action_scores):
                if score > max_score:
                    max_score = score
                    chosen_action_index = i
            
            # Map the index to a specific abstract action.
            if 0 <= chosen_action_index < len(self.possible_actions):
                chosen_action = self.possible_actions[chosen_action_index]
            else:
                chosen_action = "no_op" # Fallback if index is out of bounds
        else:
            chosen_action = "no_op" # Fallback if no scores from decision model

        return chosen_action

    def execute_action(self, agent_action):
        """
        Translates the agent's chosen abstract action into an environment command
        and returns it. The actual execution by the environment happens externally.
        """
        environment_command = self.action_module.translate(agent_action)
        self.internal_state['last_agent_action'] = agent_action # Store agent's abstract action
        self.internal_state['last_environment_command'] = environment_command
        return environment_command

    def learn_from_experience(self, previous_observation, action_taken, reward, next_observation, done):
        """
        Adds the experience to memory and conceptually triggers the agent's decision model to learn.
        """
        self.memory.add_experience(previous_observation, action_taken, reward, next_observation, done)

        # Conceptual training step for the decision model.
        # In a real setup, this would sample from memory and perform backpropagation
        # using a reinforcement learning algorithm (e.g., Q-learning, policy gradient).
        if isinstance(self.decision_model, SimpleNeuralNetwork):
            # A very simplified conceptual 'training' for the placeholder NN.
            # Involves feeding the previous observation and a simplified target
            # based on the action taken, without complex RL logic.
            if len(self.memory.experiences) > 10: # Only 'train' after collecting some experiences
                inputs = [previous_observation]
                target_output = [0.0] * self.decision_model.output_size
                if action_taken in self.possible_actions:
                    action_index = self.possible_actions.index(action_taken)
                    target_output[action_index] = 1.0 # Simple one-hot-like target for the chosen action
                targets = [target_output]
                self.decision_model.train(inputs, targets)

    def run_cycle(self, raw_environment_state):
        """
        Executes a single perceive-decide-act cycle for the agent.
        It returns the environment command to be executed by the external environment.
        Learning (via `receive_feedback`) happens *after* the environment executes the command.
        """
        current_observation = self.perceive(raw_environment_state)
        chosen_agent_action = self.decide_action(current_observation)
        environment_command = self.execute_action(chosen_agent_action)

        # Store necessary information so `receive_feedback` can associate the outcome
        # with the action that led to it.
        self.internal_state['pending_prev_obs'] = current_observation
        self.internal_state['pending_action_taken'] = chosen_agent_action

        # print(f"{self.name} perceived (raw): {raw_environment_state}, decided: {chosen_agent_action}, command: {environment_command}")
        return environment_command

    def receive_feedback(self, new_raw_environment_state, reward, done):
        """
        Receives feedback from the environment after an action was executed.
        This method is called *after* the `run_cycle`'s command has been executed
        by the environment. It updates the agent's internal state and triggers learning.
        """
        # Retrieve information about the action that was taken in the previous cycle.
        previous_observation = self.internal_state.get('pending_prev_obs')
        action_taken = self.internal_state.get('pending_action_taken')

        if previous_observation is None or action_taken is None:
            print(f"Warning: {self.name} received feedback without a pending action. No learning performed.")
            # Still update internal state with new observation if possible
            _ = self.perceive(new_raw_environment_state)
            return

        next_observation = self.perception_module.process(new_raw_environment_state)

        # Perform learning based on the experience.
        self.learn_from_experience(previous_observation, action_taken, reward, next_observation, done)

        # Update current internal state with the latest feedback.
        self.internal_state['last_reward'] = reward
        self.internal_state['is_done'] = done
        self.internal_state['last_observation'] = next_observation
        self.internal_state['last_raw_environment_state'] = new_raw_environment_state

        # Evaluate progress towards the current goal.
        if self.current_goal:
            goal_reward = self.current_goal.get_reward(previous_observation, action_taken, next_observation, new_raw_environment_state)
            # If the goal is achieved based on the new environment state.
            if self.current_goal.is_achieved(new_raw_environment_state):
                 print(f"{self.name} achieved goal: '{self.current_goal.description}'")
                 # Move to the next goal if available.
                 if len(self.goals) > 1:
                     self.goals.pop(0) # Remove the achieved goal from the list
                     self.current_goal = self.goals[0] # Set the next goal
                     print(f"{self.name} is now pursuing goal: '{self.current_goal.description}'")
                 else:
                     self.current_goal = None
                     print(f"{self.name} has completed all its goals.")
        
        # Clear pending action info after feedback is fully processed to avoid stale data.
        self.internal_state.pop('pending_prev_obs', None)
        self.internal_state.pop('pending_action_taken', None)
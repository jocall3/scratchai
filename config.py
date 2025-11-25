# --- Application Server Configuration ---
API_HOST = "127.0.0.1"
API_PORT = 8000

# --- Model Configuration ---
# General Model Parameters
MODEL_NAME = "FromScratchAI"
MODEL_TYPE = "transformer"  # Options: 'transformer', 'cnn', 'rnn', 'mlp', etc.
DEVICE = "cpu"              # Target device for computations. 'gpu' could be added if custom GPU kernels are implemented.

# Transformer-specific Parameters (example values for a small model)
VOCAB_SIZE = 10000          # Size of the vocabulary (number of unique tokens)
EMBEDDING_DIM = 256         # Dimension of token embeddings
NUM_HEADS = 8               # Number of attention heads in MultiHeadAttention
NUM_LAYERS = 3              # Number of encoder/decoder layers
FFN_DIM = 1024              # Dimension of the feed-forward network in Transformer blocks
MAX_SEQ_LEN = 256           # Maximum sequence length for inputs/outputs
DROPOUT_RATE = 0.1          # Dropout rate for regularization

# CNN-specific Parameters (if MODEL_TYPE is 'cnn')
# CONV_FILTERS = 128
# KERNEL_SIZE = 3
# NUM_CONV_LAYERS = 3

# Training/Optimization Parameters (even if training is not the immediate focus, these are model-related)
LEARNING_RATE = 0.001       # Initial learning rate for optimization

# --- Data and Paths Configuration ---
DATA_DIR = "data/"          # Directory for datasets and processed data
MODEL_SAVE_PATH = "models/" # Directory to save/load model weights
VOCAB_PATH = f"{DATA_DIR}vocab.txt"          # Path to the vocabulary file
TOKENIZER_SAVE_PATH = f"{DATA_DIR}tokenizer.json" # Path to save/load custom tokenizer configuration
PRETRAINED_WEIGHTS_PATH = f"{MODEL_SAVE_PATH}{MODEL_NAME}_weights.json" # Example path for model weights

# --- Chatbot/Agent Configuration ---
BOT_NAME = "ScratchBot"
WELCOME_MESSAGE = "Hello! I am a simple AI built completely from scratch. How can I assist you today?"
DEFAULT_AGENT_PROMPT = "You are a helpful AI assistant. Provide concise and relevant answers."

# --- UI Configuration ---
UI_TITLE = "My From-Scratch AI Application"
UI_DEFAULT_WIDTH = 800
UI_DEFAULT_HEIGHT = 600

# --- Logging Configuration (basic, no external logging library) ---
LOG_FILE = "app.log"
LOG_LEVEL = "INFO" # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
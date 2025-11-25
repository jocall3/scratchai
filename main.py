import http.server
import socketserver
import os
import json
import urllib.parse
import threading

# Import our custom components from the src directory
# These imports assume a project structure where 'src' is a sibling of 'main.py'
from src.backend.api import handle_api_request
from src.agents.chatbot import Chatbot
from src.models.nn import NeuralNetwork
from src.models.transformer import TransformerModel
# Future models (CNN, GNN, etc.) would be imported here as they are developed:
# from src.models.cnn import CNNModel
# from src.models.gnn import GNNModel

# --- Configuration ---
PORT = 8000
# Determine the path to the frontend public directory relative to main.py
FRONTEND_PUBLIC_DIR = os.path.join(os.path.dirname(__file__), 'src', 'frontend', 'public')

# Ensure the frontend public directory exists (for serving static files)
if not os.path.isdir(FRONTEND_PUBLIC_DIR):
    print(f"Warning: Frontend public directory not found at '{FRONTEND_PUBLIC_DIR}'.")
    print("Serving will likely fail for frontend assets.")

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler to serve static files and API endpoints.
    """
    def __init__(self, *args, models=None, chatbot=None, **kwargs):
        self.models = models
        self.chatbot = chatbot
        # Call the parent constructor, setting the directory for serving static files
        super().__init__(*args, directory=FRONTEND_PUBLIC_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith('/api/'):
            self._handle_api_request()
        else:
            # If path is '/', default to serving index.html
            if self.path == '/':
                self.path = '/index.html'
            # Serve static files using the default SimpleHTTPRequestHandler logic
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/'):
            self._handle_api_request()
        else:
            self._send_error_response(405, "Method Not Allowed for static content.")

    def _handle_api_request(self):
        """Dispatches API requests to the appropriate handler in src.backend.api."""
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            
            post_data = None
            if self.command == 'POST':
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    raw_post_data = self.rfile.read(content_length).decode('utf-8')
                    post_data = json.loads(raw_post_data)
            
            # Call the central API handler from src.backend.api to process the request
            response_data, status_code = handle_api_request(
                method=self.command,
                path=parsed_path.path,
                query_params=urllib.parse.parse_qs(parsed_path.query),
                body=post_data,
                models=self.models,
                chatbot=self.chatbot
            )

            self._send_json_response(status_code, response_data)

        except json.JSONDecodeError:
            self._send_error_response(400, "Invalid JSON in request body.")
        except Exception as e:
            print(f"Error handling API request: {e}")
            self._send_error_response(500, f"Internal Server Error: {str(e)}")

    def _send_json_response(self, status_code, data):
        """Sends a JSON response to the client."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error_response(self, status_code, message):
        """Sends an error response with a JSON error message."""
        self._send_json_response(status_code, {"error": message})

def run_server(models, chatbot, server_class=socketserver.TCPServer, handler_class=CustomHTTPRequestHandler):
    """
    Sets up and runs the HTTP server.
    """
    # Create a partial class for the handler to pass models and chatbot instances
    handler_with_args = lambda *args, **kwargs: handler_class(*args, models=models, chatbot=chatbot, **kwargs)

    server_address = ('', PORT)
    with server_class(server_address, handler_with_args) as httpd:
        print(f"Serving AI app on http://localhost:{PORT}")
        print(f"Frontend static files from: {FRONTEND_PUBLIC_DIR}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.shutdown()
            httpd.server_close()

def load_ai_components():
    """
    Loads and initializes all AI models and agents.
    """
    print("Loading AI components...")
    
    # Instantiate models
    # These are placeholder instantiations. Actual model logic resides in src/models.
    nn_model = NeuralNetwork(input_size=10, hidden_size=20, output_size=2)
    transformer_model = TransformerModel(vocab_size=1000, max_seq_len=50, num_layers=2, num_heads=4, d_model=64)
    # Instantiate other models here as they are developed (e.g., CNNModel, GNNModel)
    
    loaded_models = {
        "nn": nn_model,
        "transformer": transformer_model,
        # "cnn": CNNModel(), # Placeholder for future
        # "gnn": GNNModel(), # Placeholder for future
    }
    print("Models loaded successfully.")

    # Initialize chatbot agent
    # The chatbot might depend on one or more of the loaded models.
    chatbot_agent = Chatbot(model=transformer_model) # Example: chatbot uses the transformer model
    print("Chatbot initialized successfully.")

    return loaded_models, chatbot_agent

def main():
    """
    Main entry point for the application.
    Orchestrates AI component loading and server startup.
    """
    # 1. Load AI components (models, agents)
    models, chatbot = load_ai_components()

    # 2. Start the HTTP server to serve frontend and handle API requests
    run_server(models, chatbot)

if __name__ == "__main__":
    main()
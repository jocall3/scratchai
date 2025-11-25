import json
import time # Used for simulating processing delays and timestamps

# --- Placeholder for custom AI models ---
# In a full "from scratch" implementation, these would be actual classes
# defined in other files (e.g., backend/models/chatbot_agent.py, backend/models/prediction_model.py).
# For the purpose of making THIS file self-contained and runnable without
# external file dependencies, we define simple mock versions here.

class MockChatbotAgent:
    """A mock chatbot agent simulating AI interaction."""
    def __init__(self):
        self.responses = {
            "hello": "Hi there! How can I assist you today?",
            "how are you": "I'm a computer program, so I don't have feelings, but I'm functioning perfectly!",
            "what is your purpose": "My purpose is to demonstrate a conversational AI API built from scratch.",
            "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
            "bye": "Goodbye! Have a great day!"
        }
        self.default_response = "I'm still learning and don't quite understand that yet. Can you try rephrasing?"

    def generate_response(self, message: str) -> str:
        """Generates a response based on the input message."""
        time.sleep(0.5)  # Simulate AI processing time
        message_lower = message.lower().strip()
        for keyword, response in self.responses.items():
            if keyword in message_lower:
                return response
        return self.default_response

class MockPredictionModel:
    """A mock prediction model simulating AI prediction."""
    def __init__(self):
        pass

    def predict(self, data: dict) -> dict:
        """Makes a simple prediction based on input data."""
        time.sleep(0.3)  # Simulate AI processing time
        if "input_value" in data and isinstance(data["input_value"], (int, float)):
            # A very simple linear model for demonstration
            prediction = data["input_value"] * 1.5 + 10
            confidence = 0.9 if data["input_value"] < 100 else 0.7 # Simulate varying confidence
            return {"predicted_value": prediction, "confidence": confidence}
        return {"error": "Invalid input: 'input_value' (numeric) is required.", "confidence": 0.0}

# --- Initialize our "from scratch" models ---
# These instances will be used by our API handlers.
CHATBOT_AGENT = MockChatbotAgent()
PREDICTION_MODEL = MockPredictionModel()

# --- Helper functions for API request/response processing ---

def _parse_json_body(body_bytes: bytes) -> dict:
    """
    Parses a JSON body from bytes.
    Returns a dictionary or a dictionary with an "error" key if parsing fails.
    """
    try:
        return json.loads(body_bytes.decode('utf-8'))
    except json.JSONDecodeError:
        return {"_parse_error": "Invalid JSON format."}
    except UnicodeDecodeError:
        return {"_parse_error": "Invalid UTF-8 encoding."}
    except Exception as e:
        return {"_parse_error": f"An unexpected error occurred during JSON parsing: {e}"}

def _json_response(status_code: int, data: dict, headers: dict = None) -> tuple[int, dict, bytes]:
    """
    Helper to create a standard JSON response tuple.
    Returns (status_code, response_headers, response_body_bytes).
    """
    response_headers = {"Content-Type": "application/json"}
    if headers:
        response_headers.update(headers)
    body_bytes = json.dumps(data).encode('utf-8')
    return status_code, response_headers, body_bytes

# --- API Handler Functions ---

def handle_chat(request_path: str, request_method: str, request_headers: dict, request_body_bytes: bytes) -> tuple[int, dict, bytes]:
    """
    Handles requests to the /chat endpoint.
    Expects a POST request with a JSON body containing a 'message' key.
    """
    if request_method != 'POST':
        return _json_response(405, {"error": "Method Not Allowed", "allowed_methods": ["POST"]})

    body = _parse_json_body(request_body_bytes)
    if "_parse_error" in body:
        return _json_response(400, {"error": f"Bad Request: {body['_parse_error']}"})

    message = body.get('message')
    if not isinstance(message, str) or not message.strip():
        return _json_response(400, {"error": "Bad Request: 'message' field is required and must be a non-empty string."})

    response_text = CHATBOT_AGENT.generate_response(message)
    return _json_response(200, {"reply": response_text})

def handle_predict(request_path: str, request_method: str, request_headers: dict, request_body_bytes: bytes) -> tuple[int, dict, bytes]:
    """
    Handles requests to the /predict endpoint.
    Expects a POST request with a JSON body containing input data for prediction.
    """
    if request_method != 'POST':
        return _json_response(405, {"error": "Method Not Allowed", "allowed_methods": ["POST"]})

    body = _parse_json_body(request_body_bytes)
    if "_parse_error" in body:
        return _json_response(400, {"error": f"Bad Request: {body['_parse_error']}"})

    # For a real model, we'd add extensive input validation based on expected features.
    if not isinstance(body, dict):
        return _json_response(400, {"error": "Bad Request: Request body must be a JSON object."})

    prediction_result = PREDICTION_MODEL.predict(body)

    if "error" in prediction_result:
        return _json_response(400, {"error": f"Prediction Error: {prediction_result['error']}"})

    return _json_response(200, {"prediction": prediction_result})

def handle_status(request_path: str, request_method: str, request_headers: dict, request_body_bytes: bytes) -> tuple[int, dict, bytes]:
    """
    Handles requests to the /status endpoint.
    Provides a basic health check of the API and loaded models.
    """
    if request_method != 'GET':
        return _json_response(405, {"error": "Method Not Allowed", "allowed_methods": ["GET"]})

    status_info = {
        "status": "operational",
        "api_version": "0.1.0",
        "timestamp": time.time(),
        "models_status": {
            "chatbot_agent": "ready",
            "prediction_model": "ready"
        },
        "message": "All core AI components are running."
    }
    return _json_response(200, status_info)

def handle_default(request_path: str, request_method: str, request_headers: dict, request_body_bytes: bytes) -> tuple[int, dict, bytes]:
    """
    Default handler for requests that do not match any defined route.
    Returns a 404 Not Found response.
    """
    return _json_response(404, {"error": "Not Found", "path": request_path, "message": "The requested resource was not found on this server."})

# --- API Route Dispatcher ---
# This dictionary maps URL paths to their corresponding handler functions.
# A server implementation (e.g., in a `backend/server.py` file) would use this
# to determine which function to call based on the incoming request path.
API_ROUTES = {
    '/chat': handle_chat,
    '/predict': handle_predict,
    '/status': handle_status,
    # Add other API routes here as the application grows
}

# The actual HTTP server implementation (using http.server or similar)
# would live in a separate file (e.g., main.py or server.py) and would
# import these handlers and the API_ROUTES dictionary to dispatch requests.
# Example usage in a server:
# handler_func = API_ROUTES.get(path, handle_default)
# status, headers, body = handler_func(path, method, req_headers, req_body_bytes)
```
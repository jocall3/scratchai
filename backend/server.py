import http.server
import socketserver
import json
import sys
import os

# --- Configuration ---
HOST = 'localhost'
PORT = 8000
# Define the root directory for static files relative to this script.
# For a basic backend, we might not serve static files, but if we later want
# to serve the frontend directly from the backend, this path would be used.
# For now, this server will focus on API endpoints.
STATIC_FILES_ROOT = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public')

# --- Request Handler ---
class MyRequestHandler(http.server.BaseHTTPRequestHandler):

    def _set_headers(self, status_code=200, content_type='application/json'):
        """Sets common headers for responses, including CORS."""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        # Allow CORS for development (e.g., frontend running on a different port)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        """Handles HTTP OPTIONS requests for CORS preflight."""
        self._set_headers(status_code=200)

    def do_GET(self):
        """Handles HTTP GET requests."""
        if self.path == '/':
            # Serve a basic welcome message for the root path
            self._set_headers(content_type='text/html')
            self.wfile.write(b"<h1>Welcome to the AI App Backend!</h1>")
            self.wfile.write(b"<p>API endpoints:</p>")
            self.wfile.write(b"<ul>")
            self.wfile.write(b"<li><a href=\"/api/status\">/api/status</a> (GET)</li>")
            self.wfile.write(b"<li>/api/chat (POST)</li>")
            self.wfile.write(b"<li>/api/model/predict (POST)</li>")
            self.wfile.write(b"</ul>")
        elif self.path == '/api/status':
            # Basic status check endpoint
            self._set_headers()
            response = {"status": "ok", "message": "Backend is running."}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            # Handle other paths (e.g., static files, if we were to serve them)
            # For now, default to 404 for unknown GET paths.
            self._set_headers(404)
            response = {"error": "Not Found", "path": self.path}
            self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        """Handles HTTP POST requests."""
        if self.path == '/api/chat' or self.path == '/api/model/predict':
            # Expect JSON data for AI endpoints
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._set_headers(400)
                response = {"error": "Request body is empty."}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return

            post_data_raw = self.rfile.read(content_length)

            try:
                data = json.loads(post_data_raw.decode('utf-8'))
                print(f"Received POST data for {self.path}: {data}")

                # --- Placeholder for AI Model / Chatbot Logic ---
                # In future steps, this is where actual AI model inference
                # or chatbot interaction logic would be integrated.
                # For now, we simulate a response.
                if self.path == '/api/chat':
                    user_message = data.get('message', 'No message provided.')
                    model_response = f"Echo from Chatbot: '{user_message}'"
                    # Here you would call your custom chatbot model.
                    # e.g., chatbot_response = my_chatbot.process(user_message)
                    response_data = {
                        "request_path": self.path,
                        "received_message": user_message,
                        "bot_response": model_response,
                        "timestamp": http.server.time.time()
                    }
                elif self.path == '/api/model/predict':
                    input_data = data.get('input', {})
                    prediction = f"Simulated prediction for input: {json.dumps(input_data)}"
                    # Here you would call your custom AI model.
                    # e.g., model_output = my_model.predict(input_data)
                    response_data = {
                        "request_path": self.path,
                        "received_input": input_data,
                        "prediction": prediction,
                        "timestamp": http.server.time.time()
                    }
                else:
                    # Should not reach here due to the `if` condition, but good for safety
                    raise ValueError("Unknown POST path handled incorrectly.")
                # --- End Placeholder ---

                self._set_headers(200)
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except json.JSONDecodeError:
                self._set_headers(400)
                response = {"error": "Invalid JSON format in request body."}
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self._set_headers(500)
                response = {"error": f"Internal server error: {str(e)}"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self._set_headers(404)
            response = {"error": "Not Found", "path": self.path}
            self.wfile.write(json.dumps(response).encode('utf-8'))

# --- Server Setup ---
def run_server():
    """Starts the HTTP server."""
    # TCPServer handles socket creation and management, BaseHTTPRequestHandler handles HTTP protocol.
    # HTTPServer is a convenience class that combines them.
    with socketserver.TCPServer((HOST, PORT), MyRequestHandler) as httpd:
        print(f"Serving AI backend at http://{HOST}:{PORT}")
        print("Press Ctrl+C to shut down.")
        try:
            httpd.serve_forever() # Keep the server running indefinitely
        except KeyboardInterrupt:
            print("\nShutting down the server...")
        except Exception as e:
            print(f"An unexpected server error occurred: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            httpd.shutdown() # Stop the serve_forever loop
            httpd.server_close() # Close the server socket
            print("Server stopped.")

# --- Main execution ---
if __name__ == "__main__":
    run_server()
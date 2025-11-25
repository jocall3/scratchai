```python
import json
import urllib.parse

class Request:
    """
    Represents an incoming HTTP request.
    Populated by the underlying HTTP server and passed to the router.
    """
    def __init__(self, method: str, path: str, headers: dict, body: bytes, query_string: str):
        self.method = method.upper()
        self.path = path
        self.headers = {k.lower(): v for k, v in headers.items()} # Store headers as lowercase
        self.body = body
        self.query_string = query_string
        self.query_params = self._parse_query_params(query_string)
        self.json = None

        content_type = self.headers.get('content-type', '')
        if 'application/json' in content_type and self.body:
            try:
                self.json = json.loads(self.body.decode('utf-8'))
            except json.JSONDecodeError:
                self.json = None # Invalid JSON body

    def _parse_query_params(self, query_string: str) -> dict:
        """Parses URL query string into a dictionary."""
        params = {}
        if not query_string:
            return params
        
        for pair in query_string.split('&'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                params[urllib.parse.unquote_plus(key)] = urllib.parse.unquote_plus(value)
            else:
                params[urllib.parse.unquote_plus(pair)] = ''
        return params

class Response:
    """
    Represents an HTTP response to be sent back to the client.
    Handlers should return instances of this class.
    """
    def __init__(self, status: int = 200, headers: dict = None, body: bytes = b''):
        self.status = status
        self.headers = headers if headers is not None else {}
        self.body = body

    def to_dict(self) -> dict:
        """Converts the Response object to a dictionary for easy processing by the HTTP server."""
        return {
            "status": self.status,
            "headers": self.headers,
            "body": self.body
        }

    @staticmethod
    def json_response(data: dict, status: int = 200, headers: dict = None) -> 'Response':
        """Helper to create a JSON response."""
        response_headers = {"Content-Type": "application/json"}
        if headers:
            response_headers.update(headers)
        return Response(status=status, headers=response_headers, body=json.dumps(data).encode('utf-8'))

    @staticmethod
    def text_response(text: str, status: int = 200, headers: dict = None) -> 'Response':
        """Helper to create a plain text response."""
        response_headers = {"Content-Type": "text/plain"}
        if headers:
            response_headers.update(headers)
        return Response(status=status, headers=response_headers, body=text.encode('utf-8'))

    @staticmethod
    def not_found() -> 'Response':
        """Helper for a 404 Not Found response."""
        return Response.text_response("404 Not Found", status=404)

    @staticmethod
    def method_not_allowed() -> 'Response':
        """Helper for a 405 Method Not Allowed response."""
        return Response.text_response("405 Method Not Allowed", status=405)

    @staticmethod
    def bad_request(message: str = "400 Bad Request") -> 'Response':
        """Helper for a 400 Bad Request response."""
        return Response.text_response(message, status=400)

    @staticmethod
    def internal_server_error(details: str = "Internal Server Error") -> 'Response':
        """Helper for a 500 Internal Server Error response."""
        return Response.json_response({"error": details}, status=500)


class Router:
    """
    Manages API routes, mapping HTTP methods and paths to handler functions.
    """
    def __init__(self):
        # Stores routes in the format: {method: {path: handler_func}}
        self.routes = {}

    def _add_route(self, method: str, path: str, handler):
        """Internal method to add a route."""
        if method not in self.routes:
            self.routes[method] = {}
        # Basic exact path matching. For more complex routing (e.g., /users/{id}),
        # a more sophisticated matching algorithm would be needed.
        self.routes[method][path] = handler

    def get(self, path: str, handler):
        """Registers a GET route."""
        self._add_route("GET", path, handler)

    def post(self, path: str, handler):
        """Registers a POST route."""
        self._add_route("POST", path, handler)

    def put(self, path: str, handler):
        """Registers a PUT route."""
        self._add_route("PUT", path, handler)

    def delete(self, path: str, handler):
        """Registers a DELETE route."""
        self._add_route("DELETE", path, handler)

    def dispatch(self, request: Request) -> Response:
        """
        Dispatches an incoming request to the appropriate handler function.

        Args:
            request: An instance of the Request object.

        Returns:
            An instance of the Response object.
        """
        method_routes = self.routes.get(request.method)
        if not method_routes:
            return Response.method_not_allowed()

        handler = method_routes.get(request.path)
        if not handler:
            return Response.not_found()

        try:
            # Call the handler with the request object
            response = handler(request)
            if not isinstance(response, Response):
                # Ensure handlers return a Response object
                raise TypeError(f"Handler for {request.method} {request.path} did not return a Response object.")
            return response
        except Exception as e:
            # Catch any exceptions raised by the handler and return a 500 error
            print(f"Error handling request {request.method} {request.path}: {e}")
            return Response.internal_server_error(details=str(e))

```
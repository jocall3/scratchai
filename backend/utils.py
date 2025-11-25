import urllib.parse
import json # Only imported if HTTPRequest.get_json_body() is called

class HTTPRequest:
    """
    Represents a parsed HTTP request.
    """
    def __init__(self, method: str, path: str, query_params: dict, headers: dict, body: bytes):
        self.method = method
        self.path = path
        self.query_params = query_params
        self.headers = headers  # Stored with keys in lowercase for easy lookup
        self.body = body

    def __repr__(self):
        return (f"HTTPRequest(method='{self.method}', path='{self.path}', "
                f"query_params={self.query_params}, headers_keys={list(self.headers.keys())}, "
                f"body_len={len(self.body)})")

    def get_header(self, name: str, default=None):
        """
        Retrieves a header value by name (case-insensitive).
        """
        return self.headers.get(name.lower(), default)

    def get_json_body(self):
        """
        Attempts to parse the request body as JSON.
        Requires 'Content-Type: application/json' header.
        Raises ValueError if not JSON or parsing fails.
        """
        content_type = self.get_header("content-type", "")
        if "application/json" in content_type.lower():
            try:
                return json.loads(self.body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to parse JSON body: {e}")
        raise ValueError("Request body is not application/json or content-type header is missing/incorrect.")


def parse_http_request(raw_request_bytes: bytes) -> HTTPRequest:
    """
    Parses a raw HTTP request byte string into an HTTPRequest object.
    This is a simplified parser and might not handle all edge cases or malformed requests.
    It expects a complete request, or it will parse what's available.
    """
    
    header_end_marker = b'\r\n\r\n'
    header_end_pos = raw_request_bytes.find(header_end_marker)

    if header_end_pos == -1:
        # Request is likely malformed or incomplete (no header/body separator)
        # For simplicity, we treat the entire bytes as headers and an empty body.
        header_bytes = raw_request_bytes
        body_bytes = b''
    else:
        header_bytes = raw_request_bytes[:header_end_pos]
        body_bytes = raw_request_bytes[header_end_pos + len(header_end_marker):]

    try:
        header_str = header_bytes.decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        header_str = header_bytes.decode('latin-1', errors='ignore') # Fallback for headers

    header_lines = header_str.split('\r\n')

    # Filter out empty lines from the header_lines list.
    # We expect at least one line (request line).
    header_lines = [line for line in header_lines if line.strip()]

    if not header_lines:
        raise ValueError("Empty HTTP request after filtering empty lines.")

    # Request Line: METHOD /path?query HTTP/1.1
    request_line = header_lines[0]
    parts = request_line.split(' ')
    if len(parts) != 3:
        raise ValueError(f"Malformed request line: '{request_line}' (expected 3 parts)")

    method = parts[0].upper()
    full_path = parts[1]
    
    path_parts = full_path.split('?', 1)
    path = urllib.parse.unquote(path_parts[0]) # Decode URL path
    
    query_params = {}
    if len(path_parts) > 1:
        query_string = path_parts[1]
        # Use urllib.parse.parse_qs for robust query parameter parsing
        # It returns a dict where values are lists, so convert to single value per key
        parsed_qs = urllib.parse.parse_qs(query_string, keep_blank_values=True)
        query_params = {k: v[0] for k, v in parsed_qs.items()}

    headers = {}
    for line in header_lines[1:]:
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip().lower()] = value.strip() # Store headers as lowercase for easier lookup
    
    # Adjust body_bytes based on Content-Length header for specific methods
    content_length_str = headers.get("content-length")
    if content_length_str and method in ("POST", "PUT", "PATCH"):
        try:
            content_length = int(content_length_str)
            if len(body_bytes) < content_length:
                # This means the request passed to the parser is incomplete.
                # In a real server, the network layer would buffer until content_length bytes are received.
                # For this utility, we'll just use the received body and assume it might be truncated.
                print(f"Warning: Incomplete body received. Expected {content_length} bytes, got {len(body_bytes)}")
            elif len(body_bytes) > content_length:
                 # If more bytes are sent than Content-Length specifies (e.g., pipelining),
                 # we should only consider up to Content-Length.
                body_bytes = body_bytes[:content_length]
        except ValueError:
            print(f"Warning: Invalid Content-Length header: {content_length_str}")

    return HTTPRequest(method, path, query_params, headers, body_bytes)


# --- Response Building ---

HTTP_STATUS_CODES = {
    100: "Continue",
    101: "Switching Protocols",
    200: "OK",
    201: "Created",
    202: "Accepted",
    204: "No Content", # Must not have body or Content-Length header
    301: "Moved Permanently",
    302: "Found",
    303: "See Other",
    304: "Not Modified", # Must not have body or Content-Length header
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    408: "Request Timeout",
    409: "Conflict",
    410: "Gone",
    413: "Payload Too Large",
    414: "URI Too Long",
    415: "Unsupported Media Type",
    429: "Too Many Requests",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

DEFAULT_RESPONSE_HEADERS = {
    "Server": "AIPyServer/0.1-alpha",
    "Connection": "close",
}

def _title_case_header(key: str) -> str:
    """Converts a header key to Title-Case (e.g., "content-type" -> "Content-Type")."""
    return '-'.join(word.capitalize() for word in key.split('-'))

def build_http_response(status_code: int, body=b'', headers: dict = None) -> bytes:
    """
    Builds a raw HTTP response byte string.
    :param status_code: The HTTP status code (e.g., 200, 404).
    :param body: The response body, can be bytes or str. If str, it's encoded to utf-8.
                 If None, no body is sent (useful for 204 No Content, 304 Not Modified).
    :param headers: A dictionary of additional headers to include/override.
                    Keys will be converted to Title-Case in the final response.
    :return: A byte string representing the complete HTTP response.
    """
    status_text = HTTP_STATUS_CODES.get(status_code, "Unknown Status")

    response_line = f"HTTP/1.1 {status_code} {status_text}"

    final_headers = DEFAULT_RESPONSE_HEADERS.copy()
    
    body_bytes = b''
    # Determine body and set default Content-Type if not already specified
    if body is not None:
        if isinstance(body, str):
            body_bytes = body.encode('utf-8')
            if "content-type" not in (h.lower() for h in final_headers):
                final_headers["Content-Type"] = "text/plain; charset=utf-8"
        elif isinstance(body, bytes):
            body_bytes = body
            if "content-type" not in (h.lower() for h in final_headers):
                final_headers["Content-Type"] = "application/octet-stream"
        else:
            # Attempt to convert other types to string and then bytes
            body_bytes = str(body).encode('utf-8')
            if "content-type" not in (h.lower() for h in final_headers):
                final_headers["Content-Type"] = "text/plain; charset=utf-8"

    # Add/override user-provided headers, converting keys to lowercase for internal consistency
    if headers:
        for key, value in headers.items():
            final_headers[key.lower()] = value 

    # Handle status codes that must not have a body or Content-Length
    if status_code in (204, 304):
        body_bytes = b'' # Ensure no body is sent for these status codes
        # Remove Content-Length if it was set by default or provided by user
        final_headers.pop("content-length", None)
    else:
        # For other status codes, set Content-Length based on the actual body
        final_headers["content-length"] = str(len(body_bytes))

    header_lines = []
    # Iterate over headers, format keys to Title-Case for the output
    for key, value in final_headers.items():
        formatted_key = _title_case_header(key)
        header_lines.append(f"{formatted_key}: {value}")
    
    # Construct the full response string and convert to bytes
    # Format: Status-Line\r\nHeader1: Value\r\nHeader2: Value\r\n\r\nBody
    response_str = response_line + "\r\n" + "\r\n".join(header_lines) + "\r\n\r\n"
    response_bytes = response_str.encode('utf-8') + body_bytes

    return response_bytes
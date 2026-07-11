"""Mock Fireworks endpoint for local smoke testing.
Runs a fake OpenAI-compatible server that returns canned responses,
so we can verify the container works end-to-end without any real API credits."""
import http.server
import json
import threading

MOCK_PORT = 8899

class MockFireworksHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        prompt = body["messages"][-1]["content"]
        print(f"  [mock] got prompt: {prompt[:60]}...")
        response = {
            "id": "mock-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"MOCK_ANSWER for: {prompt[:40]}"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args, **kwargs):
        pass  # quiet the default request log

def main():
    server = http.server.HTTPServer(("0.0.0.0", MOCK_PORT), MockFireworksHandler)
    print(f"Mock Fireworks server running at http://localhost:{MOCK_PORT}")
    print("Waiting for requests... (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping mock server.")

if __name__ == "__main__":
    main()
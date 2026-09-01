"""Localhost-only HTTP relay for OpenRouter calls from an SSH reverse tunnel."""

from __future__ import annotations

import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


UPSTREAM_HOST = "openrouter.ai"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class OpenRouterRelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._relay()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        self._relay()

    def _relay(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
        }
        connection = http.client.HTTPSConnection(UPSTREAM_HOST, timeout=600)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:
            payload = f"OpenRouter relay error: {type(error).__name__}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            connection.close()

    def log_message(self, format: str, *args: object) -> None:
        # Do not log request paths or headers from authenticated model traffic.
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relay an SSH reverse tunnel to OpenRouter without logging payloads."
    )
    parser.add_argument("--port", type=int, default=18041)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), OpenRouterRelayHandler)
    print(f"OpenRouter relay listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

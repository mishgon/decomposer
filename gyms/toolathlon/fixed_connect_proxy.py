"""Loopback-only CONNECT proxy with a fixed DNS override.

Used on Hertz-2 when the system resolver cannot resolve lmrouter, while TLS
hostname verification and SNI must remain intact. The proxy never inspects or
logs request payloads.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class Proxy(socketserver.BaseRequestHandler):
    target_host = "lmrouter.2a2i.org"
    target_ip = "176.108.242.226"

    def handle(self) -> None:
        request = b""
        while b"\r\n\r\n" not in request and len(request) < 65536:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            request += chunk
        first_line = request.split(b"\r\n", 1)[0].decode("ascii", "replace")
        if first_line != f"CONNECT {self.target_host}:443 HTTP/1.1":
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        with socket.create_connection((self.target_ip, 443), timeout=20) as upstream:
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sockets = (self.request, upstream)
            while True:
                readable, _, _ = select.select(sockets, (), (), 60)
                if not readable:
                    continue
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.request else self.request).sendall(data)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18042)
    args = parser.parse_args()
    with Server(("127.0.0.1", args.port), Proxy) as server:
        print(f"Fixed CONNECT proxy listening on 127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

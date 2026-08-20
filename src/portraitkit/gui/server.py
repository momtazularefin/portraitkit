"""Local HTTP server for the inspector UI.

A thin adapter. Routing and transport live here; every decision about what to compute
lives in :mod:`portraitkit.gui.service`, which is why that module is testable without a
socket and this one needs almost no tests of its own.

The server is built on the standard library on purpose. PortraitKit's default detector is
227 KiB and its dependency list is four packages; adding a web framework or a desktop
toolkit to show that off would cost more than the thing it displays. It binds to loopback
only, because it will happily process whatever image it is handed.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from portraitkit.gui.service import AnalysisService

__all__ = ["InspectorServer", "serve"]

_APP_HTML = Path(__file__).with_name("app.html")
_MAX_UPLOAD_BYTES = 32 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    """Routes three endpoints; everything else is a 404."""

    server_version = "PortraitKitInspector/1.0"

    def __init__(self, service: AnalysisService, *args: object, **kwargs: object) -> None:
        self._service = service
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def log_message(self, format: str, *args: object) -> None:
        """Stay quiet. The CLI prints the one line a user needs."""

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send(200, _APP_HTML.read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/options":
            payload = self._service.options().to_dict()
            payload["samples"] = len(self._service.samples())
            self._send_json(200, payload)
        elif route.startswith("/api/sample/"):
            try:
                index = int(route.rsplit("/", 1)[1])
            except ValueError:
                self._send_json(400, {"error": "sample index must be an integer"})
                return
            blob = self._service.sample_bytes(index)
            if blob is None:
                self._send_json(404, {"error": f"no sample {index}"})
            else:
                self._send(200, blob, "image/jpeg")
        else:
            self._send_json(404, {"error": f"no route {route}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyse":
            self._send_json(404, {"error": f"no route {parsed.path}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "malformed Content-Length"})
            return
        if length <= 0:
            self._send_json(400, {"error": "empty upload"})
            return
        if length > _MAX_UPLOAD_BYTES:
            self._send_json(
                413, {"error": f"image exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"}
            )
            return

        query = parse_qs(parsed.query)
        payload = self._service.analyse(
            self.rfile.read(length),
            detector=query.get("detector", ["yunet-2023mar"])[0],
            preset=query.get("preset", ["icao-portrait-35x45"])[0],
        )
        self._send_json(200, payload)


class InspectorServer:
    """Owns the HTTP server's lifetime, so tests can start and stop it cleanly."""

    def __init__(self, service: AnalysisService, *, host: str = "127.0.0.1", port: int = 8000):
        self._httpd = ThreadingHTTPServer((host, port), partial(_Handler, service))
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Address the UI is reachable at."""
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        """Serve in a background thread."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and release the port."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def serve_forever(self) -> None:
        """Serve until interrupted."""
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._httpd.server_close()


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    allow_download: bool | None = None,
) -> None:
    """Run the inspector until interrupted."""
    server = InspectorServer(AnalysisService(allow_download=allow_download), host=host, port=port)
    print(f"PortraitKit Inspector on {server.url}  (ctrl-c to stop)")
    if open_browser:
        webbrowser.open(server.url)
    server.serve_forever()

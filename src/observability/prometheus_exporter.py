"""
Prometheus metrics exporter.

HTTP server that exposes metrics in Prometheus exposition format.
"""

import http.server
import socketserver
from typing import Optional

from .config import get_config
from .metrics import get_registry


class PrometheusHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics."""
    
    def do_GET(self):
        """Handle GET request."""
        if self.path == "/metrics":
            # Get metrics in Prometheus format
            registry = get_registry()
            metrics_text = registry.export_prometheus()
            
            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(metrics_text.encode("utf-8"))
        else:
            # 404 for other paths
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class PrometheusExporter:
    """Prometheus metrics exporter."""
    
    def __init__(self, port: Optional[int] = None):
        """
        Initialize exporter.
        
        Args:
            port: Port to listen on (defaults to config)
        """
        self.config = get_config()
        self.port = port or self.config.prometheus_port
        self.httpd = None
    
    def start(self):
        """Start the exporter server."""
        with socketserver.TCPServer(("", self.port), PrometheusHandler) as httpd:
            self.httpd = httpd
            print(f"Prometheus exporter started on http://localhost:{self.port}/metrics")
            print("Press Ctrl+C to stop")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nExporter stopped.")
    
    def stop(self):
        """Stop the exporter server."""
        if self.httpd:
            self.httpd.shutdown()


def run_exporter():
    """CLI entry point for exporter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prometheus metrics exporter")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on"
    )
    
    args = parser.parse_args()
    
    exporter = PrometheusExporter(port=args.port)
    exporter.start()


if __name__ == "__main__":
    run_exporter()


"""
Sandbox execution package for secure test running.

Provides both Docker-based and fallback tempfile-based sandboxes.
"""

from src.sandbox.docker_sandbox import (
    DockerSandbox,
    DockerSandboxConfig,
    DockerTestResult,
    create_docker_sandbox,
)

__all__ = [
    "DockerSandbox",
    "DockerSandboxConfig",
    "DockerTestResult",
    "create_docker_sandbox",
]


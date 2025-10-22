"""
Docker-based secure sandbox for test execution.

This module provides containerized test execution with:
- Network isolation
- Resource limits (CPU, memory, disk)
- Filesystem isolation
- Timeout protection
- Security sandboxing
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import docker
from docker.errors import ContainerError, DockerException, ImageNotFound
from pydantic import BaseModel, Field
from rich.console import Console

from config.settings import settings

console = Console()


class DockerSandboxConfig(BaseModel):
    """Configuration for Docker sandbox."""
    
    image: str = Field(default="python:3.11-slim", description="Docker image to use")
    mem_limit: str = Field(default="512m", description="Memory limit")
    cpu_quota: int = Field(default=50000, description="CPU quota (50% = 50000/100000)")
    cpu_period: int = Field(default=100000, description="CPU period")
    network_disabled: bool = Field(default=True, description="Disable network")
    readonly_rootfs: bool = Field(default=False, description="Read-only root filesystem")
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    working_dir: str = Field(default="/workspace", description="Working directory in container")


class DockerTestResult(BaseModel):
    """Result from Docker test execution."""
    
    success: bool = Field(..., description="Tests passed")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    exit_code: int = Field(..., description="Exit code")
    tests_run: int = Field(default=0, description="Tests executed")
    tests_passed: int = Field(default=0, description="Tests passed")
    tests_failed: int = Field(default=0, description="Tests failed")
    execution_time: float = Field(default=0.0, description="Execution time")
    coverage: Optional[float] = Field(default=None, description="Coverage %")
    error_details: Optional[str] = Field(default=None, description="Error details")
    container_id: Optional[str] = Field(default=None, description="Container ID")


class DockerSandbox:
    """
    Docker-based secure sandbox for test execution.
    
    Provides enterprise-grade isolation and security for running untrusted test code.
    """
    
    def __init__(
        self,
        config: Optional[DockerSandboxConfig] = None
    ) -> None:
        """
        Initialize Docker sandbox.
        
        Args:
            config: Sandbox configuration
            
        Raises:
            DockerException: If Docker is not available
        """
        self.config = config or DockerSandboxConfig()
        
        try:
            self.client = docker.from_env()
            # Verify Docker is accessible
            self.client.ping()
            console.print("[green]✓[/green] Docker sandbox initialized")
        except DockerException as e:
            console.print(f"[red]✗[/red] Docker not available: {e}")
            raise DockerException(
                "Docker is not available. Please install and start Docker."
            ) from e
        
        self._ensure_image_available()
    
    def _ensure_image_available(self) -> None:
        """Ensure the Docker image is available, pull if necessary."""
        try:
            self.client.images.get(self.config.image)
            console.print(f"[green]✓[/green] Image {self.config.image} is available")
        except ImageNotFound:
            console.print(f"[yellow]⚠[/yellow] Pulling image {self.config.image}...")
            self.client.images.pull(self.config.image)
            console.print(f"[green]✓[/green] Image {self.config.image} pulled successfully")
    
    def execute_tests(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        framework: str = "pytest"
    ) -> DockerTestResult:
        """
        Execute tests in a secure Docker container.
        
        Args:
            test_code: Test code to execute
            source_code: Source code being tested
            requirements: Additional Python packages to install
            framework: Test framework ('pytest', 'unittest', 'jest', 'junit')
            
        Returns:
            DockerTestResult with execution results
            
        Example:
            >>> sandbox = DockerSandbox()
            >>> result = sandbox.execute_tests(
            ...     test_code="def test_add(): assert add(1, 2) == 3",
            ...     source_code="def add(a, b): return a + b"
            ... )
            >>> print(f"Success: {result.success}")
        """
        console.print("[cyan]Executing tests in Docker sandbox...[/cyan]")
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory(prefix="docker_sandbox_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write test file
            test_file = temp_path / "test_generated.py"
            test_file.write_text(test_code, encoding='utf-8')
            
            # Write source file if provided
            if source_code:
                source_file = temp_path / "source_module.py"
                source_file.write_text(source_code, encoding='utf-8')
            
            # Create requirements file
            if requirements:
                req_file = temp_path / "requirements.txt"
                req_file.write_text("\n".join(requirements), encoding='utf-8')
            
            # Create execution script
            exec_script = self._create_execution_script(framework, bool(requirements))
            script_file = temp_path / "run_tests.sh"
            script_file.write_text(exec_script, encoding='utf-8')
            script_file.chmod(0o755)
            
            # Execute in Docker
            result = self._run_container(temp_path)
            
            return result
    
    def execute_with_coverage(
        self,
        test_code: str,
        source_code: str,
        framework: str = "pytest"
    ) -> DockerTestResult:
        """
        Execute tests with coverage analysis.
        
        Args:
            test_code: Test code
            source_code: Source code to measure coverage
            framework: Test framework
            
        Returns:
            DockerTestResult with coverage information
        """
        console.print("[cyan]Executing tests with coverage in Docker...[/cyan]")
        
        requirements = ["pytest-cov"] if framework == "pytest" else ["coverage"]
        
        with tempfile.TemporaryDirectory(prefix="docker_sandbox_cov_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write files
            test_file = temp_path / "test_generated.py"
            test_file.write_text(test_code, encoding='utf-8')
            
            source_file = temp_path / "source_module.py"
            source_file.write_text(source_code, encoding='utf-8')
            
            req_file = temp_path / "requirements.txt"
            req_file.write_text("\n".join(requirements), encoding='utf-8')
            
            # Create coverage execution script
            exec_script = self._create_coverage_script(framework)
            script_file = temp_path / "run_tests.sh"
            script_file.write_text(exec_script, encoding='utf-8')
            script_file.chmod(0o755)
            
            # Execute in Docker
            result = self._run_container(temp_path)
            
            # Parse coverage from output
            if result.success or "TOTAL" in result.stdout:
                result.coverage = self._parse_coverage(result.stdout)
            
            return result
    
    def _create_execution_script(
        self,
        framework: str,
        has_requirements: bool
    ) -> str:
        """Create bash script for test execution."""
        script = "#!/bin/bash\nset -e\n\n"
        
        # Install requirements
        if has_requirements:
            script += "pip install -q -r requirements.txt\n\n"
        else:
            script += "pip install -q pytest\n\n"
        
        # Run tests based on framework
        if framework == "pytest":
            script += "pytest test_generated.py -v --tb=short --no-header 2>&1\n"
        elif framework == "unittest":
            script += "python -m unittest test_generated.py -v 2>&1\n"
        else:
            script += f"echo 'Unsupported framework: {framework}'\nexit 1\n"
        
        return script
    
    def _create_coverage_script(self, framework: str) -> str:
        """Create bash script for test execution with coverage."""
        script = "#!/bin/bash\nset -e\n\n"
        
        if framework == "pytest":
            script += "pip install -q pytest pytest-cov\n\n"
            script += "pytest test_generated.py --cov=source_module --cov-report=term-missing -v --tb=short 2>&1\n"
        else:
            script += "pip install -q coverage\n\n"
            script += "coverage run -m unittest test_generated.py\n"
            script += "coverage report -m\n"
        
        return script
    
    def _run_container(self, host_path: Path) -> DockerTestResult:
        """
        Run Docker container with test execution.
        
        Args:
            host_path: Path to directory with test files
            
        Returns:
            DockerTestResult
        """
        start_time = time.time()
        container = None
        
        try:
            # Container configuration
            container_config = {
                "image": self.config.image,
                "command": "/bin/bash /workspace/run_tests.sh",
                "volumes": {
                    str(host_path): {
                        'bind': self.config.working_dir,
                        'mode': 'rw'
                    }
                },
                "working_dir": self.config.working_dir,
                "mem_limit": self.config.mem_limit,
                "cpu_quota": self.config.cpu_quota,
                "cpu_period": self.config.cpu_period,
                "network_disabled": self.config.network_disabled,
                "detach": True,
                "remove": False,  # Keep for log retrieval
            }
            
            # Run container
            container = self.client.containers.run(**container_config)
            
            # Wait for completion with timeout
            result = container.wait(timeout=self.config.timeout)
            exit_code = result['StatusCode']
            
            # Get logs
            logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='replace')
            
            execution_time = time.time() - start_time
            
            # Parse test results
            tests_run, tests_passed, tests_failed = self._parse_test_output(logs)
            
            # Clean up container
            container.remove(force=True)
            
            return DockerTestResult(
                success=exit_code == 0,
                stdout=logs,
                stderr="",
                exit_code=exit_code,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                execution_time=execution_time,
                container_id=container.id[:12] if container else None
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Clean up container if exists
            if container:
                try:
                    logs = container.logs().decode('utf-8', errors='replace')
                    container.remove(force=True)
                except:
                    logs = ""
            else:
                logs = ""
            
            # Handle timeout
            if execution_time >= self.config.timeout:
                error_msg = f"Test execution timed out after {self.config.timeout} seconds"
            else:
                error_msg = f"Container execution failed: {str(e)}"
            
            return DockerTestResult(
                success=False,
                stdout=logs,
                stderr=error_msg,
                exit_code=-1,
                execution_time=execution_time,
                error_details=error_msg
            )
    
    def _parse_test_output(self, output: str) -> tuple:
        """Parse test output to extract statistics."""
        import re
        
        # Pytest format: "5 passed in 0.12s" or "3 failed, 2 passed in 1.23s"
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        
        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        tests_run = tests_passed + tests_failed
        
        return tests_run, tests_passed, tests_failed
    
    def _parse_coverage(self, output: str) -> Optional[float]:
        """Parse coverage percentage from output."""
        import re
        
        # Look for coverage line: "TOTAL    100    20    80%"
        match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
        if match:
            return float(match.group(1))
        
        return None
    
    def cleanup(self) -> None:
        """Clean up Docker resources."""
        try:
            # Remove dangling containers
            containers = self.client.containers.list(
                all=True,
                filters={"status": "exited", "ancestor": self.config.image}
            )
            for container in containers:
                try:
                    container.remove(force=True)
                except:
                    pass
            
            console.print("[green]✓[/green] Docker sandbox cleaned up")
        except Exception as e:
            console.print(f"[yellow]Warning: Cleanup failed: {e}[/yellow]")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def create_docker_sandbox(
    timeout: int = 30,
    mem_limit: str = "512m",
    network_disabled: bool = True
) -> DockerSandbox:
    """
    Factory function to create a Docker sandbox.
    
    Args:
        timeout: Execution timeout in seconds
        mem_limit: Memory limit (e.g., '512m', '1g')
        network_disabled: Disable network access
        
    Returns:
        Configured DockerSandbox instance
    """
    config = DockerSandboxConfig(
        timeout=timeout,
        mem_limit=mem_limit,
        network_disabled=network_disabled
    )
    return DockerSandbox(config=config)


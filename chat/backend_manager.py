"""Backend manager for starting/stopping LLM inference servers."""

import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
import httpx

from config import (
    VLLM_CMD, LLAMACPP_CMD,
    DEFAULT_VLLM_URL, DEFAULT_LLAMACPP_URL,
    BACKEND_STARTUP_TIMEOUT, BACKEND_LOGS_DIR
)


class BackendManager:
    """Manages LLM inference backend processes."""

    def __init__(self):
        self.current_backend: Optional[str] = None
        self.current_model: Optional[str] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.log_file = None

    def get_status(self) -> dict:
        """Get current backend status."""
        if self.current_process is None:
            return {
                "status": "stopped",
                "backend": None,
                "model": None
            }

        # Check if process is still running
        if self.current_process.poll() is not None:
            # Process has terminated
            self.current_process = None
            self.current_backend = None
            self.current_model = None
            return {
                "status": "stopped",
                "backend": None,
                "model": None
            }

        return {
            "status": "running",
            "backend": self.current_backend,
            "model": self.current_model
        }

    def get_backend_url(self, backend: str) -> str:
        """Get the URL for a backend."""
        if backend == "vllm":
            return DEFAULT_VLLM_URL
        elif backend == "llamacpp":
            return DEFAULT_LLAMACPP_URL
        raise ValueError(f"Unknown backend: {backend}")

    def is_server_ready(self, backend: str) -> bool:
        """Check if the server is ready by pinging /v1/models."""
        url = self.get_backend_url(backend)
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{url}/v1/models")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def stop_current(self) -> Generator[str, None, None]:
        """Stop the current backend process."""
        if self.current_process is None:
            yield "No backend running"
            return

        yield f"Stopping {self.current_backend}..."

        # Send SIGTERM for graceful shutdown
        self.current_process.terminate()

        # Wait for process to terminate (with timeout)
        try:
            self.current_process.wait(timeout=10)
            yield f"{self.current_backend} stopped"
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't respond
            yield f"Force killing {self.current_backend}..."
            self.current_process.kill()
            self.current_process.wait()
            yield f"{self.current_backend} killed"

        if self.log_file:
            self.log_file.close()
            self.log_file = None

        self.current_process = None
        self.current_backend = None
        self.current_model = None

    def start_backend(
        self,
        backend: str,
        model_path: str
    ) -> Generator[str, None, bool]:
        """
        Start a backend with the specified model.

        Yields status messages during startup.
        Returns True if startup was successful.
        """
        # Check if already running with same config
        if (self.current_backend == backend and
            self.current_model == model_path and
            self.current_process is not None and
            self.current_process.poll() is None):
            yield "Backend already running with this model"
            return True

        # Stop current backend if different
        if self.current_process is not None:
            yield from self.stop_current()
            # Wait a bit for port to be released
            time.sleep(2)

        # Prepare command
        if backend == "vllm":
            cmd = [arg.replace("{model_path}", model_path) for arg in VLLM_CMD]
        elif backend == "llamacpp":
            cmd = [arg.replace("{model_path}", model_path) for arg in LLAMACPP_CMD]
        else:
            yield f"Unknown backend: {backend}"
            return False

        # Create logs directory
        BACKEND_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = BACKEND_LOGS_DIR / f"{backend}_{timestamp}.log"

        yield f"Starting {backend}..."
        yield f"Model: {Path(model_path).name}"
        yield f"Command: {' '.join(cmd)}"
        yield f"Log file: {log_path}"

        # Start the process
        try:
            self.log_file = open(log_path, "w")
            self.current_process = subprocess.Popen(
                cmd,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
            )
        except FileNotFoundError as e:
            yield f"Error: Command not found - {e}"
            return False
        except Exception as e:
            yield f"Error starting backend: {e}"
            return False

        self.current_backend = backend
        self.current_model = model_path

        # Wait for server to be ready
        yield "Waiting for server to be ready..."
        start_time = time.time()
        check_interval = 2  # seconds

        while time.time() - start_time < BACKEND_STARTUP_TIMEOUT:
            # Check if process crashed
            if self.current_process.poll() is not None:
                yield f"Error: {backend} process terminated unexpectedly"
                yield f"Check log file: {log_path}"
                self.current_process = None
                self.current_backend = None
                self.current_model = None
                return False

            if self.is_server_ready(backend):
                yield f"{backend} is ready!"
                return True

            elapsed = int(time.time() - start_time)
            yield f"Waiting... ({elapsed}s / {BACKEND_STARTUP_TIMEOUT}s)"
            time.sleep(check_interval)

        # Timeout reached
        yield f"Timeout: Server not ready after {BACKEND_STARTUP_TIMEOUT}s"
        yield from self.stop_current()
        return False

    def cleanup(self):
        """Clean up resources on shutdown."""
        if self.current_process is not None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

        if self.log_file:
            self.log_file.close()


# Global instance
backend_manager = BackendManager()


if __name__ == "__main__":
    # Test the backend manager
    import sys

    if len(sys.argv) < 3:
        print("Usage: python backend_manager.py <backend> <model_path>")
        print("Example: python backend_manager.py vllm models/hf/Qwen3-8B-AWQ")
        sys.exit(1)

    backend = sys.argv[1]
    model_path = sys.argv[2]

    print(f"Starting {backend} with {model_path}...")
    for msg in backend_manager.start_backend(backend, model_path):
        print(f"  {msg}")

    print("\nStatus:", backend_manager.get_status())

    input("\nPress Enter to stop...")

    for msg in backend_manager.stop_current():
        print(f"  {msg}")

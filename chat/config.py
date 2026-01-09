"""Configuration for the LLM Chat UI."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
CHAT_HISTORY_DIR = BASE_DIR / ".chat_history"
BACKEND_LOGS_DIR = BASE_DIR / ".backend_logs"

# Model subdirectories
VLLM_MODELS_SUBDIR = "hf"
LLAMACPP_MODELS_SUBDIR = "gguf"

# Backend URLs
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_LLAMACPP_URL = "http://localhost:8001"

# Backend commands
VLLM_CMD = [
    "vllm", "serve", "{model_path}",
    "--port", "8000",
    "--gpu-memory-utilization", "0.90"
]

LLAMACPP_CMD = [
    str(BASE_DIR / "llama.cpp" / "build" / "bin" / "llama-server"),
    "--model", "{model_path}",
    "--n-gpu-layers", "-1",
    "--ctx-size", "4096",
    "--port", "8001",
    "--host", "0.0.0.0"
]

# Timeouts
BACKEND_STARTUP_TIMEOUT = 120  # seconds
LLM_REQUEST_TIMEOUT = 300  # seconds

# Generation defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

"""Model scanner for finding available models."""

from pathlib import Path
from typing import Optional
from config import MODELS_DIR, VLLM_MODELS_SUBDIR, LLAMACPP_MODELS_SUBDIR


def scan_vllm_models(models_dir: Optional[Path] = None) -> list[dict]:
    """
    Scan for vLLM models (HuggingFace format).

    vLLM models are stored as directories containing model files
    (config.json, safetensors, etc.)
    """
    models_dir = models_dir or MODELS_DIR
    hf_dir = models_dir / VLLM_MODELS_SUBDIR

    if not hf_dir.exists():
        return []

    models = []
    for item in hf_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Check if it looks like a model directory
            config_file = item / "config.json"
            if config_file.exists():
                models.append({
                    "name": item.name,
                    "path": str(item),
                    "backend": "vllm"
                })

    return sorted(models, key=lambda x: x["name"])


def scan_llamacpp_models(models_dir: Optional[Path] = None) -> list[dict]:
    """
    Scan for llama.cpp models (GGUF format).

    llama.cpp models are .gguf files.
    """
    models_dir = models_dir or MODELS_DIR
    gguf_dir = models_dir / LLAMACPP_MODELS_SUBDIR

    if not gguf_dir.exists():
        return []

    models = []
    for item in gguf_dir.glob("*.gguf"):
        if item.is_file():
            models.append({
                "name": item.stem,  # filename without extension
                "path": str(item),
                "backend": "llamacpp"
            })

    return sorted(models, key=lambda x: x["name"])


def get_all_models(models_dir: Optional[Path] = None) -> dict:
    """Get all available models for both backends."""
    return {
        "vllm": scan_vllm_models(models_dir),
        "llamacpp": scan_llamacpp_models(models_dir)
    }


if __name__ == "__main__":
    # Test the scanner
    models = get_all_models()
    print("vLLM models:")
    for m in models["vllm"]:
        print(f"  - {m['name']}: {m['path']}")
    print("\nllama.cpp models:")
    for m in models["llamacpp"]:
        print(f"  - {m['name']}: {m['path']}")

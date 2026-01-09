"""LLM client for streaming chat with vLLM and llama.cpp backends."""

import json
from typing import Generator
import httpx

from config import (
    DEFAULT_VLLM_URL, DEFAULT_LLAMACPP_URL,
    LLM_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P
)


def get_backend_url(backend: str) -> str:
    """Get the URL for a backend."""
    if backend == "vllm":
        return DEFAULT_VLLM_URL
    elif backend == "llamacpp":
        return DEFAULT_LLAMACPP_URL
    raise ValueError(f"Unknown backend: {backend}")


def get_model_name(backend: str) -> str:
    """Get the model name from the server."""
    url = get_backend_url(backend)
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{url}/v1/models")
            data = response.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id", "unknown")
    except Exception:
        pass
    return "unknown"


def stream_chat(
    backend: str,
    messages: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_p: float = DEFAULT_TOP_P
) -> Generator[dict, None, None]:
    """
    Stream chat completion from the backend.

    Yields dicts with:
    - {"type": "content", "text": "..."}
    - {"type": "reasoning", "text": "..."}
    - {"type": "done"}
    - {"type": "error", "message": "..."}
    """
    url = get_backend_url(backend)
    model = get_model_name(backend)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    try:
        with httpx.Client(timeout=LLM_REQUEST_TIMEOUT) as client:
            with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    # Handle bytes or str
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")

                    line = line.strip()

                    # Parse SSE format
                    if not line.startswith("data:"):
                        # Handle raw JSON without SSE prefix
                        if line.startswith("{") and '"choices"' in line:
                            data_str = line
                        else:
                            continue
                    else:
                        data_str = line[5:].strip()

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])

                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Extract content
                        content = delta.get("content", "") or ""
                        if content:
                            yield {"type": "content", "text": content}

                        # Extract reasoning content (for reasoning models)
                        reasoning = delta.get("reasoning_content", "") or ""
                        if reasoning:
                            yield {"type": "reasoning", "text": reasoning}

                        # Fallbacks for non-standard responses
                        if not content and not reasoning:
                            # Try text field
                            if "text" in choices[0]:
                                yield {"type": "content", "text": choices[0]["text"]}
                            # Try message.content
                            elif "message" in choices[0]:
                                msg_content = choices[0]["message"].get("content", "")
                                if msg_content:
                                    yield {"type": "content", "text": msg_content}

                    except json.JSONDecodeError:
                        continue

        yield {"type": "done"}

    except httpx.ConnectError:
        yield {"type": "error", "message": f"Cannot connect to {backend} at {url}"}
    except httpx.HTTPStatusError as e:
        yield {"type": "error", "message": f"HTTP error: {e.response.status_code}"}
    except httpx.TimeoutException:
        yield {"type": "error", "message": "Request timed out"}
    except Exception as e:
        yield {"type": "error", "message": str(e)}


if __name__ == "__main__":
    # Test the client
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "vllm"
    messages = [{"role": "user", "content": "Hello! Tell me a short joke."}]

    print(f"Testing {backend}...")
    print("Response:", end=" ", flush=True)

    for chunk in stream_chat(backend, messages):
        if chunk["type"] == "content":
            print(chunk["text"], end="", flush=True)
        elif chunk["type"] == "reasoning":
            print(f"\n[Reasoning: {chunk['text']}]", end="", flush=True)
        elif chunk["type"] == "error":
            print(f"\nError: {chunk['message']}")
        elif chunk["type"] == "done":
            print("\n[Done]")

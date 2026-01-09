#!/usr/bin/env python3
"""
LLM Inference Benchmark Script
Compares vLLM and llama.cpp performance metrics.

Metrics:
- TTFT (Time to First Token)
- TPS (Tokens Per Second)
- ITL (Inter-Token Latency)
- E2E Latency (End-to-End)
"""

import argparse
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx


def create_client(base_url: str, timeout: float = 300.0) -> httpx.Client:
    """Create HTTP client for API requests."""
    return httpx.Client(base_url=base_url, timeout=timeout)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: 4 chars per token)."""
    return max(1, len(text) // 4)


def run_single_benchmark(
    client: httpx.Client,
    prompt: str,
    max_tokens: int,
    model: str = "default"
) -> dict:
    """Run a single inference request and measure metrics."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    full_response = ""
    token_count = 0

    try:
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                
                # Handle bytes or str
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                
                # DEBUG: Uncomment to see raw lines
                # print(f"RAW: {line!r}")
                    
                line = line.strip()
                if not line.startswith("data:"):
                    # Handle case where "data:" might be missing or different format
                    if line.startswith("{") and '"choices"' in line:
                         # Maybe raw JSON without SSE prefix?
                         data_str = line
                    else:
                         continue
                else:
                    data_str = line[5:].strip()

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    
                    # DEBUG: Print structure of first chunk
                    if first_token_time is None and token_count == 0:
                         # print(f"FIRST CHUNK: {data}")
                         pass

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                        
                    delta = choices[0].get("delta", {})
                    
                    # Combine content and reasoning_content to catch everything
                    content_part = delta.get("content", "") or ""
                    reasoning_part = delta.get("reasoning_content", "") or ""
                    content = content_part + reasoning_part
                    
                    # Fallback for some non-standard responses
                    if not content and "text" in choices[0]:
                        content = choices[0]["text"]
                        
                    # Another fallback: "message" field
                    if not content:
                        content = choices[0].get("message", {}).get("content", "")

                    if content:
                        current_time = time.perf_counter()

                        if first_token_time is None:
                            first_token_time = current_time

                        token_times.append(current_time)
                        full_response += content
                        token_count += 1
                    else:
                        # Debug: Print why content is missing only for the first few failures
                        # print(f"DEBUG: No content in delta: {delta}")
                        pass

                except json.JSONDecodeError:
                    print(f"DEBUG: JSON error: {data_str[:50]}...")
                    continue

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time

    # Inter-token latencies
    itl_values = []
    if len(token_times) > 1:
        for i in range(1, len(token_times)):
            itl_values.append(token_times[i] - token_times[i-1])

    # Tokens per second (generation only, excluding TTFT)
    generation_time = total_time - ttft if token_count > 1 else total_time
    tps = token_count / generation_time if generation_time > 0 else 0

    return {
        "ttft_ms": ttft * 1000,
        "tps": tps,
        "itl_mean_ms": statistics.mean(itl_values) * 1000 if itl_values else 0,
        "itl_p50_ms": statistics.median(itl_values) * 1000 if itl_values else 0,
        "itl_p95_ms": (sorted(itl_values)[int(len(itl_values) * 0.95)] * 1000) if len(itl_values) > 1 else 0,
        "e2e_latency_ms": total_time * 1000,
        "tokens_generated": token_count,
        "prompt_tokens": count_tokens_approx(prompt),
    }


def run_benchmark_suite(
    client: httpx.Client,
    prompt: str,
    max_tokens: int,
    warmup_runs: int = 5,
    measurement_runs: int = 20,
    model: str = "default"
) -> dict:
    """Run full benchmark suite with warmup and measurements."""

    print(f"  Warming up ({warmup_runs} runs)...", end=" ", flush=True)
    for _ in range(warmup_runs):
        run_single_benchmark(client, prompt, max_tokens, model)
    print("done")

    print(f"  Measuring ({measurement_runs} runs)...", end=" ", flush=True)
    results = []
    for i in range(measurement_runs):
        result = run_single_benchmark(client, prompt, max_tokens, model)
        if "error" not in result:
            results.append(result)
        print(".", end="", flush=True)
    print(" done")

    if not results:
        return {"error": "All runs failed"}

    # Aggregate results
    def agg(key):
        values = [r[key] for r in results if key in r]
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p50": sorted_vals[n // 2],
            "p95": sorted_vals[int(n * 0.95)] if n > 1 else sorted_vals[-1],
            "p99": sorted_vals[int(n * 0.99)] if n > 1 else sorted_vals[-1],
        }

    return {
        "ttft_ms": agg("ttft_ms"),
        "tps": agg("tps"),
        "itl_mean_ms": agg("itl_mean_ms"),
        "e2e_latency_ms": agg("e2e_latency_ms"),
        "tokens_generated": agg("tokens_generated"),
        "successful_runs": len(results),
        "total_runs": measurement_runs,
    }


def get_model_name(client: httpx.Client) -> str:
    """Get model name from server."""
    try:
        response = client.get("/v1/models")
        data = response.json()
        models = data.get("data", [])
        if models:
            return models[0].get("id", "unknown")
    except:
        pass
    return "unknown"


def print_results(results: dict, backend: str, scenario: str):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Results: {backend} - {scenario}")
    print(f"{'='*60}")

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"Successful runs: {results['successful_runs']}/{results['total_runs']}")
    print()

    metrics = [
        ("TTFT (ms)", "ttft_ms"),
        ("TPS (tokens/sec)", "tps"),
        ("ITL mean (ms)", "itl_mean_ms"),
        ("E2E Latency (ms)", "e2e_latency_ms"),
        ("Tokens generated", "tokens_generated"),
    ]

    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'P50':>10} {'P95':>10}")
    print("-" * 65)

    for name, key in metrics:
        m = results[key]
        if key == "tps":
            print(f"{name:<25} {m['mean']:>10.1f} {m['std']:>10.1f} {m['p50']:>10.1f} {m['p95']:>10.1f}")
        elif key == "tokens_generated":
            print(f"{name:<25} {m['mean']:>10.0f} {m['std']:>10.1f} {m['p50']:>10.0f} {m['p95']:>10.0f}")
        else:
            print(f"{name:<25} {m['mean']:>10.1f} {m['std']:>10.1f} {m['p50']:>10.1f} {m['p95']:>10.1f}")


# Test prompts
PROMPTS = {
    "short": {
        "prompt": "Explain quantum computing in simple terms.",
        "max_tokens": 100,
        "description": "Short prompt (~10 tokens) + 100 token generation"
    },
    "medium": {
        "prompt": """You are a helpful AI assistant. Please analyze the following text and provide a detailed summary:

The development of artificial intelligence has progressed rapidly over the past decade. Machine learning algorithms have become increasingly sophisticated, enabling applications from image recognition to natural language processing. Deep learning, a subset of machine learning using neural networks with multiple layers, has been particularly transformative. These systems can now generate human-like text, create realistic images, and even assist in scientific research.

Provide a comprehensive analysis of the key points and their implications for the future.""",
        "max_tokens": 300,
        "description": "Medium prompt (~100 tokens) + 300 token generation"
    },
    "long": {
        "prompt": """You are an expert software architect. Please provide a detailed technical design document for the following system:

Design a distributed microservices architecture for a large-scale e-commerce platform that needs to handle:
- 10 million daily active users
- 100,000 concurrent transactions
- Real-time inventory management across multiple warehouses
- Personalized recommendations based on user behavior
- Multi-region deployment for global availability
- Event-driven architecture for order processing
- CQRS pattern for read/write separation
- Circuit breaker patterns for resilience

Include details about:
1. Service decomposition strategy
2. Inter-service communication patterns
3. Data storage solutions for different services
4. Caching strategies
5. Message queue implementation
6. API gateway design
7. Authentication and authorization
8. Monitoring and observability
9. Deployment and scaling strategies
10. Disaster recovery planning

Provide specific technology recommendations and justify your choices.""",
        "max_tokens": 500,
        "description": "Long prompt (~200 tokens) + 500 token generation"
    }
}


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--backend", type=str, required=True,
                       choices=["vllm", "llamacpp", "both"],
                       help="Backend to benchmark")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000",
                       help="vLLM server URL (default: http://localhost:8000)")
    parser.add_argument("--llamacpp-url", type=str, default="http://localhost:8001",
                       help="llama.cpp server URL (default: http://localhost:8001)")
    parser.add_argument("--prompt", type=str, default="all",
                       choices=["short", "medium", "long", "all"],
                       help="Prompt type to use (default: all)")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Number of warmup runs (default: 5)")
    parser.add_argument("--runs", type=int, default=20,
                       help="Number of measurement runs (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--scenario", type=str, default="custom",
                       help="Scenario name for results (default: custom)")

    args = parser.parse_args()

    backends = []
    if args.backend == "both":
        backends = [("vllm", args.vllm_url), ("llamacpp", args.llamacpp_url)]
    elif args.backend == "vllm":
        backends = [("vllm", args.vllm_url)]
    else:
        backends = [("llamacpp", args.llamacpp_url)]

    prompts = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "scenario": args.scenario,
        "config": {
            "warmup_runs": args.warmup,
            "measurement_runs": args.runs,
        },
        "results": {}
    }

    for backend_name, url in backends:
        print(f"\n{'#'*60}")
        print(f"# Benchmarking: {backend_name} at {url}")
        print(f"{'#'*60}")

        try:
            client = create_client(url)
            model = get_model_name(client)
            print(f"Model: {model}")

            all_results["results"][backend_name] = {"model": model, "prompts": {}}

            for prompt_name in prompts:
                prompt_config = PROMPTS[prompt_name]
                print(f"\nPrompt: {prompt_name} - {prompt_config['description']}")

                results = run_benchmark_suite(
                    client,
                    prompt_config["prompt"],
                    prompt_config["max_tokens"],
                    warmup_runs=args.warmup,
                    measurement_runs=args.runs,
                    model=model
                )

                all_results["results"][backend_name]["prompts"][prompt_name] = results
                print_results(results, backend_name, prompt_name)

            client.close()

        except httpx.ConnectError:
            print(f"Error: Could not connect to {backend_name} at {url}")
            print("Make sure the server is running.")
            all_results["results"][backend_name] = {"error": f"Connection failed to {url}"}

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"benchmark_{args.scenario}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print comparison summary if both backends tested
    if len(all_results["results"]) == 2 and all("error" not in v for v in all_results["results"].values()):
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)

        for prompt_name in prompts:
            print(f"\n{prompt_name.upper()}:")
            vllm_res = all_results["results"]["vllm"]["prompts"].get(prompt_name, {})
            llama_res = all_results["results"]["llamacpp"]["prompts"].get(prompt_name, {})

            if "error" in vllm_res or "error" in llama_res:
                continue

            metrics = [("TTFT", "ttft_ms", "ms", True),
                      ("TPS", "tps", "tok/s", False),
                      ("E2E", "e2e_latency_ms", "ms", True)]

            print(f"  {'Metric':<10} {'vLLM':>12} {'llama.cpp':>12} {'Winner':>12}")
            print(f"  {'-'*46}")

            for name, key, unit, lower_better in metrics:
                v_val = vllm_res.get(key, {}).get("mean", 0)
                l_val = llama_res.get(key, {}).get("mean", 0)

                if lower_better:
                    winner = "vLLM" if v_val < l_val else "llama.cpp"
                else:
                    winner = "vLLM" if v_val > l_val else "llama.cpp"

                print(f"  {name:<10} {v_val:>10.1f}{unit:>2} {l_val:>10.1f}{unit:>2} {winner:>12}")


if __name__ == "__main__":
    main()

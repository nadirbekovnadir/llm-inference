#!/usr/bin/env python3
"""
Compare benchmark results from separate runs
"""

import argparse
import json
from pathlib import Path


def load_results(file_path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_comparison(results1: dict, results2: dict, backend1: str, backend2: str):
    """Print comparison between two benchmark results."""

    print("\n" + "="*80)
    print(f"COMPARISON: {backend1} vs {backend2}")
    print("="*80)

    # Get common prompts
    prompts1 = set(results1.get("results", {}).get(backend1, {}).get("prompts", {}).keys())
    prompts2 = set(results2.get("results", {}).get(backend2, {}).get("prompts", {}).keys())
    common_prompts = prompts1 & prompts2

    if not common_prompts:
        print("No common prompts found between results")
        return

    for prompt_name in sorted(common_prompts):
        print(f"\n{prompt_name.upper()}:")

        res1 = results1["results"][backend1]["prompts"][prompt_name]
        res2 = results2["results"][backend2]["prompts"][prompt_name]

        if "error" in res1 or "error" in res2:
            print(f"  Skipped: One or both runs failed")
            continue

        metrics = [
            ("TTFT", "ttft_ms", "ms", True),
            ("TPS", "tps", "tok/s", False),
            ("E2E", "e2e_latency_ms", "ms", True)
        ]

        print(f"  {'Metric':<10} {backend1:>15} {backend2:>15} {'Winner':>15} {'Speedup':>12}")
        print(f"  {'-'*70}")

        for name, key, unit, lower_better in metrics:
            val1 = res1.get(key, {}).get("mean", 0)
            val2 = res2.get(key, {}).get("mean", 0)

            if val1 == 0 or val2 == 0:
                continue

            if lower_better:
                winner = backend1 if val1 < val2 else backend2
                speedup = val2 / val1 if val1 < val2 else val1 / val2
                speedup_str = f"{speedup:.2f}x"
            else:
                winner = backend1 if val1 > val2 else backend2
                speedup = val1 / val2 if val1 > val2 else val2 / val1
                speedup_str = f"{speedup:.2f}x"

            print(f"  {name:<10} {val1:>13.1f}{unit:>2} {val2:>13.1f}{unit:>2} {winner:>15} {speedup_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("file1", type=str, help="First benchmark results JSON file")
    parser.add_argument("file2", type=str, help="Second benchmark results JSON file")
    parser.add_argument("--backend1", type=str, default="vllm",
                       help="Backend name in first file (default: vllm)")
    parser.add_argument("--backend2", type=str, default="llamacpp",
                       help="Backend name in second file (default: llamacpp)")

    args = parser.parse_args()

    results1 = load_results(args.file1)
    results2 = load_results(args.file2)

    print(f"\nFile 1: {args.file1}")
    print(f"  Scenario: {results1.get('scenario', 'unknown')}")
    print(f"  Timestamp: {results1.get('timestamp', 'unknown')}")
    print(f"  Backend: {args.backend1}")

    print(f"\nFile 2: {args.file2}")
    print(f"  Scenario: {results2.get('scenario', 'unknown')}")
    print(f"  Timestamp: {results2.get('timestamp', 'unknown')}")
    print(f"  Backend: {args.backend2}")

    print_comparison(results1, results2, args.backend1, args.backend2)


if __name__ == "__main__":
    main()

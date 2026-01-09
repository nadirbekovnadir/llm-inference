# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM inference playground for experimenting with local LLM model inference using vLLM and llama.cpp backends. The project is focused on performance benchmarking and comparison between these two inference engines.

**Primary Use Cases:**
- Running local LLM inference with vLLM (HuggingFace models) and llama.cpp (GGUF models)
- Benchmarking and comparing performance metrics (TTFT, TPS, ITL, E2E latency)
- Testing different quantization strategies and GPU/CPU offloading configurations

## Repository Structure

```
llm-inference/
├── requirements.txt       # Python dependencies (vllm, httpx)
├── venv-vllm/            # Python virtual environment (gitignored)
├── llama.cpp/            # llama.cpp submodule (gitignored, cloned separately)
├── models/               # Model storage (gitignored)
│   ├── hf/              # HuggingFace format models (for vLLM)
│   └── gguf/            # GGUF quantized models (for llama.cpp)
├── benchmark/           # Benchmarking tools
│   ├── benchmark.py     # Main benchmark script (streaming API-based measurements)
│   ├── compare_results.py # Comparison tool for separate runs
│   └── README.md        # Detailed benchmark documentation
└── results/             # Benchmark results JSON files (gitignored)
```

## Common Commands

### Environment Setup

```bash
# Create and activate virtual environment
uv venv venv-vllm
source venv-vllm/bin/activate

# Install Python dependencies
uv pip install -r requirements.txt

# Clone and build llama.cpp with CUDA support
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
rm -rf build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

### Downloading Models

```bash
# vLLM models (HuggingFace format)
huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir models/hf/Qwen3-8B-AWQ
huggingface-cli download Qwen/Qwen3-32B-FP8 --local-dir models/hf/Qwen3-32B-FP8

# llama.cpp models (GGUF format)
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models/gguf
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf --local-dir models/gguf
```

### Running Inference Servers

```bash
# vLLM server (default port 8000)
source venv-vllm/bin/activate
vllm serve models/hf/Qwen3-8B-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.90

# vLLM with CPU offloading (for larger models)
vllm serve models/hf/Qwen3-32B-FP8 \
    --port 8000 \
    --cpu-offload-gb 20 \
    --gpu-memory-utilization 0.95

# llama.cpp server (default port 8001)
./llama.cpp/build/bin/llama-server \
    --model models/gguf/Qwen3-8B-Q4_K_M.gguf \
    --n-gpu-layers -1 \
    --ctx-size 4096 \
    --port 8001 \
    --host 0.0.0.0

# llama.cpp with partial GPU offload (30 layers)
./llama.cpp/build/bin/llama-server \
    --model models/gguf/Qwen3-32B-Q8_0.gguf \
    --n-gpu-layers 30 \
    --ctx-size 4096 \
    --port 8001
```

### Testing API Endpoints

```bash
# Check vLLM is running
curl http://localhost:8000/v1/models

# Test vLLM completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Check llama.cpp is running
curl http://localhost:8001/v1/models

# Test llama.cpp completion
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Running Benchmarks

```bash
source venv-vllm/bin/activate

# Benchmark single backend (recommended approach due to GPU memory constraints)
python benchmark/benchmark.py \
    --backend vllm \
    --scenario "qwen3-8b-gpu" \
    --prompt all \
    --warmup 5 \
    --runs 20

python benchmark/benchmark.py \
    --backend llamacpp \
    --scenario "qwen3-8b-gpu" \
    --prompt all \
    --warmup 5 \
    --runs 20

# Benchmark specific prompt type
python benchmark/benchmark.py \
    --backend vllm \
    --scenario "qwen3-8b-gpu" \
    --prompt short \
    --warmup 5 \
    --runs 20

# Compare results from separate runs
python benchmark/compare_results.py \
    benchmark/results/benchmark_qwen3-8b-gpu_*.json \
    benchmark/results/benchmark_qwen3-8b-gpu_*.json \
    --backend1 vllm \
    --backend2 llamacpp

# Simultaneous testing (only if >24GB VRAM available)
python benchmark/benchmark.py \
    --backend both \
    --scenario "qwen3-8b-comparison" \
    --prompt all
```

## Architecture and Design

### Benchmark Implementation

The benchmarking system in `benchmark/benchmark.py` uses streaming API calls to accurately measure inference performance metrics:

**Key Components:**
- `run_single_benchmark()` - Executes a single inference request with streaming, measures TTFT and inter-token latencies
- `run_benchmark_suite()` - Runs warmup + measurement cycles, aggregates statistics (mean, std, p50, p95, p99)
- `create_client()` - Creates httpx client with 300s timeout for long-running requests
- `count_tokens_approx()` - Rough token count estimation (4 chars per token)

**Metrics Collected:**
- **TTFT (Time to First Token)**: Time from request start to first token received (ms)
- **TPS (Tokens Per Second)**: Generation speed excluding TTFT
- **ITL (Inter-Token Latency)**: Average delay between consecutive tokens (ms)
- **E2E Latency**: Total time from request to completion (ms)

**Prompt Types:**
- `short`: ~10 tokens input, 100 tokens output (minimal overhead testing)
- `medium`: ~100 tokens input, 300 tokens output (realistic chatbot scenario)
- `long`: ~200 tokens input, 500 tokens output (stress test with long context)

**Results Format:**
- JSON files saved to `benchmark/results/benchmark_<scenario>_<timestamp>.json`
- Includes configuration, per-prompt statistics, and comparison summaries
- Statistics include mean, std, min, max, p50, p95, p99 for all metrics

### Backend Comparison

**vLLM:**
- Uses PagedAttention for efficient memory management
- Better for batch inference and parallel requests
- Optimized for models fully loaded in GPU memory
- Supports AWQ, FP8 quantization formats
- Default port: 8000

**llama.cpp:**
- Efficient CPU/GPU hybrid inference with flexible layer offloading
- Better single-request latency in CPU offload scenarios
- Supports GGUF quantization (Q4_K_M, Q8_0, etc.)
- More memory-efficient with aggressive quantization
- Default port: 8001

### Testing Workflow

Due to GPU memory constraints, the typical benchmark workflow is:
1. Start vLLM server on port 8000
2. Run benchmark for vLLM backend
3. Stop vLLM server
4. Start llama.cpp server on port 8001
5. Run benchmark for llama.cpp backend
6. Use `compare_results.py` to analyze both results

## Development Notes

- The project uses `uv` for fast Python dependency management, but standard `pip` works too
- Virtual environment is named `venv-vllm` (not `venv`) to indicate it includes vLLM installation
- llama.cpp must be built with CUDA support using cmake flags: `-DGGML_CUDA=ON`
- Models are excluded from git due to size (stored in `models/` directory)
- Benchmark results are gitignored but can be tracked by uncommenting in `.gitignore`
- All Python code is standalone scripts, no package structure
- The README is in Russian but code/comments use English conventions

## Important Parameters

**vLLM Server:**
- `--gpu-memory-utilization`: GPU VRAM usage (0.7-0.95), default 0.90
- `--cpu-offload-gb`: Amount of model to offload to CPU RAM
- `--max-model-len`: Maximum context length
- `--tensor-parallel-size`: Number of GPUs for parallelism

**llama.cpp Server:**
- `--n-gpu-layers`: Number of layers in GPU (-1 = all layers)
- `--ctx-size`: Context size (e.g., 4096)
- `--parallel`: Number of parallel requests
- `--threads`: CPU threads for non-GPU operations

**Benchmark Script:**
- `--backend`: Which backend to test (vllm|llamacpp|both)
- `--prompt`: Prompt type (short|medium|long|all)
- `--warmup`: Number of warmup runs (default 5)
- `--runs`: Number of measurement runs (default 20)
- `--scenario`: Scenario name for result files (default: custom)

## Hardware Context

The project was developed on:
- Ubuntu 24.04 (WSL2)
- RTX 4080 SUPER GPU
- CUDA 12.4
- Typical VRAM constraints require sequential testing of backends

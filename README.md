# LLM Inference Playground

Локальный инференс LLM моделей с использованием vLLM и llama.cpp. Проект для экспериментов с различными моделями, настройками и бенчмарком производительности.

## 🎯 Возможности

- **vLLM** - высокопроизводительный инференс через PagedAttention
- **llama.cpp** - эффективный CPU/GPU инференс с GGUF квантизацией
- **Benchmark** - сравнение производительности разных бэкендов
- **Гибкая конфигурация** - CPU offloading, квантизация, memory management

## 📁 Структура проекта

```
llm-inference/
├── venv-vllm/          # Python окружение для vLLM
├── llama.cpp/          # llama.cpp с CUDA поддержкой
├── models/             # Скачанные модели
│   ├── hf/            # HuggingFace модели (для vLLM)
│   └── gguf/          # GGUF модели (для llama.cpp)
├── benchmark/         # Инструменты бенчмарка
│   ├── benchmark.py   # Скрипт измерения производительности
│   ├── compare_results.py  # Сравнение результатов
│   └── results/       # Сохранённые результаты
└── README.md
```

## 🔧 Системные требования

- **GPU**: NVIDIA с 16+ GB VRAM (тестировалось на RTX 4080 SUPER)
- **RAM**: 32+ GB (62+ GB для CPU offloading больших моделей)
- **CUDA**: 12.4+
- **OS**: Linux / WSL2
- **Python**: 3.12+
- **System Packages**: `build-essential`, `python3-dev`, `cmake` (для сборки llama.cpp)

## 🚀 Установка

### 1. Подготовка системы

Перед началом работы установите необходимые системные пакеты:

```bash
sudo apt update
sudo apt install build-essential python3-dev cmake git
```

### 2. Клонирование репозитория

```bash
git clone https://github.com/your-username/llm-inference.git
cd llm-inference
```

### 3. vLLM

```bash
# Создание и активация окружения (используя uv для скорости)
# Если uv не установлен: pip install uv
uv venv venv-vllm
source venv-vllm/bin/activate

# Установка зависимостей
uv pip install vllm httpx

# Проверка установки
python -c "from vllm import LLM; print('vLLM OK')"
```

**Известные проблемы на WSL2**: vLLM v0.13.0 имеет проблемы с v1 engine на WSL2. Возможные решения:
- Использовать `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- Откатиться на vLLM v0.6.x
- Использовать нативный Linux

Подробнее: [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/)

### 4. llama.cpp

Для работы llama.cpp необходимо склонировать репозиторий и скомпилировать его (особенно для поддержки CUDA).

```bash
# Клонирование llama.cpp внутрь проекта
git clone https://github.com/ggml-org/llama.cpp

cd llama.cpp

# Очистка старой сборки (если есть)
rm -rf build

# Конфигурация с CUDA
cmake -B build -DGGML_CUDA=ON

# Сборка (используем все ядра CPU)
cmake --build build --config Release -j$(nproc)

# Проверка
./build/bin/llama-server --version
```

### 5. Зависимости для бенчмарка

```bash
# Возвращаемся в корень проекта
cd ..

source venv-vllm/bin/activate
pip install httpx
```

## 📥 Скачивание моделей

### vLLM (HuggingFace формат)

```bash
source venv-vllm/bin/activate

# Qwen3-8B-AWQ (~5GB) - 4-bit квантизация
huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir models/hf/Qwen3-8B-AWQ

# Qwen3-32B-FP8 (~35GB) - 8-bit квантизация
huggingface-cli download Qwen/Qwen3-32B-FP8 --local-dir models/hf/Qwen3-32B-FP8

# Альтернативные модели
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/hf/Llama-3.1-8B
huggingface-cli download Qwen/Qwen3-14B --local-dir models/hf/Qwen3-14B
```

### llama.cpp (GGUF формат)

```bash
# Qwen3-8B Q4_K_M (~5GB) - 4-bit
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models/gguf

# Qwen3-8B Q8_0 (~9GB) - 8-bit (лучшее качество)
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf --local-dir models/gguf

# Qwen3-32B Q8_0 (~35GB) - 8-bit
huggingface-cli download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q8_0.gguf --local-dir models/gguf

# Qwen3-32B Q4_K_M (~20GB) - 4-bit (экономия памяти)
huggingface-cli download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf --local-dir models/gguf
```

**Репозитории моделей:**
- [Qwen3 8B GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [Qwen3 32B GGUF](https://huggingface.co/Qwen/Qwen3-32B-GGUF)
- [Qwen3 Collection](https://huggingface.co/collections/Qwen/qwen3)

## 🎮 Использование

### vLLM Server

```bash
source venv-vllm/bin/activate

# Базовый запуск (модель полностью в GPU)
vllm serve models/hf/Qwen3-8B-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.90

# С CPU offloading (для больших моделей)
vllm serve models/hf/Qwen3-32B-FP8 \
    --port 8000 \
    --cpu-offload-gb 20 \
    --gpu-memory-utilization 0.95

# Из HuggingFace напрямую
vllm serve Qwen/Qwen3-8B-AWQ --port 8000
```

**Параметры:**
- `--gpu-memory-utilization` - процент использования VRAM (0.7-0.95)
- `--cpu-offload-gb` - сколько GB выгружать в RAM
- `--max-model-len` - максимальная длина контекста
- `--tensor-parallel-size` - количество GPU для tensor parallelism

**Документация**: [vLLM Serving](https://docs.vllm.ai/en/stable/serving/distributed_serving/)

### llama.cpp Server

```bash
# Все слои в GPU
./llama.cpp/build/bin/llama-server \
    --model models/gguf/Qwen3-8B-Q4_K_M.gguf \
    --n-gpu-layers -1 \
    --ctx-size 4096 \
    --port 8001 \
    --host 0.0.0.0

# Частичный GPU offload (30 слоёв из 64)
./llama.cpp/build/bin/llama-server \
    --model models/gguf/Qwen3-32B-Q8_0.gguf \
    --n-gpu-layers 30 \
    --ctx-size 4096 \
    --port 8001

# С параллельными запросами
./llama.cpp/build/bin/llama-server \
    --model models/gguf/Qwen3-8B-Q8_0.gguf \
    --n-gpu-layers -1 \
    --parallel 4 \
    --port 8001
```

**Параметры:**
- `--n-gpu-layers` - количество слоёв в GPU (-1 = все)
- `--ctx-size` - размер контекста
- `--parallel` - количество параллельных запросов
- `--threads` - CPU потоки для не-GPU операций

**Документация**: [llama.cpp Server](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)

## 🧪 Тестирование API

```bash
# vLLM (порт 8000)
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# llama.cpp (порт 8001)
curl http://localhost:8001/v1/models

curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## 📊 Бенчмаркинг

Подробная документация в [benchmark/README.md](benchmark/README.md)

```bash
source venv-vllm/bin/activate
# (опционально) cd в корень проекта если вы не там

# Запустить серверы в отдельных терминалах
# Терминал 1: vLLM на порту 8000
# Терминал 2: llama.cpp на порту 8001

# Тест одного бэкенда
python benchmark/benchmark.py \
    --backend vllm \
    --scenario "qwen3-8b-gpu" \
    --prompt short

# Сравнение обоих (если запущены одновременно не получается из-за памяти)
# Запускаем по отдельности и сравниваем результаты
python benchmark/compare_results.py \
    benchmark/results/benchmark_vllm_*.json \
    benchmark/results/benchmark_llamacpp_*.json
```

**Метрики:**
- **TTFT** (Time to First Token) - время до первого токена
- **TPS** (Tokens Per Second) - скорость генерации
- **ITL** (Inter-Token Latency) - задержка между токенами
- **E2E Latency** - полное время генерации

## 🔍 Сравнение подходов

| Характеристика | vLLM | llama.cpp |
|----------------|------|-----------|
| **Формат** | HuggingFace (safetensors) | GGUF |
| **Установка** | Python package | Компиляция из исходников |
| **CPU Offload** | `--cpu-offload-gb` | `--n-gpu-layers` |
| **Механизм** | Стриминг весов CPU→GPU | Послойное распределение |
| **Batch throughput** | ⭐⭐⭐⭐⭐ Отлично | ⭐⭐⭐ Хорошо |
| **Single request** | ⭐⭐⭐⭐ Хорошо | ⭐⭐⭐⭐⭐ Отлично |
| **Квантизация** | AWQ, GPTQ, FP8 | Q2-Q8, FP16 |
| **Memory efficient** | ⭐⭐⭐ Средне | ⭐⭐⭐⭐⭐ Отлично |
| **WSL2 совместимость** | ⚠️ Проблемы в v0.13.0 | ✅ Отлично |

### Когда использовать vLLM

- Высокая пропускная способность (batch inference)
- Production deployment с API
- Модели в HuggingFace формате
- Tensor parallelism на нескольких GPU

### Когда использовать llama.cpp

- CPU/GPU гибридный режим
- Экономия памяти через квантизацию
- Single request latency критична
- WSL2 / Windows окружение
- Модели в GGUF формате

## 🔗 Полезные ссылки

### Документация

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Installation Guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [vLLM Serving](https://docs.vllm.ai/en/stable/serving/distributed_serving/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama.cpp Server Guide](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)

### Модели

- [Qwen3 Models](https://huggingface.co/collections/Qwen/qwen3)
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [HuggingFace Model Hub](https://huggingface.co/models)

### Квантизация

- [GGUF Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md)
- [AWQ Quantization](https://github.com/mit-han-lab/llm-awq)
- [Qwen GGUF Documentation](https://qwen.readthedocs.io/en/latest/quantization/gguf.html)

### Бенчмарки

- [vLLM Performance](https://blog.vllm.ai/2023/06/20/vllm.html)
- [LLM Benchmarking Guide](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [Anyscale Benchmarking](https://docs.anyscale.com/llm/serving/benchmarking/metrics)

### Troubleshooting

- [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/)
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [llama.cpp Discussions](https://github.com/ggml-org/llama.cpp/discussions)
- [CUDA on WSL Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

## 💡 Примеры использования

### Python API (vLLM)

```python
from vllm import LLM, SamplingParams

# Загрузка модели
llm = LLM(
    model="models/hf/Qwen3-8B-AWQ",
    gpu_memory_utilization=0.9
)

# Генерация
prompts = ["Explain quantum computing in simple terms."]
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=200
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### Python API (llama.cpp)

```python
from llama_cpp import Llama

# Загрузка модели
llm = Llama(
    model_path="models/gguf/Qwen3-8B-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096
)

# Генерация
response = llm(
    "Explain quantum computing in simple terms.",
    max_tokens=200,
    temperature=0.7
)

print(response["choices"][0]["text"])
```

### OpenAI Compatible Client

```python
from openai import OpenAI

# vLLM
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# llama.cpp
# client = OpenAI(
#     base_url="http://localhost:8001/v1",
#     api_key="dummy"
# )

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## 🛠️ Troubleshooting

### vLLM на WSL2 не запускается

**Проблема**: `RuntimeError: Engine core initialization failed`

**Решение**:
```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
vllm serve models/hf/Qwen3-8B-AWQ --port 8000
```

Или установите более старую версию:
```bash
pip install vllm==0.6.3
```

### Out of Memory (OOM) ошибки

**vLLM**:
```bash
# Уменьшите gpu-memory-utilization
vllm serve model --gpu-memory-utilization 0.7

# Используйте CPU offload
vllm serve model --cpu-offload-gb 10

# Уменьшите context length
vllm serve model --max-model-len 2048
```

**llama.cpp**:
```bash
# Уменьшите количество GPU слоёв
llama-server --model model.gguf --n-gpu-layers 20

# Используйте более агрессивную квантизацию (Q4 вместо Q8)
```

### Медленная генерация

1. Проверьте что модель в GPU:
```bash
nvidia-smi  # Должна показывать использование
```

2. Для llama.cpp убедитесь что CUDA работает:
```bash
./llama-server --version  # Должно показывать CUDA
```

3. Уменьшите batch size / parallel requests

## 📝 TODO

- [ ] Добавить поддержку Ollama
- [ ] Автоматический выбор оптимальных параметров
- [ ] WebUI для управления моделями
- [ ] Интеграция с LangChain
- [ ] Docker контейнеры для vLLM и llama.cpp
- [ ] Сравнение с другими бэкендами (TensorRT-LLM, ExLlamaV2)

## 📄 Лицензия

Этот проект создан в образовательных целях. Модели имеют свои лицензии:
- Qwen3: Apache 2.0
- Llama 3: Meta License

---

**Создано**: 2026-01-09
**Система**: Ubuntu 24.04 (WSL2), RTX 4080 SUPER, CUDA 12.4

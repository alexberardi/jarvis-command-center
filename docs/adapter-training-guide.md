# Adapter Training & Benchmarking Guide

Quick reference for training LoRA adapters and running E2E command parsing benchmarks.

## Prerequisites

**Services required:**
- `jarvis-command-center` (port 7703) — `bash run-docker-dev.sh`
- `jarvis-llm-proxy-api` (port 7704 API, 7705 worker) — `./run.sh`

**Verify:**
```bash
curl -s http://localhost:7703/health | python3 -m json.tool
curl -s http://localhost:7705/health | python3 -m json.tool  # shows loaded model
```

## Available Models

| Model | GGUF Path | HuggingFace ID | Interface |
|-------|-----------|----------------|-----------|
| Gemma 2 9B Instruct | `.models/gemma-2-9b-it.Q4_K_M.gguf` | `google/gemma-2-9b-it` | `Gemma2MediumUntrained` |
| Qwen 2.5 7B Instruct | `.models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | `Qwen/Qwen2.5-7B-Instruct` | `Qwen25MediumUntrained` |
| Hermes 3 Llama 3.1 8B | `.models/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf` | `NousResearch/Hermes-3-Llama-3.1-8B` | `HermesMediumUntrained` |
| Llama 3.1 8B Instruct | `.models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf` | `meta-llama/Llama-3.1-8B-Instruct` | `Llama31MediumUntrained` |

## Step 1: Train an Adapter

From `jarvis-node-setup/`:

```bash
python scripts/train_node_adapter.py \
  --base-model-id .models/<GGUF_FILE> \
  --hf-base-model-id <HF_ORG>/<HF_MODEL>
```

**Examples:**
```bash
# Gemma 2
python scripts/train_node_adapter.py \
  --base-model-id .models/gemma-2-9b-it.Q4_K_M.gguf \
  --hf-base-model-id google/gemma-2-9b-it

# Qwen 2.5
python scripts/train_node_adapter.py \
  --base-model-id .models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --hf-base-model-id Qwen/Qwen2.5-7B-Instruct

# Hermes 3
python scripts/train_node_adapter.py \
  --base-model-id .models/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf \
  --hf-base-model-id NousResearch/Hermes-3-Llama-3.1-8B

# Llama 3.1
python scripts/train_node_adapter.py \
  --base-model-id .models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf \
  --hf-base-model-id meta-llama/Llama-3.1-8B-Instruct
```

### Monitor Training

```bash
curl -s http://localhost:7704/v1/training/status/<JOB_ID> | python3 -m json.tool
```

Status progression: `QUEUED` -> `RUNNING` -> `COMPLETE` (or `FAILED`)

The `artifact_path` field shows the adapter location when complete.

## Step 2: Swap Model in LLM Proxy

Load the desired GGUF in the llm-proxy (restart with the new model path).

Verify:
```bash
curl -s http://localhost:7705/health  # should show the new model
```

## Step 3: Set Command Center Interface

The command center must use the matching prompt provider. Set via `llm.interface` setting or `JARVIS_MODEL_INTERFACE` env var, then restart the container.

Verify from container logs:
```bash
docker logs jarvis-command-center-jarvis-voice-api-1 2>&1 | grep "PromptProviderFactory"
```

Should show: `PromptProviderFactory: found provider <InterfaceName>`

## Step 4: Run E2E Benchmarks

From `jarvis-node-setup/`:

```bash
# All tests
python test_command_parsing.py -o /path/to/jarvis-command-center/temp/test_results_<model>.json

# Specific tests
python test_command_parsing.py -t 5 7 11
python test_command_parsing.py -c get_weather set_timer

# List available tests
python test_command_parsing.py -l
```

### Extract Results

```bash
python3 -c "
import json
with open('temp/test_results_<model>.json') as f:
    data = json.load(f)
print(json.dumps(data['summary'], indent=2))
for t in data['test_results']:
    if not t['passed']:
        print(f\"  Test {t['test_number']}: {t['voice_command']} -> {t['actual']['command']} (expected {t['expected']['command']})\")
"
```

## Important Notes

### Dataset Hash Includes Model

The training dataset hash includes `base_model_id`, so each model gets a unique adapter. Training the same dataset on different models produces separate artifacts in `/tmp/jarvis-adapters/<hash>/`.

### Adapter Compatibility Guard

The GGUF backend checks `adapter_config.json` -> `base_model_name_or_path` against the loaded model before applying an adapter. If you swap models without retraining, the old adapter is silently skipped (not loaded).

### Untrained vs Trained Runs

- **Untrained**: Just swap model + interface, run benchmarks. No adapter involved.
- **Trained**: Train adapter first (Step 1), then swap model + interface, run benchmarks. The node's `adapter_hash` is updated automatically via the training callback.

### Typical Workflow for a New Model

1. Create prompt provider in `app/core/prompt_providers/medium/untrained/<model>.py`
2. Create tests in `tests/test_<model>_medium_untrained.py`
3. Run `pytest tests/test_<model>_medium_untrained.py`
4. Swap model in llm-proxy, set interface in command center
5. Run E2E: `python test_command_parsing.py -o temp/test_results_<model>.json` (untrained)
6. Train adapter: `python scripts/train_node_adapter.py --base-model-id ... --hf-base-model-id ...`
7. Run E2E again (trained)
8. Update README.md benchmark table

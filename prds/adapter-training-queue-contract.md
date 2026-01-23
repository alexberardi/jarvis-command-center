# Adapter Training Queue Contract (llm-proxy)

## Summary
This document describes how external services enqueue adapter training jobs into `llm-proxy` and how to receive results. Jobs are processed asynchronously via the existing Redis queue and return results through a callback URL.

## Primary Flow
1. Client calls `POST /internal/queue/enqueue` on llm-proxy with `job_type=adapter_train`.
2. llm-proxy enqueues the job to its internal Redis queue.
3. The llm-proxy queue worker runs the training command, packages the adapter, and invokes the callback URL.

## Endpoint
`POST /internal/queue/enqueue`

### Auth
- Requires Jarvis app-to-app auth (same as chat endpoints).
- Also requires `X-Internal-Token` when `LLM_PROXY_INTERNAL_TOKEN` is set.

### Request Body
```json
{
  "job_id": "uuid",
  "job_type": "adapter_train",
  "created_at": "ISO-8601",
  "priority": "normal|high",
  "trace_id": "uuid-or-string",
  "idempotency_key": "uuid-or-string",
  "job_type_version": "v1",
  "ttl_seconds": 86400,
  "metadata": {
    "anything": "optional"
  },
  "request": {
    "node_id": "node-123",
    "base_model_id": "llm-proxy-model-name",
    "dataset_ref": {
      "format": "inline-json",
      "data": {
        "commands": [
          {
            "command_name": "calculate",
            "examples": [
              {
                "voice_command": "add 5 and 7",
                "expected_tool_call": {
                  "name": "calculate",
                  "arguments": { "operation": "add", "a": 5, "b": 7 }
                }
              }
            ]
          }
        ]
      }
    },
    "dataset_hash": "optional-sha256",
    "params": {
      "rank": 16,
      "epochs": 2,
      "batch_size": 4,
      "max_seq_len": 2048
    }
  },
  "callback": {
    "url": "https://jcc.local/api/v0/adapters/jobs/callback",
    "auth_type": "bearer",
    "token": "service-to-service-token"
  }
}
```

### Field Requirements
- `job_id` (required): Unique job id for idempotency and correlation.
- `job_type` (required): Must be `adapter_train`.
- `created_at` (required): ISO-8601 timestamp.
- `idempotency_key` (required): Recommend using the same value as `job_id`.
- `ttl_seconds` (required): Job expiry window.
- `request.node_id` (required).
- `request.base_model_id` (required).
- `request.dataset_ref` (required): Inline JSON payload for the dataset (first iteration).
- `request.dataset_hash` (optional): If omitted, llm-proxy computes a hash from `dataset_ref`.
- `request.params` (optional): Training hyperparameters.
- `callback.url` (required): Callback endpoint to receive results.

### Response
```json
{
  "accepted": true,
  "job_id": "uuid",
  "deduped": false
}
```

## Callback Contract
The llm-proxy worker posts to the callback URL once complete (success or failure).

### Success Payload
```json
{
  "job_id": "uuid",
  "job_type": "adapter_train",
  "finished_at": "ISO-8601",
  "status": "succeeded",
  "result": {
    "artifact_url": "file:///tmp/jarvis-adapters/store/node-123/model-x/hash/adapter.zip",
    "artifact_metadata": {
      "node_id": "node-123",
      "base_model_id": "llm-proxy-model-name",
      "dataset_hash": "sha256",
      "adapter_format": "peft_lora",
      "artifact_path": "/tmp/jarvis-adapters/store/node-123/model-x/hash/adapter.zip",
      "artifact_size_bytes": 123456,
      "train_duration_seconds": 842.12,
      "cached": false
    }
  },
  "timing": {
    "processing_ms": 842120
  }
}
```

### Failure Payload
```json
{
  "job_id": "uuid",
  "job_type": "adapter_train",
  "finished_at": "ISO-8601",
  "status": "failed",
  "error": {
    "code": "exception|expired",
    "message": "string",
    "traceback": "optional"
  },
  "timing": {
    "processing_ms": 1200
  }
}
```

## Idempotency
- Enqueue calls are deduped by `(job_id, idempotency_key)` for the TTL.
- If an adapter artifact already exists for the same `dataset_hash`, the worker returns a cached result without retraining.

## Where to Get the Result
- The **only** result delivery mechanism is the callback URL.
- The callback includes `artifact_url` and metadata. Download the adapter from that URL.
- If `JARVIS_ADAPTER_PUBLIC_URL_PREFIX` is set, `artifact_url` uses that prefix. Otherwise it is a `file://` path on the llm-proxy host.

## Notes
- Jobs are processed asynchronously by the existing Redis queue used by llm-proxy.
- The training command is configured by `JARVIS_ADAPTER_TRAIN_CMD` on the llm-proxy worker.

# Jarvis llm-proxy HTTP Surface, Models, and App-to-App Auth

This PRD documents the current public HTTP surface of llm-proxy, request/response shapes, authentication, and model selection (including vision).

## Endpoints
- `POST /v1/chat/completions` — OpenAI-compatible chat (text + multimodal). **App-to-app auth required.**
- `GET /health` — liveness; no auth.
- `GET /health/live` — liveness; no auth.
- `GET /health/ready` — readiness; no auth.

Legacy routes (`/api/v*/chat`, warmup/model swap/reset/status, etc.) are removed.

## Authentication (app-to-app)
All protected endpoints (including `/v1/chat/completions`) require Jarvis app headers, forwarded to jarvis-auth `/internal/app-ping`:
```
X-Jarvis-App-Id: <your-app-id>
X-Jarvis-App-Key: <your-app-key>
```
Health endpoints remain unauthenticated.
Auth base URL: `JARVIS_AUTH_BASE_URL` (e.g., `http://.../`), loaded at request time.
On missing/invalid headers: 401. If auth service unreachable: 502.

## Request Format (/v1/chat/completions)
- `model`: alias or concrete id. Aliases:
  - `full` → primary text model
  - `lightweight` → smaller text model
  - `vision` → primary vision model
- `messages`: list of objects `{role, content}` where `content` may be:
  - string, or
  - array of parts:
    - `{ "type": "text", "text": "..." }`
    - `{ "type": "image_url", "image_url": { "url": "data:<mime>;base64,<...>", "detail"?: "high"|"low" } }`
- `temperature` (float), `max_tokens` (int, optional), `stream` (bool).

Notes:
- Images must be data URLs (base64). HTTP(S) fetch is not supported.
- Vision backends may enforce limits (e.g., some models reject multi-image).

## Response Format
OpenAI-style:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "<resolved-model-id>",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z }
}
```
Streaming uses `chat.completion.chunk` SSE frames.

## Error Behavior
- Images sent to text-only model → `invalid_request_error` with capability message.
- Malformed data URL / decode failure → `invalid_request_error`.
- Auth failure → 401 `Missing`/`Invalid app credentials`.
- Upstream/auth unreachable → 502 `Auth service unavailable...`.
- Other backend errors → `server_error`.

## Models & Backends
Resolution is centralized in `ModelManager`; handlers do not reach into configs directly.

Env-driven backend selection:
- `JARVIS_MODEL_BACKEND` — primary text (`full`).
- `JARVIS_LIGHTWEIGHT_MODEL_BACKEND` — `lightweight`.
- `JARVIS_VISION_MODEL_BACKEND` — `vision` (supports `MLX-VISION`, `TRANSFORMERS_VISION`, etc.).

Common backends:
- Text: GGUF, MLX, TRANSFORMERS, VLLM, REST, MOCK.
- Vision:
  - MLX-VISION (mlx-vlm; single-image for mllama; multi-image limited).
  - TRANSFORMERS_VISION (e.g., Qwen2/Qwen3 VL, SmolVLM2). Supports multi-image when the model does.
  - MOCK (tests).
  - GGUF vision: not supported (text-only GGUF client).

Model examples:
- `full` → text model (e.g., jarvis-text-8b)
- `lightweight` → smaller text (e.g., jarvis-text-1b)
- `vision` → vision model (e.g., Qwen/Qwen2-VL-2B-Instruct with TRANSFORMERS_VISION, or Llama-3.2 Vision with MLX-VISION)

## Environment Variables (key ones)
```
# Auth
JARVIS_AUTH_BASE_URL=...              # required for app auth forwarding
JARVIS_APP_ID=...                     # your app id (outbound, if needed)
JARVIS_APP_KEY=...                    # your app key (outbound, if needed)

# Backends/models
JARVIS_MODEL_BACKEND=...
JARVIS_LIGHTWEIGHT_MODEL_BACKEND=...
JARVIS_VISION_MODEL_BACKEND=...
JARVIS_MODEL_NAME=...
JARVIS_LIGHTWEIGHT_MODEL_NAME=...
JARVIS_VISION_MODEL_NAME=...

# Vision (transformers)
JARVIS_VISION_TORCH_DTYPE=float16|bfloat16|float32 (default: fp16 on MPS/CUDA, else fp32)
JARVIS_VISION_ATTN_IMPL=flash_attention_2 (optional; use on CUDA, not MPS)
```

## Usage Examples
### Text
```bash
curl -X POST https://llm-proxy/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Jarvis-App-Id: $APP_ID" \
  -H "X-Jarvis-App-Key: $APP_KEY" \
  -d '{
    "model": "full",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```
### Vision (single or multi-image, model permitting)
```bash
IMG1=$(base64 -i /path/to/img1.png | tr -d '\n')
IMG2=$(base64 -i /path/to/img2.png | tr -d '\n')  # optional second image
curl -X POST https://llm-proxy/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Jarvis-App-Id: $APP_ID" \
  -H "X-Jarvis-App-Key: $APP_KEY" \
  -d "{
    \"model\": \"vision\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe the images.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${IMG1}\"}},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${IMG2}\"}}
      ]
    }],
    \"temperature\": 0.2,
    \"max_tokens\": 200
  }"
```
Note: Some vision models (e.g., mllama via MLX-VISION) may only support one image; TRANSFORMERS_VISION with Qwen2/3 VL supports multi-image.

## Testing/Validation
- Health: `GET /health` (no headers).
- Auth connectivity: `curl -H "X-Jarvis-App-Id:..." -H "X-Jarvis-App-Key:..." $JARVIS_AUTH_BASE_URL/internal/app-ping`.
- Chat: send sample requests as above with required headers.


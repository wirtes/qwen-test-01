# Qwen Lightweight Container

A minimal Docker container setup for running Qwen 0.5B Instruct model with a simple REST API.

## Quick Start

1. **Build and run the container:**
   ```bash
   docker compose up --build
   ```

2. **Test the API:**
   ```bash
   python test_api.py
   ```

## API Endpoints

- `GET /health` - Check if the service is running and model is loaded
- `POST /generate` - Generate text from a prompt

### API Parameters

The `/generate` endpoint accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | The input text to generate from |
| `max_length` | integer | 50 | Maximum total tokens (prompt + response) |
| `max_new_tokens` | integer | 30 | Maximum new tokens to generate (response only) |
| `temperature` | float | 0.7 | Controls randomness (0.1-2.0, lower = more focused) |

### Example Usage

**Basic generation:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is artificial intelligence?"}'
```

**Short response (10 new tokens):**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_new_tokens": 10}'
```

**Focused response (low temperature):**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain Python", "max_new_tokens": 20, "temperature": 0.3}'
```

**Creative response (high temperature):**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a story", "max_new_tokens": 50, "temperature": 1.2}'
```

## Configuration

- **Model**: Qwen/Qwen2-0.5B-Instruct (ultra-lightweight 0.5B parameter model)
- **Port**: 8000
- **Cache**: Models are cached in `./cache` directory

## Requirements

- Docker with Compose plugin
- At least 2GB RAM recommended
- GPU support optional (will use CPU if no GPU available)

## Notes

- First run will download the model (~1GB)
- The container uses CPU by default for maximum compatibility
- Model responses are cached locally to speed up subsequent runs
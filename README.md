# Qwen Lightweight Container

A minimal Docker container setup for running Qwen 1.8B Chat model with a simple REST API.

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

### Example Usage

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is artificial intelligence?", "max_length": 100}'
```

## Configuration

- **Model**: Qwen/Qwen-1_8B-Chat (lightweight 1.8B parameter model)
- **Port**: 8000
- **Cache**: Models are cached in `./cache` directory

## Requirements

- Docker with Compose plugin
- At least 4GB RAM recommended
- GPU support optional (will use CPU if no GPU available)

## Notes

- First run will download the model (~3.6GB)
- The container uses CPU by default for maximum compatibility
- Model responses are cached locally to speed up subsequent runs
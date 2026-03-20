# Setup Guide

## Hardware Requirements

**Your i7 12th Gen, 16GB RAM:**

| Component | Works? | Speed | RAM Usage |
|-----------|--------|-------|-----------|
| pdfplumber | ✅ | Instant | Minimal |
| Surya OCR | ✅ | 8–15s/page on CPU | ~3GB |
| Ollama + phi3:mini | ✅ | 3–6 tok/sec | ~2.3GB |
| Ollama + mistral:7b | ✅ | 1–3 tok/sec | ~5GB |

## Step 1: System Dependencies

### Windows (PowerShell)

```powershell
# Python 3.11+ required
python --version

# Install Poppler for pdf2image
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# Extract to C:\poppler and add C:\poppler\Library\bin to PATH
```

### Windows (WSL2)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv poppler-utils redis-server
sudo systemctl start redis
```

## Step 2: Python Environment

```bash
cd pharma_parser
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/WSL
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Environment Config

```bash
cp .env.example .env
# Edit .env and set your values
```

## Step 4: Database

```bash
python manage.py makemigrations core
python manage.py migrate
python manage.py createsuperuser  # optional
```

## Step 5: Local LLM (Ollama)

```bash
# Install Ollama: https://ollama.ai/download
# Then pull a model:
ollama pull phi3:mini     # Recommended: fast, 2.3GB

# Start Ollama server:
ollama serve
```

Set in `.env`:
```
USE_LOCAL_LLM=true
OLLAMA_MODEL=phi3:mini
```

## Step 6: Run

```bash
# Terminal 1 — Django
python manage.py runserver 8000

# Terminal 2 — Celery (requires Redis)
celery -A pharma_parser worker --loglevel=info --concurrency=2

# Terminal 3 — Ollama
ollama serve
```

## Step 7: Test

```bash
curl -X POST http://localhost:8000/api/process/ -F "pdf=@path/to/invoice.pdf"
curl http://localhost:8000/api/results/1/
```

## NVIDIA GPU Check

```bash
nvidia-smi
# If found → set SURYA_DEVICE=cuda in .env for 10x faster OCR
# If "not found" → use CPU mode (default), everything still works
```

## Common Issues

| Error | Fix |
|-------|-----|
| `poppler not found` | Install Poppler and add to PATH |
| `redis connection refused` | Start Redis: `redis-server` or `sudo systemctl start redis` |
| `torch error` for Surya | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `celery not found` | Activate venv first |

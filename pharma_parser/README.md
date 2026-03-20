# Pharma Invoice Parser

Production-grade pipeline for extracting structured data from Indian pharmaceutical tax invoices. Handles 100+ invoices/day across 11+ vendor formats.

## Architecture

```
PDF Input → Detector → Digital/Scanned Router
    ├── Digital  → pdfplumber + Vendor Parser ──→ Validator
    └── Scanned  → Surya OCR → Table Rebuild ──→ Validator
                                                     │
                                    LOW confidence → React Agent → Re-validate
                                                     │
                                                     ↓
                                              Excel + JSON Output
```

## Quick Start

```bash
cd pharma_parser
cp .env.example .env          # fill in your values
pip install -r requirements.txt

# Run migrations
python manage.py makemigrations core
python manage.py migrate

# Start Django (Terminal 1)
python manage.py runserver 8000

# Start Celery (Terminal 2, requires Redis)
celery -A pharma_parser worker --loglevel=info

# Start Ollama (Terminal 3, for local LLM)
ollama serve
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process/` | Upload PDF, returns `job_id` |
| `GET` | `/api/results/<id>/` | Get job status + extracted data |
| `GET` | `/api/jobs/` | List all jobs |
| `GET` | `/api/review-queue/` | LOW-confidence items for review |

### Example

```bash
# Upload invoice
curl -X POST http://localhost:8000/api/process/ -F "pdf=@invoice.pdf"
# → {"job_id": 1, "status": "pending"}

# Check result
curl http://localhost:8000/api/results/1/
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | Django 4.2 + DRF |
| Task Queue | Celery + Redis |
| PDF Text | pdfplumber |
| OCR | Surya OCR (local, open-source) |
| Error Recovery | LangChain ReAct Agent |
| LLM | Ollama (local) or OpenRouter |
| Output | openpyxl (Excel) + JSON |
| Rules | CSV files (corrections, HSN codes) |

## Supported Vendors

| Vendor | GSTIN | PDF Type |
|--------|-------|----------|
| Dr. Reddy's | 07AAACD7999Q1ZM | Digital |
| FDC Limited | 07AAACF0253H1Z2 | Digital |
| Maviga Logisys | 07AARCM6852G1ZZ | Digital |
| MSN Laboratories | 07AADCM6283F1ZE | Digital |
| Nidhi Associates | 07AAACN3519N1ZA | Digital |
| Mardia Pharma | 07AAAHR3612M1Z4 | Digital |
| Pangea Enterprises | 06AAFCP6328J1Z8 | Digital |
| Corona Remedies | 07AACCC5173F1ZU | Digital |
| Mankind Pharma | 07AAACM9401C1ZX | Digital |
| Eris Healthcare | 07AATCS3717L1ZQ | Mixed |
| AbbVie | — | Scanned |

## Project Structure

```
pharma_parser/
├── pharma_parser/        # Django project config
├── core/                 # Django app (models, views, tasks)
├── pipeline/             # Processing logic (no Django dependency)
│   ├── detector.py       # Digital vs scanned detection
│   ├── digital_extractor.py
│   ├── ocr_extractor.py  # Surya OCR
│   ├── validators.py     # GST math validation
│   └── agent.py          # LangChain React Agent
├── parsers/              # One file per vendor
├── rules/                # CSV-driven rules (no code change needed)
└── outputs/              # Excel + JSON writers
```

## Adding a New Vendor

1. Create `parsers/<vendor_id>.py` with `class Parser(GenericParser)`
2. Add GSTIN to `pipeline/vendor_router.py` → `VENDOR_SIGNATURES`
3. Add product corrections to `rules/product_corrections.csv`

## Environment Variables

See [.env.example](.env.example) for all available configuration.

## Documentation

- [SETUP.md](SETUP.md) — Step-by-step installation guide
- [DEV_DOC.md](DEV_DOC.md) — Architecture deep-dive for developers

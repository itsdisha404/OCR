# Developer Documentation

## Pipeline Flow

```
process_invoice_task(job_id, pdf_path)
    │
    ├── Stage 1: detect_pdf_type(path) → DetectionResult
    │   Returns: pdf_type (DIGITAL/SCANNED/MIXED), page lists
    │
    ├── Stage 2: Extract
    │   ├── DIGITAL → extract_digital(path, pages)
    │   │   Uses pdfplumber → detect_vendor() → get_parser().parse()
    │   │
    │   ├── SCANNED → extract_with_surya(path, pages)
    │   │   pdf2image → Surya OCR → reconstruct_tables() → parse()
    │   │
    │   └── MIXED → both paths, merge results
    │
    ├── Stage 3: validate_invoice(invoice)
    │   Checks: GSTIN format, GST math, HSN codes, expiry, grand total
    │   Sets: per-item flags, overall_confidence
    │
    ├── Stage 4: run_error_agent(invoice) [if LOW confidence]
    │   LangChain ReAct → tools: lookup_hsn, fix_gst, correct_name, etc.
    │   Re-validates after fixes
    │
    └── Stage 5: Output
        save_json() + append_to_excel()
```

## Data Model

Every invoice passes through the pipeline as a Python dict:

```python
{
    "header": {          # from field_spec.HEADER_FIELDS
        "invoice_no": "INV-001",
        "vendor_gstin": "07AAACD7999Q1ZM",
        "vendor_id": "dr_reddys",
        ...
    },
    "line_items": [      # list of field_spec.LINE_ITEM_FIELDS dicts
        {
            "product_name": "TELMISARTAN",
            "hsn_code": "30049099",
            "cgst_rate": 2.5,
            "confidence": "HIGH",
            "flags": [],
            ...
        }
    ],
    "summary": { ... },                  # from field_spec.SUMMARY_FIELDS
    "validation_summary": {              # added by validators.py
        "overall_confidence": "HIGH",
        "header_flags": [],
        "item_errors": [],
    },
}
```

## Vendor Parser Contract

Every parser must extend `GenericParser` and implement:

```python
class Parser(GenericParser):
    VENDOR_NAME = "My Vendor"

    def parse(self, full_text: str, tables: list, source_file: str) -> dict:
        # Returns: {"header": {...}, "line_items": [...], "summary": {...}}
```

Arguments:
- `full_text`: All pages concatenated with `\n`.
- `tables`: `list[list[table]]` where each `table` is `list[list[str|None]]` from pdfplumber.
- `source_file`: Filename string for audit trail.

## Rules Layer

CSVs in `rules/` can be edited without code changes:

| File | Purpose | Columns |
|------|---------|---------|
| `product_corrections.csv` | Fix OCR typos | `bad_name,good_name` |
| `valid_hsn_codes.csv` | HSN validation | `hsn_code,description` |
| `valid_products.csv` | Known products | `product_name,hsn_code` |
| `vendor_patterns.csv` | Vendor detection | `vendor_id,gstin,name_pattern` |
| `gst_rate_rules.csv` | GST rate lookup | `hsn_code,gst_rate,cgst_rate,sgst_rate` |

## React Agent

Triggered when `overall_confidence == "LOW"`. Uses:
- **OpenRouter** (cloud, default) or **Ollama** (local) based on `.env`
- 5 tools: `lookup_hsn_code`, `fix_gst_math`, `validate_gstin`, `correct_product_name`, `flag_for_human_review`
- Max 8 iterations to prevent infinite loops

## Performance

| Operation | Time (CPU) | RAM |
|-----------|-----------|-----|
| pdfplumber extraction | 1–3s | ~50MB |
| Surya OCR per page | 8–15s | ~3GB |
| Validation | <0.1s | ~10MB |
| LLM agent (Ollama phi3) | 5–20s | ~2.3GB |
| Excel write | <0.5s | ~30MB |

## Adding New Vendor Checklist

1. [ ] Create `parsers/<vendor_id>.py`
2. [ ] Add GSTIN to `pipeline/vendor_router.py`
3. [ ] Add to `rules/vendor_patterns.csv`
4. [ ] Add OCR corrections to `rules/product_corrections.csv`
5. [ ] Test with 3+ sample invoices
6. [ ] Verify GST math passes validation

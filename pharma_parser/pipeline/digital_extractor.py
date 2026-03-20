"""
digital_extractor.py
────────────────────
Stage 2a: Extract structured data from digital (text-layer) PDF pages
using pdfplumber, then route to the correct vendor parser.
"""

from pathlib import Path

import pdfplumber

from .vendor_router import detect_vendor, get_parser


def extract_digital(pdf_path, pages: list) -> dict:
    """
    Extract from digital PDF pages using pdfplumber.

    Args:
        pdf_path: Path to PDF file.
        pages: List of page indices (0-based) to extract from.

    Returns:
        Invoice dict with header, line_items, summary.
    """
    pdf_path = Path(pdf_path)
    raw_pages_text = []
    raw_pages_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i in pages:
            page = pdf.pages[i]
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            raw_pages_text.append(text)
            raw_pages_tables.append(tables)

    full_text = "\n".join(raw_pages_text)

    # Detect vendor from GSTIN or company name
    vendor_id = detect_vendor(full_text)

    # Route to vendor-specific parser
    parser = get_parser(vendor_id)
    result = parser.parse(full_text, raw_pages_tables, str(pdf_path))
    result.setdefault("header", {})
    result["header"]["extraction_method"] = "digital"
    result["header"]["vendor_id"] = vendor_id
    result["header"]["source_file"] = pdf_path.name
    return result

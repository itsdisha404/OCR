"""
detector.py
───────────
Stage 1: Detect whether a PDF is digital (text-layer), scanned (image-only),
or mixed. Uses pdfplumber character count per page as the heuristic.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pdfplumber


class PDFType(Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    MIXED = "mixed"


@dataclass
class DetectionResult:
    pdf_type: PDFType
    total_pages: int
    digital_pages: list = field(default_factory=list)
    scanned_pages: list = field(default_factory=list)
    chars_per_page: list = field(default_factory=list)


MIN_CHARS = int(os.getenv("MIN_TEXT_CHARS_PER_PAGE", 100))


def detect_pdf_type(pdf_path) -> DetectionResult:
    """
    Opens PDF with pdfplumber and checks extractable characters per page.
    Pages with fewer than MIN_CHARS characters are classified as scanned.
    """
    pdf_path = Path(pdf_path)
    digital_pages = []
    scanned_pages = []
    chars_per_page = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Also count table cell text
            tables = page.extract_tables() or []
            table_text = " ".join(
                str(cell)
                for table in tables
                for row in table
                for cell in (row or [])
                if cell
            )
            total_chars = len(text.strip()) + len(table_text.strip())
            chars_per_page.append(total_chars)

            if total_chars >= MIN_CHARS:
                digital_pages.append(i)
            else:
                scanned_pages.append(i)

    if not scanned_pages:
        pdf_type = PDFType.DIGITAL
    elif not digital_pages:
        pdf_type = PDFType.SCANNED
    else:
        pdf_type = PDFType.MIXED

    return DetectionResult(
        pdf_type=pdf_type,
        total_pages=total,
        digital_pages=digital_pages,
        scanned_pages=scanned_pages,
        chars_per_page=chars_per_page,
    )

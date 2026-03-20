"""
ocr_extractor.py
────────────────
Stage 2b: Extract text from scanned PDF pages using Surya OCR.
Converts pages to images at 300 DPI, runs Surya, then reconstructs
tables from bounding-box positions using Y-coordinate clustering.
"""

import os
from pathlib import Path


def extract_with_surya(pdf_path, pages: list) -> list:
    """
    Converts specified PDF pages to images at 300 DPI,
    runs Surya OCR, returns one dict per page with 'text' and 'raw_surya'.

    Args:
        pdf_path: Path to PDF file.
        pages: List of page indices (0-based) that need OCR.

    Returns:
        List of dicts: [{"text": str, "raw_surya": OCRResult}, ...]
    """
    from pdf2image import convert_from_path

    pdf_path = Path(pdf_path)
    dpi = 300

    # Convert only the scanned pages (first_page/last_page are 1-indexed)
    all_images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=min(pages) + 1,
        last_page=max(pages) + 1,
    )

    # Lazy-load Surya models (they load into memory ~3GB)
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.detection.processor import load_processor as load_det_proc
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_proc

    device = os.getenv("SURYA_DEVICE", "cpu")

    det_model = load_det_model(device=device)
    det_processor = load_det_proc()
    rec_model = load_rec_model(device=device)
    rec_processor = load_rec_proc()

    langs = [["en"]] * len(all_images)
    results = run_ocr(
        all_images,
        langs,
        det_model,
        det_processor,
        rec_model,
        rec_processor,
    )

    page_results = []
    for result in results:
        lines = [line.text for line in result.text_lines]
        page_results.append({
            "text": "\n".join(lines),
            "raw_surya": result,
        })

    return page_results


def reconstruct_tables_from_surya(surya_page_result: dict) -> list:
    """
    Reconstruct table rows from Surya OCR bounding boxes.
    Groups text lines into rows by Y-coordinate proximity,
    sorts cells within each row by X-coordinate (left to right).

    Args:
        surya_page_result: dict with 'raw_surya' key containing OCRResult.

    Returns:
        List of rows, each row is a list of cell text strings.
        Same format as pdfplumber tables for compatibility.
    """
    raw = surya_page_result.get("raw_surya")
    if raw is None:
        return []

    lines = raw.text_lines
    if not lines:
        return []

    def y_center(line):
        return (line.bbox[1] + line.bbox[3]) / 2

    def x_left(line):
        return line.bbox[0]

    sorted_lines = sorted(lines, key=y_center)

    # Cluster lines into rows by Y proximity (12px tolerance at 300 DPI)
    rows = []
    cur_row = [sorted_lines[0]]
    cur_y = y_center(sorted_lines[0])

    for line in sorted_lines[1:]:
        if abs(y_center(line) - cur_y) < 12:
            cur_row.append(line)
        else:
            rows.append(sorted(cur_row, key=x_left))
            cur_row = [line]
            cur_y = y_center(line)
    if cur_row:
        rows.append(sorted(cur_row, key=x_left))

    # Convert to list-of-lists (same format as pdfplumber tables)
    return [[cell.text for cell in row] for row in rows]

"""
json_writer.py
──────────────
Save individual invoice results as JSON files.
One file per invoice, named by invoice number and date.
"""

import json
from pathlib import Path


def save_json(invoice: dict, output_dir) -> Path:
    """
    Save a single invoice dict as a JSON file.

    Args:
        invoice: Full invoice dict with header, line_items, summary.
        output_dir: Directory to save JSON files.

    Returns:
        Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename from invoice_no + date
    h = invoice.get("header", {})
    invoice_no = h.get("invoice_no", "unknown").replace("/", "_").replace("\\", "_")
    invoice_date = h.get("invoice_date", "")
    vendor_id = h.get("vendor_id", "unknown")

    if invoice_date:
        fname = f"{vendor_id}_{invoice_no}_{invoice_date}.json"
    else:
        fname = f"{vendor_id}_{invoice_no}.json"

    # Clean filename
    fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)
    filepath = output_dir / fname

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(invoice, f, indent=2, ensure_ascii=False, default=str)

    return filepath

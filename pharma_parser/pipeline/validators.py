"""
validators.py
─────────────
Stage 3: Validate extracted invoice data. Checks GST math, HSN codes,
expiry dates, grand total cross-check. Sets confidence per item and overall.
"""

import re
from datetime import date
from pathlib import Path

import pandas as pd

from .field_spec import COMMON_HSN_CODES


# ── Load HSN codes from CSV if available ──────────────────────────────────────

_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


def _load_valid_hsn_codes() -> set:
    """Load valid HSN codes from CSV, falling back to built-in set."""
    path = _RULES_DIR / "valid_hsn_codes.csv"
    if not path.exists():
        return COMMON_HSN_CODES
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        codes = set(str(c).strip() for c in df.iloc[:, 0].tolist() if c)
        return codes | COMMON_HSN_CODES
    except Exception:
        return COMMON_HSN_CODES


VALID_HSN = _load_valid_hsn_codes()


# ── Validation functions ──────────────────────────────────────────────────────

def validate_invoice(invoice: dict) -> dict:
    """
    Run all validation checks on an invoice dict.
    Modifies items in-place: downgrades confidence, appends flags.
    Adds 'validation_summary' key to the invoice.

    Returns the same dict with validation results added.
    """
    item_errors = []
    header_flags = []

    # ── Header checks ─────────────────────────────────────────────────────
    h = invoice.get("header", {})
    if not h.get("invoice_no"):
        header_flags.append("MISSING_INVOICE_NO")
    if not h.get("invoice_date"):
        header_flags.append("MISSING_DATE")
    if not h.get("vendor_gstin"):
        header_flags.append("MISSING_VENDOR_GSTIN")
    if h.get("buyer_gstin") and not _valid_gstin(h["buyer_gstin"]):
        header_flags.append("INVALID_BUYER_GSTIN")

    # ── Line-item checks ──────────────────────────────────────────────────
    computed_taxable = 0.0
    computed_cgst = 0.0
    computed_sgst = 0.0

    for item in invoice.get("line_items", []):
        flags = list(item.get("flags", []))

        # HSN code validation
        hsn = str(item.get("hsn_code", "")).strip()
        if hsn and VALID_HSN and hsn not in VALID_HSN:
            flags.append("INVALID_HSN")

        # GST math check
        taxable = float(item.get("taxable_value", 0) or 0)
        cgst_rate = float(item.get("cgst_rate", 0) or 0)
        cgst_amt = float(item.get("cgst_amount", 0) or 0)
        sgst_rate = float(item.get("sgst_rate", 0) or 0)
        sgst_amt = float(item.get("sgst_amount", 0) or 0)
        total = float(item.get("total_amount", 0) or 0)

        if taxable > 0 and cgst_rate > 0:
            expected_cgst = round(taxable * cgst_rate / 100, 2)
            if abs(expected_cgst - cgst_amt) > 0.50:
                flags.append(f"CGST_MISMATCH:{expected_cgst:.2f}!={cgst_amt:.2f}")

        if taxable > 0 and sgst_rate > 0:
            expected_sgst = round(taxable * sgst_rate / 100, 2)
            if abs(expected_sgst - sgst_amt) > 0.50:
                flags.append("SGST_MISMATCH")

        expected_total = round(taxable + cgst_amt + sgst_amt, 2)
        if total > 0 and abs(expected_total - total) > 1.0:
            flags.append("TOTAL_MISMATCH")

        # Expiry date sanity
        exp = item.get("expiry_date", "")
        if exp and _is_expired(exp):
            flags.append("EXPIRED_BATCH")

        # Update confidence based on flags
        if flags and not item.get("flags"):
            item["confidence"] = "LOW" if len(flags) > 1 else "MEDIUM"
        elif flags:
            item["confidence"] = "LOW"
        item["flags"] = flags

        if flags:
            item_errors.append({
                "product": item.get("product_name", ""),
                "flags": flags,
            })

        computed_taxable += taxable
        computed_cgst += cgst_amt
        computed_sgst += sgst_amt

    # ── Summary check ─────────────────────────────────────────────────────
    s = invoice.get("summary", {})
    grand_total = float(s.get("grand_total", 0) or 0)
    expected_grand = round(computed_taxable + computed_cgst + computed_sgst, 2)
    summary_flags = []

    if grand_total > 0 and abs(expected_grand - grand_total) > 5.0:
        summary_flags.append(
            f"GRAND_TOTAL_MISMATCH:computed={expected_grand},invoice={grand_total}"
        )

    invoice["validation_summary"] = {
        "header_flags": header_flags,
        "item_errors": item_errors,
        "summary_flags": summary_flags,
        "total_flags": len(header_flags) + len(item_errors) + len(summary_flags),
        "overall_confidence": _overall_confidence(invoice),
    }
    return invoice


def _valid_gstin(gstin: str) -> bool:
    """Check if a GSTIN string matches the expected 15-char format."""
    return bool(re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]$", gstin or ""))


def _is_expired(exp_str: str) -> bool:
    """Check if an expiry date string represents an already-expired batch."""
    m = re.match(r"(\d{1,2})[./\-](\d{4})", exp_str)
    if not m:
        # Try YYYY-MM-DD format
        m2 = re.match(r"(\d{4})-(\d{1,2})", exp_str)
        if m2:
            year, month = int(m2.group(1)), int(m2.group(2))
            return date(year, month, 1) < date.today()
        return False
    month, year = int(m.group(1)), int(m.group(2))
    try:
        return date(year, month, 1) < date.today()
    except ValueError:
        return False


def _overall_confidence(invoice: dict) -> str:
    """Compute overall invoice confidence from per-item confidences and flags."""
    items = invoice.get("line_items", [])
    if not items:
        return "LOW"

    low_count = sum(1 for i in items if i.get("confidence") == "LOW")
    total_flags = sum(len(i.get("flags", [])) for i in items)
    header_flags = len(invoice.get("validation_summary", {}).get("header_flags", []))

    if low_count == 0 and total_flags == 0 and header_flags == 0:
        return "HIGH"
    elif low_count / len(items) > 0.3 or total_flags > 5:
        return "LOW"
    return "MEDIUM"

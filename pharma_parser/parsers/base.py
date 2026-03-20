"""
base.py
───────
Base parser with shared regex, CSV loaders, and generic fallback extraction.
All vendor parsers extend GenericParser.
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Rules directory ───────────────────────────────────────────────────────────

_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


# ── CSV loaders (run once at import time) ─────────────────────────────────────

def _load_csv_mapping(filename: str, key_col: str, val_col: str) -> dict:
    """Load a CSV file as a key→value dict."""
    path = _RULES_DIR / filename
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
        return dict(zip(df[key_col].astype(str).str.strip(), df[val_col].astype(str).str.strip()))
    except Exception:
        return {}


PRODUCT_CORRECTIONS = _load_csv_mapping("product_corrections.csv", "bad_name", "good_name")

_VALID_HSN_PATH = _RULES_DIR / "valid_hsn_codes.csv"
VALID_HSN = set()
if _VALID_HSN_PATH.exists():
    try:
        VALID_HSN = set(
            pd.read_csv(_VALID_HSN_PATH, encoding="utf-8-sig")
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .tolist()
        )
    except Exception:
        pass


# ── Shared regex patterns ─────────────────────────────────────────────────────

GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b")
DATE_RE = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b")
DATE_MMYYYY_RE = re.compile(r"\b(\d{1,2})[./\-](\d{4})\b")
IRN_RE = re.compile(r"[a-f0-9]{64}")
PO_RE = re.compile(r"P\.?O\.?\s*(?:No\.?|Number)?[\s:]*([A-Z0-9\-/]+)", re.I)
INVOICE_RE = re.compile(r"(?:Invoice|GST Invoice|Bill|Tax Invoice)[\s#No.:]*([A-Z0-9/\-]+)", re.I)
AMOUNT_RE = re.compile(r"[\d,]+\.\d{2}")
HSN_RE = re.compile(r"\b(\d{8})\b")
DL_RE = re.compile(r"(?:D\.?L\.?\s*No\.?|Drug\s*Lic)[\s.:]*([A-Z0-9\-/]+)", re.I)
FSSAI_RE = re.compile(r"(?:FSSAI)[\s.:]*(\d{14})", re.I)


# ── Utility functions ─────────────────────────────────────────────────────────

def normalise_date(d: str, m: str, y: str) -> str:
    """Convert any date components to YYYY-MM-DD."""
    y_int = int(y)
    if y_int < 100:
        y_int += 2000
    return f"{y_int:04d}-{int(m):02d}-{int(d):02d}"


def normalise_mmyyyy(m: str, y: str) -> str:
    """Convert MM/YYYY to YYYY-MM-01."""
    return f"{int(y):04d}-{int(m):02d}-01"


def parse_amount(s) -> float:
    """Parse a numeric string with commas to float."""
    if not s:
        return 0.0
    return float(str(s).replace(",", "").strip() or 0)


def apply_product_correction(name: str) -> tuple:
    """
    Check if name has a known correction.
    Returns (corrected_name, was_corrected).
    """
    key = name.upper().strip()
    corrected = PRODUCT_CORRECTIONS.get(key)
    if corrected:
        return corrected, True
    return name, False


def normalize_product_name(s: str) -> str:
    """Normalize drug name for dictionary lookups."""
    if not s:
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = s.split("(", 1)[0].strip()
    s = re.sub(r"\s*(TAB|TABS|CAP|CAPS|INJ|SYR|OINT|CREAM|GEL|DROP)S?\s*$", "", s)
    return s.strip()


# ── GenericParser (fallback for unknown vendors) ──────────────────────────────

class GenericParser:
    """
    Fallback parser for unknown vendors.
    Extracts what it can using generic regex and table heuristics.
    """

    def parse(self, full_text: str, tables: list, source_file: str) -> dict:
        """
        Parse invoice text and tables into structured dict.

        Args:
            full_text:   Entire PDF text joined by newlines.
            tables:      List of pages, each a list of tables,
                         each table a list[list[str | None]].
            source_file: Original filename for audit trail.

        Returns:
            Dict with keys: header, line_items, summary.
        """
        header = self._extract_header(full_text, source_file)
        line_items = self._extract_line_items_from_tables(tables)
        summary = self._extract_summary(full_text)
        return {"header": header, "line_items": line_items, "summary": summary}

    def _extract_header(self, text: str, source: str) -> dict:
        h = {
            "invoice_no": "",
            "invoice_date": "",
            "due_date": "",
            "po_number": "",
            "vendor_name": "",
            "vendor_gstin": "",
            "vendor_pan": "",
            "vendor_state": "",
            "vendor_address": "",
            "buyer_name": "",
            "buyer_gstin": "",
            "buyer_pan": "",
            "buyer_dl_no": "",
            "buyer_fssai": "",
            "buyer_address": "",
            "payment_terms": "",
            "eway_bill_no": "",
            "irn": "",
            "source_file": source,
            "vendor_id": "unknown",
            "extraction_method": "",
            "delivery_number": "",
        }

        m = INVOICE_RE.search(text)
        if m:
            h["invoice_no"] = m.group(1).strip()

        dates = DATE_RE.findall(text)
        if dates:
            h["invoice_date"] = normalise_date(*dates[0])

        gstins = GSTIN_RE.findall(text)
        if len(gstins) > 0:
            h["vendor_gstin"] = gstins[0]
        if len(gstins) > 1:
            h["buyer_gstin"] = gstins[1]

        m2 = PO_RE.search(text)
        if m2:
            h["po_number"] = m2.group(1).strip()

        irn_m = IRN_RE.search(text)
        if irn_m:
            h["irn"] = irn_m.group(0)

        dl_m = DL_RE.search(text)
        if dl_m:
            h["buyer_dl_no"] = dl_m.group(1).strip()

        fssai_m = FSSAI_RE.search(text)
        if fssai_m:
            h["buyer_fssai"] = fssai_m.group(1)

        return h

    def _extract_line_items_from_tables(self, tables: list) -> list:
        items = []
        for page_tables in tables:
            if not page_tables:
                continue
            for table in page_tables:
                if not table or len(table) < 2:
                    continue
                # Try to find header row and parse remaining rows
                header_row = None
                for i, row in enumerate(table):
                    row_str = " ".join(str(c or "").lower() for c in row)
                    if "hsn" in row_str or "product" in row_str or "description" in row_str:
                        header_row = i
                        break
                if header_row is not None:
                    col_map = self._map_columns(table[header_row])
                    for row in table[header_row + 1:]:
                        item = self._parse_generic_row(row, col_map)
                        if item:
                            items.append(item)
        return items

    def _map_columns(self, header_row: list) -> dict:
        """Map header cell text to column indices."""
        col_map = {}
        keywords = {
            "product": ["product", "description", "goods", "material", "item"],
            "hsn": ["hsn", "sac"],
            "batch": ["batch", "lot"],
            "expiry": ["exp", "expiry"],
            "qty": ["qty", "quantity", "sale"],
            "mrp": ["mrp"],
            "ptr": ["ptr", "p.t.r"],
            "pts": ["pts", "p.t.s"],
            "rate": ["rate"],
            "disc": ["disc", "discount"],
            "taxable": ["taxable"],
            "cgst": ["cgst"],
            "sgst": ["sgst"],
            "total": ["total", "amount", "net"],
        }
        for i, cell in enumerate(header_row):
            cell_lower = str(cell or "").lower().strip()
            for col_key, patterns in keywords.items():
                if any(p in cell_lower for p in patterns):
                    if col_key not in col_map:
                        col_map[col_key] = i
                    break
        return col_map

    def _parse_generic_row(self, row: list, col_map: dict) -> Optional[dict]:
        """Parse a single table row using column mapping."""
        cells = [str(c or "").strip() for c in row]

        def g(key):
            idx = col_map.get(key)
            return cells[idx] if idx is not None and idx < len(cells) else ""

        product = g("product")
        if not product or len(product) < 2:
            return None

        hsn = g("hsn")
        if not hsn:
            # Try to find 8-digit code in the row
            row_str = " ".join(cells)
            hsn_m = HSN_RE.search(row_str)
            hsn = hsn_m.group(1) if hsn_m else ""

        product_fixed, corrected = apply_product_correction(product)

        return {
            "sr_no": 0,
            "product_name": product_fixed,
            "product_raw": product,
            "hsn_code": hsn,
            "batch_no": g("batch"),
            "expiry_date": g("expiry"),
            "mfg_name": "",
            "pack_size": "",
            "category": "",
            "qty_billed": parse_amount(g("qty")),
            "qty_free": 0.0,
            "qty_total": parse_amount(g("qty")),
            "mrp": parse_amount(g("mrp")),
            "ptr": parse_amount(g("ptr")),
            "pts": parse_amount(g("pts")),
            "rate": parse_amount(g("rate")),
            "discount_pct": parse_amount(g("disc")),
            "discount_value": 0.0,
            "taxable_value": parse_amount(g("taxable")),
            "cgst_rate": parse_amount(g("cgst")),
            "cgst_amount": 0.0,
            "sgst_rate": parse_amount(g("sgst")),
            "sgst_amount": 0.0,
            "igst_rate": 0.0,
            "igst_amount": 0.0,
            "total_amount": parse_amount(g("total")),
            "correction_applied": corrected,
            "confidence": "MEDIUM" if corrected else "HIGH",
            "flags": ["NAME_CORRECTED"] if corrected else [],
        }

    def _extract_summary(self, text: str) -> dict:
        """Extract summary totals from invoice text."""
        amounts = [parse_amount(m) for m in AMOUNT_RE.findall(text)]

        def find_amount(pattern: str) -> float:
            m = re.search(pattern, text, re.I)
            return parse_amount(m.group(1)) if m else 0.0

        return {
            "subtotal": 0.0,
            "total_discount": 0.0,
            "total_cgst": find_amount(r"(?:Total\s*)?CGST.*?([\d,]+\.\d{2})"),
            "total_sgst": find_amount(r"(?:Total\s*)?SGST.*?([\d,]+\.\d{2})"),
            "total_igst": 0.0,
            "total_gst": 0.0,
            "grand_total": find_amount(r"Grand\s*Total.*?([\d,]+\.\d{2})") or (max(amounts) if amounts else 0.0),
            "tds_amount": find_amount(r"TDS.*?([\d,]+\.\d{2})"),
            "net_payable": find_amount(r"(?:Net\s*)?Payable.*?([\d,]+\.\d{2})"),
            "amount_in_words": "",
        }

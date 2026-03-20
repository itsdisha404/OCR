"""
mardia.py — Mardia Pharma
GSTIN: 07AAAHR3612M1Z4
Digital PDF, 6 pages, 200+ items. Has Old/New MRP column. Very dense.
Products: Amlong, Arbitel, Aziderm, Betafree etc.
"""

import re
from .base import GenericParser, parse_amount, apply_product_correction, HSN_RE


class Parser(GenericParser):
    VENDOR_NAME = "Mardia Pharma"

    def parse(self, full_text: str, tables: list, source_file: str) -> dict:
        header = self._extract_header(full_text, source_file)
        header["vendor_name"] = self.VENDOR_NAME
        line_items = self._extract_items(tables)
        summary = self._extract_summary(full_text)
        return {"header": header, "line_items": line_items, "summary": summary}

    def _extract_items(self, tables: list) -> list:
        items = []
        sr = 1
        for page_tables in tables:
            if not page_tables:
                continue
            for table in page_tables:
                if not table or len(table) < 2:
                    continue
                col = self._find_columns(table[0])
                if not col:
                    continue
                for row in table[1:]:
                    if not row or not any(row):
                        continue
                    item = self._parse_row(row, col, sr)
                    if item:
                        items.append(item)
                        sr += 1
        return items

    def _find_columns(self, header_row: list) -> dict:
        col = {}
        for i, cell in enumerate(header_row or []):
            h = str(cell or "").lower().strip()
            if "product" in h or "desc" in h or "particulars" in h:
                col["product"] = i
            elif "hsn" in h:
                col["hsn"] = i
            elif "batch" in h:
                col["batch"] = i
            elif "exp" in h:
                col["expiry"] = i
            elif "new" in h and "mrp" in h:
                col["mrp"] = i
            elif "old" in h and "mrp" in h:
                col["old_mrp"] = i
            elif "mrp" in h and "mrp" not in col:
                col["mrp"] = i
            elif "ptr" in h:
                col["ptr"] = i
            elif "qty" in h:
                col["qty"] = i
            elif "disc" in h:
                col["disc"] = i
            elif "taxable" in h:
                col["taxable"] = i
            elif "cgst" in h:
                col["cgst"] = i
            elif "sgst" in h:
                col["sgst"] = i
            elif "total" in h or "net" in h:
                col["total"] = i
        return col if len(col) >= 4 else None

    def _parse_row(self, row, col, sr):
        def g(key):
            idx = col.get(key)
            return str(row[idx] or "").strip() if idx is not None and idx < len(row) else ""

        product_raw = g("product")
        if not product_raw or len(product_raw) < 2:
            return None

        hsn = g("hsn")
        if not hsn:
            row_str = " ".join(str(c or "") for c in row)
            hsn_m = HSN_RE.search(row_str)
            hsn = hsn_m.group(1) if hsn_m else ""

        product_fixed, corrected = apply_product_correction(product_raw)

        return {
            "sr_no": sr,
            "product_name": product_fixed,
            "product_raw": product_raw,
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
            "pts": 0.0,
            "rate": parse_amount(g("ptr")),
            "discount_pct": parse_amount(g("disc")),
            "discount_value": 0.0,
            "taxable_value": parse_amount(g("taxable")),
            "cgst_rate": 2.5,
            "cgst_amount": 0.0,
            "sgst_rate": 2.5,
            "sgst_amount": 0.0,
            "igst_rate": 0.0,
            "igst_amount": 0.0,
            "total_amount": parse_amount(g("total")),
            "correction_applied": corrected,
            "confidence": "MEDIUM" if corrected else "HIGH",
            "flags": ["NAME_CORRECTED"] if corrected else [],
        }

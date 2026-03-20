"""
dr_reddys.py — Dr. Reddy's Laboratories Limited
GSTIN: 07AAACD7999Q1ZM
Digital PDF, 6 pages, clean tables.
HSN code appears below product name in same cell.
GST mostly 5% (CGST 2.5 + SGST 2.5), some 18%.
"""

import re
from .base import GenericParser, parse_amount, apply_product_correction, HSN_RE


class Parser(GenericParser):
    VENDOR_NAME = "Dr. Reddy's Laboratories Limited"

    def parse(self, full_text: str, tables: list, source_file: str) -> dict:
        header = self._extract_header(full_text, source_file)
        header["vendor_name"] = self.VENDOR_NAME
        line_items = self._extract_items(tables, full_text)
        summary = self._extract_dr_reddys_summary(full_text)
        return {"header": header, "line_items": line_items, "summary": summary}

    def _extract_items(self, tables: list, full_text: str) -> list:
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
            if "desc" in h or "goods" in h or "product" in h:
                col["product"] = i
            elif "batch" in h:
                col["batch"] = i
            elif "exp" in h:
                col["expiry"] = i
            elif "mrp" in h:
                col["mrp"] = i
            elif "ptr" in h:
                col["ptr"] = i
            elif "qty" in h and "sale" in h:
                col["qty"] = i
            elif "qty" in h and col.get("qty") is None:
                col["qty"] = i
            elif "pts" in h or "rate" in h:
                col["pts"] = i
            elif "taxable" in h:
                col["taxable"] = i
            elif "cgst" in h and "rate" in h:
                col["cgst_r"] = i
            elif "cgst" in h and ("amt" in h or "amount" in h):
                col["cgst_a"] = i
            elif "sgst" in h and "rate" in h:
                col["sgst_r"] = i
            elif "sgst" in h and ("amt" in h or "amount" in h):
                col["sgst_a"] = i
            elif "total" in h:
                col["total"] = i
        return col if len(col) >= 4 else None

    def _parse_row(self, row: list, col: dict, sr: int) -> dict:
        def g(key):
            idx = col.get(key)
            return str(row[idx] or "").strip() if idx is not None and idx < len(row) else ""

        product_raw = g("product")
        if not product_raw or len(product_raw) < 3:
            return None

        # HSN may be in the product cell on a new line
        hsn_m = HSN_RE.search(product_raw)
        hsn = hsn_m.group(1) if hsn_m else ""
        product_name = product_raw.split("\n")[0].strip()
        product_fixed, corrected = apply_product_correction(product_name)

        taxable = parse_amount(g("taxable"))
        cgst_rate = parse_amount(g("cgst_r")) or 2.5
        cgst_amt = parse_amount(g("cgst_a"))
        sgst_rate = parse_amount(g("sgst_r")) or 2.5
        sgst_amt = parse_amount(g("sgst_a"))
        total = parse_amount(g("total"))

        return {
            "sr_no": sr,
            "product_name": product_fixed,
            "product_raw": product_name,
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
            "rate": parse_amount(g("pts")),
            "discount_pct": 0.0,
            "discount_value": 0.0,
            "taxable_value": taxable,
            "cgst_rate": cgst_rate,
            "cgst_amount": cgst_amt,
            "sgst_rate": sgst_rate,
            "sgst_amount": sgst_amt,
            "igst_rate": 0.0,
            "igst_amount": 0.0,
            "total_amount": total,
            "correction_applied": corrected,
            "confidence": "MEDIUM" if corrected else "HIGH",
            "flags": ["NAME_CORRECTED"] if corrected else [],
        }

    def _extract_dr_reddys_summary(self, text: str) -> dict:
        def find_amt(pattern):
            m = re.search(pattern, text, re.I)
            return parse_amount(m.group(1)) if m else 0.0

        return {
            "subtotal": 0.0,
            "total_discount": 0.0,
            "total_cgst": find_amt(r"(?:Total\s*)?CGST.*?([\d,]+\.\d{2})"),
            "total_sgst": find_amt(r"(?:Total\s*)?SGST.*?([\d,]+\.\d{2})"),
            "total_igst": 0.0,
            "total_gst": 0.0,
            "grand_total": find_amt(r"Grand\s*Total.*?([\d,]+\.\d{2})"),
            "tds_amount": find_amt(r"TDS.*?([\d,]+\.\d{2})"),
            "net_payable": find_amt(r"Payable.*?([\d,]+\.\d{2})"),
            "amount_in_words": "",
        }

"""
eris.py — Eris Healthcare
GSTIN: 07AATCS3717L1ZQ
Partially scanned/mixed quality. Bill To = Chhabra Healthcare.
Has P.T.R column (vs PTR elsewhere).
"""

from .base import GenericParser, parse_amount, apply_product_correction, HSN_RE


class Parser(GenericParser):
    VENDOR_NAME = "Eris Healthcare"

    def parse(self, full_text: str, tables: list, source_file: str) -> dict:
        header = self._extract_header(full_text, source_file)
        header["vendor_name"] = self.VENDOR_NAME
        line_items = self._extract_line_items_from_tables(tables)
        summary = self._extract_summary(full_text)
        return {"header": header, "line_items": line_items, "summary": summary}

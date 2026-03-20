"""
msn.py — MSN Laboratories
GSTIN: 07AADCM6283F1ZE
Digital PDF, partially cut off on right side. GST all 5%.
"""

from .base import GenericParser, parse_amount, apply_product_correction, HSN_RE


class Parser(GenericParser):
    VENDOR_NAME = "MSN Laboratories"

    def parse(self, full_text: str, tables: list, source_file: str) -> dict:
        header = self._extract_header(full_text, source_file)
        header["vendor_name"] = self.VENDOR_NAME
        line_items = self._extract_line_items_from_tables(tables)
        # Set default GST rates for MSN (all 5%)
        for item in line_items:
            if item.get("cgst_rate", 0) == 0:
                item["cgst_rate"] = 2.5
            if item.get("sgst_rate", 0) == 0:
                item["sgst_rate"] = 2.5
        summary = self._extract_summary(full_text)
        return {"header": header, "line_items": line_items, "summary": summary}

"""
abbvie.py — Abbvie
Scanned/photographed invoice — needs Surya OCR.
Photographed at angle, poor quality. Ophthalmic and derma products.
Products: Acular, Alphagan, Combigan, Lumigan, Osmega, Refresh etc.
"""

from .base import GenericParser, parse_amount, apply_product_correction, HSN_RE


class Parser(GenericParser):
    """
    Abbvie invoices are scanned images. The base GenericParser already
    handles table extraction. This parser adds Abbvie-specific defaults.
    """
    VENDOR_NAME = "AbbVie"

    def parse(self, full_text: str, tables: list, source_file: str) -> dict:
        header = self._extract_header(full_text, source_file)
        header["vendor_name"] = self.VENDOR_NAME
        line_items = self._extract_line_items_from_tables(tables)

        # Abbvie invoices are scanned → lower default confidence
        for item in line_items:
            if item.get("confidence") == "HIGH":
                item["confidence"] = "MEDIUM"
            if "SCANNED_SOURCE" not in item.get("flags", []):
                item.setdefault("flags", []).append("SCANNED_SOURCE")

        summary = self._extract_summary(full_text)
        return {"header": header, "line_items": line_items, "summary": summary}

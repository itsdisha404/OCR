"""
vendor_router.py
────────────────
Maps GSTIN fragments and company name patterns to vendor IDs.
Dynamically imports vendor-specific parsers from the parsers/ package.
"""

from importlib import import_module

# Maps vendor_id => list of GSTIN or company name patterns to match in text
VENDOR_SIGNATURES = {
    "dr_reddys": ["07AAACD7999Q1ZM", "Dr. Reddy", "DRLLDEL"],
    "eris":      ["07AATCS3717L1ZQ", "Eris Healthcare"],
    "fdc":       ["07AAACF0253H1Z2", "FDC Limited"],
    "mardia":    ["07AAAHR3612M1Z4", "MARDIA PHARMA"],
    "maviga":    ["07AARCM6852G1ZZ", "MAVIGA LOGISYS"],
    "msn":       ["07AADCM6283F1ZE", "MSN LABORATORIES"],
    "nidhi":     ["07AAACN3519N1ZA", "NIDHI ASSOCIATES"],
    "pangea":    ["06AAFCP6328J1Z8", "PANGEA ENTERPRISES"],
    "corona":    ["07AACCC5173F1ZU", "CORONA Remedies"],
    "mankind":   ["07AAACM9401C1ZX", "MANKIND PHARMA", "Curis Mankind"],
    "abbvie":    ["AbbVie", "Allergan"],
}


def detect_vendor(text: str) -> str:
    """
    Detect vendor from invoice text by matching GSTIN or company name patterns.
    Returns vendor_id string or 'unknown'.
    """
    text_upper = text.upper()
    for vendor_id, patterns in VENDOR_SIGNATURES.items():
        for p in patterns:
            if p.upper() in text_upper:
                return vendor_id
    return "unknown"


def get_parser(vendor_id: str):
    """
    Dynamically import and return a Parser instance from parsers/<vendor_id>.py.
    Falls back to GenericParser if vendor module not found.
    """
    try:
        module = import_module(f"parsers.{vendor_id}")
        return module.Parser()
    except (ImportError, AttributeError):
        from parsers.base import GenericParser
        return GenericParser()

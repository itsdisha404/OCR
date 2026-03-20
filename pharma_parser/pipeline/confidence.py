"""
confidence.py
─────────────
Confidence scoring for individual line items and overall invoices.
Used by validators.py and callable independently.
"""


def score_confidence(
    product_name_norm: str,
    hsn_code: str,
    gst_computed: float,
    gst_invoice: float,
    valid_hsn: set = None,
    valid_products: set = None,
    correction_applied: bool = False,
) -> tuple:
    """
    Score the confidence of a single line item extraction.

    Returns:
        (confidence_level, reasons) where confidence_level is
        "HIGH", "MEDIUM", or "LOW" and reasons is a list of strings.
    """
    reasons = []

    hsn_valid = (not valid_hsn) or (hsn_code in valid_hsn)
    product_valid = (not valid_products) or (product_name_norm in valid_products)
    gst_match = abs(gst_computed - gst_invoice) < 0.50

    if not hsn_valid:
        reasons.append("invalid_hsn")
    if not product_valid:
        reasons.append("unknown_product")
    if not gst_match:
        reasons.append(f"gst_mismatch_diff={round(gst_computed - gst_invoice, 2)}")
    if correction_applied:
        reasons.append("name_corrected")

    if not reasons:
        return "HIGH", ["exact_match"]
    elif len(reasons) == 1 and "name_corrected" in reasons:
        return "MEDIUM", reasons
    else:
        return "LOW", reasons

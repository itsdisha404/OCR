"""
field_spec.py
─────────────
Authoritative field definitions for the entire pipeline.
All pipeline functions accept and return plain Python dicts matching these specs.
"""

# ── Header fields ─────────────────────────────────────────────────────────────

HEADER_FIELDS = {
    # Identification
    "invoice_no":       str,
    "invoice_date":     str,    # YYYY-MM-DD always
    "due_date":         str,
    "po_number":        str,
    "delivery_number":  str,

    # Seller
    "vendor_name":      str,
    "vendor_gstin":     str,    # 15-char GSTIN
    "vendor_pan":       str,
    "vendor_state":     str,
    "vendor_address":   str,

    # Buyer
    "buyer_name":       str,
    "buyer_gstin":      str,
    "buyer_pan":        str,
    "buyer_dl_no":      str,    # Drug Licence number
    "buyer_fssai":      str,
    "buyer_address":    str,
    "payment_terms":    str,

    # Extra
    "eway_bill_no":     str,
    "irn":              str,    # 64-char hash

    # Pipeline metadata
    "source_file":      str,
    "vendor_id":        str,    # dr_reddys, fdc, maviga, etc.
    "extraction_method": str,   # digital / surya_ocr / mixed
}


# ── Line-item fields ─────────────────────────────────────────────────────────

LINE_ITEM_FIELDS = {
    "sr_no":            int,
    "product_name":     str,
    "product_raw":      str,    # before correction
    "hsn_code":         str,    # 8-digit string
    "batch_no":         str,
    "expiry_date":      str,    # YYYY-MM-DD
    "mfg_name":         str,
    "pack_size":        str,
    "category":         str,    # DS / DN / F / AM / C

    # Quantities
    "qty_billed":       float,
    "qty_free":         float,
    "qty_total":        float,

    # Pricing
    "mrp":              float,
    "ptr":              float,  # price to retailer
    "pts":              float,  # price to stockist
    "rate":             float,  # actual rate charged

    # Tax
    "discount_pct":     float,
    "discount_value":   float,
    "taxable_value":    float,
    "cgst_rate":        float,  # 2.5 or 9.0
    "cgst_amount":      float,
    "sgst_rate":        float,
    "sgst_amount":      float,
    "igst_rate":        float,  # 0.0 for intra-state
    "igst_amount":      float,
    "total_amount":     float,

    # Quality
    "correction_applied": bool,
    "confidence":       str,    # HIGH / MEDIUM / LOW
    "flags":            list,   # ["GST_MISMATCH", "INVALID_HSN", ...]
}


# ── Summary fields ────────────────────────────────────────────────────────────

SUMMARY_FIELDS = {
    "subtotal":         float,
    "total_discount":   float,
    "total_cgst":       float,
    "total_sgst":       float,
    "total_igst":       float,
    "total_gst":        float,
    "grand_total":      float,
    "tds_amount":       float,
    "net_payable":      float,
    "amount_in_words":  str,
}


# ── Valid HSN codes for pharma (common) ───────────────────────────────────────

COMMON_HSN_CODES = {
    "30049099",  # Medicaments NES
    "30049069",  # Topical medicaments
    "30049039",  # Ophthalmic preparations
    "30042019",  # Antibiotics
    "21069099",  # Food supplements
    "33049990",  # Cosmetics NES
    "33061020",  # Toothpaste
    "33061090",  # Other dental
    "30039090",  # Other pharmaceutical products
    "30049029",  # Other medicaments
    "30041019",  # Penicillins
    "30049011",  # Ayurvedic preparations
}


def empty_header() -> dict:
    """Return a header dict with all fields set to empty defaults."""
    return {k: "" for k in HEADER_FIELDS}


def empty_line_item() -> dict:
    """Return a line item dict with all fields set to empty defaults."""
    defaults = {}
    for k, t in LINE_ITEM_FIELDS.items():
        if t == int:
            defaults[k] = 0
        elif t == float:
            defaults[k] = 0.0
        elif t == bool:
            defaults[k] = False
        elif t == list:
            defaults[k] = []
        else:
            defaults[k] = ""
    return defaults


def empty_summary() -> dict:
    """Return a summary dict with all fields set to empty defaults."""
    defaults = {}
    for k, t in SUMMARY_FIELDS.items():
        defaults[k] = 0.0 if t == float else ""
    return defaults

#!/usr/bin/env python3
"""
Test the fixes for total amount extraction and duplicate items.
"""

import re
from typing import Optional

def _round2(val: float) -> float:
    """Round to 2 decimal places."""
    return round(val, 2)

def _extract_total_amount_from_row_text_old(text: str) -> Optional[float]:
    """OLD version - picks largest value from multiple columns."""
    if not text:
        return None

    # Extract all decimal values and pick the largest
    decimal_pattern = r'\d[\d,]*\.\d{1,2}'
    decimal_matches = re.findall(decimal_pattern, text)

    if decimal_matches:
        candidates = []
        for match in decimal_matches:
            try:
                val = float(match.replace(",", ""))
                if 50.0 <= val <= 100000:
                    candidates.append(val)
            except ValueError:
                continue

        if candidates:
            return _round2(max(candidates))  # PICKS LARGEST - BAD!

    return None


def _extract_total_amount_from_row_text_new(text: str) -> Optional[float]:
    """NEW version - picks rightmost value only."""
    if not text:
        return None

    # Find the RIGHTMOST complete numeric value at end of text (before manufacturer name)
    rightmost_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*[A-Z][A-Z]', text)

    if rightmost_match:
        try:
            value = float(rightmost_match.group(1).replace(",", ""))
            if 50.0 <= value <= 100000:
                return _round2(value)
        except ValueError:
            pass

    # Fallback: Look for explicit transaction/total value labels
    transaction_patterns = [
        r'(?:transaction|trans)\s*(?:value|val)[\s\-:]*(\d[\d,]*\.?\d*)',
        r'(\d[\d,]*\.?\d*)\s*transaction',
        r'final\s*(?:amount|value)[\s\-:]*(\d[\d,]*\.?\d*)',
    ]

    for pattern in transaction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
                if 10.0 <= value <= 100000:
                    return _round2(value)
            except ValueError:
                continue

    return None


def test_total_amount_extraction():
    """Test the total amount extraction fixes."""

    test_cases = [
        # (row_text, expected_amount, description)
        ("1 ENVAS 5 MG TABLET (40x5 T)2Ox2XIST 30049071 JKBU24006 07/2027 40 0 BOX 1,598.40 62.16 44.40 39.96 12.00 1,598.40 0.00 10.00 1,598.40 CADILA", 1598.40, "Should extract rightmost 1,598.40"),
        ("2 CILEDGE 1OMG TAB 10XIO T 30049099 PC24008 05/2026 10 0 BOX 192.24 29.90 21.36 19.22 12.00 192.24 0.00 10.00 192.24 PHARMA", 192.24, "Should extract rightmost 192.24"),
        ("3 LORFAST AM TAB 20XIO T 30049039 CJ24007 05/2026 200 0 BOX 13,558.90 116.00 82.86 67.80 12.00 13,558.90 1.00 10.00 13,558.90 LIMITED", 13558.90, "Should extract rightmost 13,558.90"),
    ]

    print("=" * 80)
    print("Testing Total Amount Extraction Fix")
    print("=" * 80)

    for row_text, expected, description in test_cases:
        old_result = _extract_total_amount_from_row_text_old(row_text)
        new_result = _extract_total_amount_from_row_text_new(row_text)

        print(f"\n{description}")
        print(f"Expected:  {expected}")
        print(f"Old Logic: {old_result} {'✅' if old_result and abs(old_result - expected) < 0.01 else '❌'}")
        print(f"New Logic: {new_result} {'✅' if new_result and abs(new_result - expected) < 0.01 else '❌'}")


def test_duplicate_detection():
    """Test that duplicate detection now uses product+batch pairs."""

    print("\n" + "=" * 80)
    print("Testing Duplicate Detection Fix")
    print("=" * 80)

    # Simulate the seen_batches logic
    seen_batches_old = set()
    seen_batches_new = set()

    items = [
        ("ENVAS 5MG TABLET", "JKBU24006"),  # Item 1
        ("CILEDGE 10MG TAB", "PC24008"),    # Item 2 (different product, different batch)
        ("ENVAS 5MG TABLET", "PC24008"),    # Item 3 (same product as 1, different batch)
        ("CILEDGE 10MG TAB", "PC24008"),    # Item 4 (DUPLICATE of item 2 - same product & batch)
        ("ENVAS 5MG TABLET", "JKBU24006"), # Item 5 (DUPLICATE of item 1 - same product & batch)
    ]

    print("\nProcessing items:")
    for i, (product, batch) in enumerate(items, 1):
        # OLD logic: just track batches
        if batch in seen_batches_old:
            print(f"  Item {i}: ({product}, {batch}) - SKIPPED (old logic)")
        else:
            seen_batches_old.add(batch)
            print(f"  Item {i}: ({product}, {batch}) - KEPT (old logic)")

    print()

    for i, (product, batch) in enumerate(items, 1):
        # NEW logic: track product+batch pairs
        product_batch_key = (product, batch)
        if product_batch_key in seen_batches_new:
            print(f"  Item {i}: ({product}, {batch}) - SKIPPED (new logic)")
        else:
            seen_batches_new.add(product_batch_key)
            print(f"  Item {i}: ({product}, {batch}) - KEPT (new logic)")

    print("\nExpected Results:")
    print("  - Old logic mistakenly skips Item 3 (same product, different batch)")
    print("  - New logic correctly keeps Item 3 and only skips true duplicates 4 & 5")


if __name__ == "__main__":
    test_total_amount_extraction()
    test_duplicate_detection()

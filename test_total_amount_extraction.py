#!/usr/bin/env python3
"""
Test the improved total amount extraction logic.
"""

import re
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_total_amount_extraction():
    """Test the improved total amount extraction function."""

    # Import the updated function
    from processor import _extract_total_amount_from_row_text

    # Test cases based on actual invoice data
    test_cases = [
        # (row_text, expected_amount)
        ("1 ENVAS 5 MG TABLET (40x5 T)2Ox2XIST 30049071 JKBU24006 07/2027 40 0 BOX 1,598.40 62.16 44.40 39.96 12.00 1,598.40 0.00 10.00 1,598.40 CADILA PHARMACEUTICALS LIMITED", 1598.40),
        ("2 CILEDGE 1OMG TAB 10XIO T 30049099 PC24008 05/2026 10 0 BOX 192.24 29.90 21.36 19.22 12.00 192.24 0.00 10.00 192.24 CADILA PHARMACEUTICALS LIMITED", 192.24),
        ("3 LORFAST AM TAB 20XIO T 30049039 CJ24007 05/2026 200 0 BOX 13,558.90 116.00 82.86 67.80 12.00 13,558.90 1.00 10.00 13,558.90 CADILA PHARMACEUTICALS LIMITED", 13558.90),
        ("7 CADIQUIS SMG TAB 3X10 T 30049099 JKJU24001 04/2026 15 0 BOX 1,055.83 109.50 78.21 70.39 12.00 1,055.83 0.00 10.00 1,055.83 CADILA PHARMACEUTICALS LIMITED", 1055.83),
    ]

    print("🧪 Testing Total Amount Extraction Logic")
    print("=" * 60)

    for i, (row_text, expected) in enumerate(test_cases, 1):
        result = _extract_total_amount_from_row_text(row_text)
        success = "✅" if result and abs(result - expected) < 0.01 else "❌"
        print(f"Test {i}: {success}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
        print(f"   Row:      {row_text[:80]}...")
        print()

def test_column_detection():
    """Test improved column detection with sample header."""

    from processor import _match_column_from_text, _extract_columns_from_complex_header

    print("🔍 Testing Column Detection Logic")
    print("=" * 60)

    # Test complex header from actual PDF
    complex_header = "PTR / TP Value- Scheme Value- TD%- Transaction Value"

    # Test single column matching
    test_texts = [
        "Transaction Value",
        "TD%",
        "Scheme Value",
        "PTR / TP Value"
    ]

    for text in test_texts:
        result = _match_column_from_text(text)
        print(f"'{text}' → '{result}'")

    print(f"\nComplex header: '{complex_header}'")
    multi_matches = _extract_columns_from_complex_header(complex_header, 400.0, 200.0)
    print(f"Multi-column matches: {multi_matches}")

if __name__ == "__main__":
    test_total_amount_extraction()
    print()
    test_column_detection()
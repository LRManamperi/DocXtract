#!/usr/bin/env python3
"""
Test pytesseract installation and functionality
"""

try:
    import pytesseract
    print("✅ pytesseract imported successfully")
    print(f"Tesseract command: {pytesseract.pytesseract.tesseract_cmd}")

    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as e:
        print(f"Could not get version: {e}")

    # Test OCR on a simple image
    import numpy as np
    import cv2

    # Create a simple test image
    test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    try:
        text = pytesseract.image_to_string(test_img)
        print(f"OCR test result: '{text.strip()}'")
        if "TEST" in text.upper():
            print("✅ OCR working correctly")
        else:
            print("⚠️ OCR working but not accurate")
    except Exception as e:
        print(f"❌ OCR test failed: {e}")

except ImportError as e:
    print(f"❌ pytesseract import failed: {e}")
    print("Please install pytesseract: pip install pytesseract")
    print("Also install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
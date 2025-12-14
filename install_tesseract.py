#!/usr/bin/env python3
"""
Script to install Tesseract OCR on Windows
"""

import os
import sys
import subprocess
import urllib.request
import zipfile

def download_tesseract():
    """Download and install Tesseract OCR for Windows"""
    print("🔄 Downloading Tesseract OCR for Windows...")

    # Tesseract download URL (64-bit)
    tesseract_url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.4.20240514.exe"

    installer_path = "tesseract_installer.exe"

    try:
        print(f"Downloading from: {tesseract_url}")
        urllib.request.urlretrieve(tesseract_url, installer_path)
        print("✅ Download complete")

        print("🔄 Running installer (please follow the installation prompts)...")
        subprocess.run([installer_path], check=True)

        print("✅ Tesseract installation complete!")
        print("📝 Please ensure Tesseract is added to your PATH")
        print("   Default installation path: C:\\Program Files\\Tesseract-OCR")

    except Exception as e:
        print(f"❌ Download/installation failed: {e}")
        print("📝 Manual installation:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Run the installer")
        print("   3. Add to PATH: C:\\Program Files\\Tesseract-OCR")

    finally:
        # Clean up installer
        if os.path.exists(installer_path):
            os.remove(installer_path)

def check_tesseract():
    """Check if Tesseract is already installed"""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract is already installed: {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract not found: {e}")
        return False

def main():
    print("🔍 Checking Tesseract OCR installation...")

    if check_tesseract():
        print("🎉 Tesseract is ready to use!")
        return

    print("\n📦 Tesseract OCR is required for table text extraction.")
    print("Choose installation method:")
    print("1. Automatic download and install")
    print("2. Manual installation instructions")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        download_tesseract()
    else:
        print("\n📝 Manual Installation Instructions:")
        print("==================================")
        print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Choose the Windows installer (tesseract-ocr-w64-setup-*.exe)")
        print("3. Run the installer and follow the prompts")
        print("4. During installation, make sure to:")
        print("   - Install all languages you need")
        print("   - Add Tesseract to your system PATH")
        print("5. Restart your command prompt/terminal")
        print("6. Test with: tesseract --version")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Development installation script for DocXtract
"""

import subprocess
import sys
import os

def main():
    """Install DocXtract in development mode"""
    print("Installing DocXtract in development mode...")

    # Install in development mode
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ DocXtract installed successfully in development mode!")
        print("\nYou can now import DocXtract from anywhere:")
        print("from docxtract import DocXtract")
        print("\nTo run the dashboard:")
        print("python run_dashboard.py")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("\nAlternative: Use the run_dashboard.py script which sets PYTHONPATH automatically")
        sys.exit(1)

if __name__ == "__main__":
    main()
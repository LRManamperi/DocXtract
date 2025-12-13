#!/usr/bin/env python3
"""
Script to run the DocXtract Dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), 'UI', 'dashboard.py')

    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)

    print("Starting DocXtract Dashboard...")
    print(f"Dashboard file: {dashboard_path}")

    # Run streamlit
    cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path]
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Script to run the DocXtract Dashboard
Automatically sets up the Python path for local development
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

    # Add current directory to Python path for local development
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env['PYTHONPATH'] = current_dir

    print("Starting DocXtract Dashboard...")
    print(f"Dashboard file: {dashboard_path}")
    print(f"PYTHONPATH set to: {current_dir}")

    # Run streamlit - use port 8501 if 8502 is in use
    import socket
    port = 8502
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    if result == 0:
        print(f"Port {port} is in use. Trying port 8501...")
        port = 8501
    
    print(f"Starting Streamlit on port {port}...")
    cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path, '--server.port', str(port)]
    subprocess.run(cmd, env=env)

if __name__ == '__main__':
    main()
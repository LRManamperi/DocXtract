#!/usr/bin/env python3
"""
Simple script to run the DocXtract Dashboard
"""

import sys
import os
import subprocess

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set environment
env = os.environ.copy()
env['PYTHONPATH'] = current_dir

# Dashboard path
dashboard_path = os.path.join(current_dir, 'UI', 'dashboard.py')

if not os.path.exists(dashboard_path):
    print(f"❌ Error: Dashboard file not found at {dashboard_path}")
    sys.exit(1)

print("🚀 Starting DocXtract Dashboard...")
print(f"📁 Dashboard: {dashboard_path}")
print(f"🌐 The dashboard will open in your browser")
print(f"🔗 If it doesn't open automatically, go to: http://localhost:8501")
print("\n💡 Press Ctrl+C to stop the dashboard\n")

# Run streamlit - it will automatically use port 8501 or find an available port
cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path]
subprocess.run(cmd, env=env)


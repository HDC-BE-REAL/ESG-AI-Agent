
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.tools.regulation_tool import _monitor_instance

print("Generated Report:")
try:
    report = _monitor_instance.generate_report()
    print(report)
except Exception as e:
    print(f"Error: {e}")

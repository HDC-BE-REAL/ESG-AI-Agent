import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tools.regulation_tool import _monitor_instance

print("Testing generate_report...")
report = _monitor_instance.generate_report("test query")
print("\n--- REPORT OUTPUT ---")
print(report)
print("---------------------")

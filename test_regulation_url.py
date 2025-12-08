import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tools.regulation_tool import _monitor_instance

# Mock history with origin_url
mock_history = {
    "test_key": {
        "title": "Test Report with Origin URL",
        "processed_at": datetime.now().isoformat(),
        "files": ["/tmp/test.pdf"],
        "summary": "This is a summary.\nLine 2.\nLine 3.",
        "origin_url": "https://www.example.com/original-report"
    }
}
_monitor_instance.history = mock_history

print("Testing generate_report with origin_url...")
report = _monitor_instance.generate_report("test query")
print("\n--- REPORT OUTPUT ---")
print(report)
print("---------------------")

if "https://www.example.com/original-report" in report:
    print("SUCCESS: Origin URL found in report.")
else:
    print("FAILURE: Origin URL NOT found in report.")

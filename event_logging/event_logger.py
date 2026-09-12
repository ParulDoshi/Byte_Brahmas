"""
Writes every confirmed/flagged event to disk in two lightweight,
timestamped, structured text formats:
  - logs/session_log.csv    (spreadsheet-friendly)
  - logs/session_log.jsonl  (one JSON object per line -- easy to
                              parse programmatically for status/outcome)
and generates a plain-text end-of-session summary report.

In the real system these are what would get buffered and downlinked
to mission control whenever bandwidth allows.
"""

import csv
import json
import os
from datetime import datetime


class EventLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, "session_log.csv")
        self.jsonl_path = os.path.join(log_dir, "session_log.jsonl")

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "activity", "status"])

        self.entries = []

    def log(self, activity, status):
        timestamp = datetime.now().isoformat(timespec="seconds")

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, activity, status])

        record = {"timestamp": timestamp, "activity": activity, "status": status}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        self.entries.append(record)

    def write_summary(self, validator, path=None):
        """Writes a plain-text end-of-session report: what was done,
        in what order, and the overall outcome. Returns the path."""
        path = path or os.path.join(
            self.log_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        completed = validator.current_step
        total = len(validator.steps)

        with open(path, "w") as f:
            f.write("ON-BOARD HAR SESSION SUMMARY\n")
            f.write("=" * 32 + "\n")
            f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"Steps completed: {completed}/{total}\n")
            f.write(f"Outcome: {'COMPLETE' if completed >= total else 'INCOMPLETE'}\n\n")
            f.write("Event timeline:\n")
            for e in self.entries:
                f.write(f"  [{e['timestamp']}] {e['activity']:<20} -> {e['status']}\n")
        return path

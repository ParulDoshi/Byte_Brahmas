"""
Entry point for the on-board HAR prototype -- launches the GUI
monitoring dashboard (video feed, current/next step, event log).

Run:  python main.py

For a lightweight terminal-only version (no Tkinter window), use
main_cli.py instead.
"""

from pathlib import Path

from gui.app import run

BASE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = str(BASE_DIR / "config" / "protocol.yaml")
APP_CONFIG_PATH = str(BASE_DIR / "config" / "app_config.yaml")

if __name__ == "__main__":
    run(PROTOCOL_PATH, APP_CONFIG_PATH)

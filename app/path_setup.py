"""Make the repo root (where ``config`` lives) and this ``app`` dir importable.

Import this module first, before importing ``config`` or any sibling module, so
absolute imports work whether the app is launched via ``app.py`` or a sibling
module is imported directly (e.g. from a test).
"""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

for _p in (str(ROOT_DIR), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

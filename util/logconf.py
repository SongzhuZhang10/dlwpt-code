"""Central logging configuration.

Creates **exactly one** log file in the main process.  Worker processes (e.g., from
multiprocessing) do not create separate log files.
"""

import logging
import os
import datetime
import multiprocessing

# ---------------------------------------------------------------------------
# Where to store .log files – default is ./logs next to your notebook/script
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Root logger setup & formatter
# ---------------------------------------------------------------------------
root = logging.getLogger()
root.setLevel(logging.INFO)
LOGFMT = (
    "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s "
    "%(message)s"
)
_formatter = logging.Formatter(LOGFMT)

# ---------------------------------------------------------------------------
# Determine if this is the main process
# ---------------------------------------------------------------------------
is_main = (multiprocessing.current_process().name == 'MainProcess')

# ---------------------------------------------------------------------------
# File handler – only once, and only in main process
# ---------------------------------------------------------------------------
if is_main and not any(isinstance(h, logging.FileHandler) for h in root.handlers):
    _timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_FILE_PATH = os.path.join(LOG_DIR, f"{_timestamp}.log")
    fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    fh.setFormatter(_formatter)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)
else:
    # Use the existing file path from the first handler for __all__
    existing = next((h for h in root.handlers if isinstance(h, logging.FileHandler)), None)
    LOG_FILE_PATH = existing.baseFilename if existing else None

# ---------------------------------------------------------------------------
# Console handler – only once
# ---------------------------------------------------------------------------
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
           for h in root.handlers):
    sh = logging.StreamHandler()
    sh.setFormatter(_formatter)
    sh.setLevel(logging.INFO)
    root.addHandler(sh)

# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------
__all__ = ["LOG_FILE_PATH", "LOG_DIR"]

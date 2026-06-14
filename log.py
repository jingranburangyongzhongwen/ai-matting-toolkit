"""Centralized logging configuration for the matting toolkit.

Usage in each module:
    from log import get_logger
    logger = get_logger(__name__)
    logger.info("model loaded")
"""

import logging
import os
import sys
import threading

_initialized = False
_init_lock = threading.Lock()


def _init_root():
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        _initialized = True

    level_name = os.environ.get("MATTING_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "[%(asctime)s %(name)s] %(message)s"
    datefmt = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger("matting")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'matting' namespace."""
    _init_root()
    # Strip package prefix so log names stay short: 'engines.rmbg2' -> 'rmbg2'
    short = name.rsplit(".", 1)[-1] if "." in name else name
    return logging.getLogger(f"matting.{short}")

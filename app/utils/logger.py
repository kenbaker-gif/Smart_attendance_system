import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os

from app.settings import settings

# --- 1. DEFINE PATHS AND CREATE DIRECTORY ---
# Use the Current Working Directory (CWD) to define the logs path.
LOG_DIR = Path(os.getcwd()) / "logs"
try:
    LOG_DIR.mkdir(exist_ok=True, parents=True)
except Exception:
    # If we cannot create the logs directory, continue with console-only logging
    pass

LOG_FILE = LOG_DIR / "attendance.log"

# --- 2. CONFIGURE LOGGER AND LEVEL ---
logger = logging.getLogger("attendance_system")
level_name = (os.getenv("LOG_LEVEL") or settings.LOG_LEVEL or "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
logger.setLevel(level)

# --- 3. DEFINE HANDLERS ---
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
file_handler.setLevel(level)

console_handler = logging.StreamHandler()
console_handler.setLevel(level)

# --- 4. DEFINE FORMATTER AND ATTACH ---
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Avoid adding duplicate handlers when module is imported multiple times
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Logging configuration loaded successfully.")
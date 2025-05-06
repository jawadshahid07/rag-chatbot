# rag/logger.py

import logging

# Configure root logger
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

# Avoid duplicate logs
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

"""
Phase 10 — Structured JSON Logging
Call setup_logging() once at app startup.
All subsequent logging.getLogger(...) calls emit JSON lines to stdout.
"""

import json
import logging
import sys
import time


class _JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                log_obj[key] = val
        return json.dumps(log_obj, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON output. Safe to call multiple times."""
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, _JSONFormatter)
           for h in root.handlers):
        return  # already set up

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

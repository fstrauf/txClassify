import os
import sys
import logging

# Gunicorn config
bind = "0.0.0.0:" + str(os.environ.get("PORT", 5001))
workers = int(os.environ.get("WEB_CONCURRENCY", 4))
worker_class = "sync"
timeout = 120  # Increased timeout for long-running operations
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Enable logging
capture_output = True
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Access log format
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" request_time=%(M)s'
)

# Logging configuration
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(process)d] [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": sys.stdout,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "gunicorn.error": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": True,
        },
        "gunicorn.access": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": True,
        },
        "": {  # Root logger
            "level": "INFO",
            "handlers": ["console"],
        },
    },
}


def when_ready(server):
    """Log when Gunicorn is ready to serve requests."""
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Ready ===")
    logger.info(f"Listening on: {bind}")
    logger.info(f"Using workers: {workers}")
    logger.info(f"Worker class: {worker_class}")


def on_starting(server):
    """Log when Gunicorn starts."""
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Starting ===")


def post_fork(server, worker):
    """Log when a worker starts."""
    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker spawned (pid: {worker.pid})")


def worker_abort(worker):
    """Log when a worker is aborted."""
    logger = logging.getLogger("gunicorn.error")
    logger.error(f"Worker aborted (pid: {worker.pid})")


def on_exit(server):
    """Log when Gunicorn exits."""
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Shutting Down ===")


def worker_exit(server, worker):
    """Clean up when a worker exits."""
    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker exiting (pid: {worker.pid})")

    # Attempt to clean up database connections
    try:
        # Close any remaining database connections
        from psycopg2 import pool
        from utils.db_utils import get_connection_pool

        # Try to close the connection pool if it exists
        connection_pool = get_connection_pool()
        if connection_pool:
            connection_pool.closeall()
            logger.info(f"Worker {worker.pid} closed all database connections")
    except Exception as e:
        logger.error(f"Error closing database connections in worker {worker.pid}: {e}")

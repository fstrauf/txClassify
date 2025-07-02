import os
import sys
import logging
import gc

# Gunicorn config
bind = "0.0.0.0:" + str(os.environ.get("PORT", 5001))

# Optimize worker count and memory management
# For memory-intensive operations, fewer workers is better
workers = int(os.environ.get("WEB_CONCURRENCY", 2))  # Reduced from 4 to 2
worker_class = "sync"
timeout = int(os.environ.get("WORKER_TIMEOUT", 180))  # Increased timeout for embedding operations (was 120)
keepalive = 2

# Memory management settings
max_requests = int(os.environ.get("MAX_REQUESTS", 200))  # Reduced from 1000 to prevent memory accumulation
max_requests_jitter = int(os.environ.get("MAX_REQUESTS_JITTER", 40))  # Reduced from 50

# Memory limits (if supported by system)
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance
preload_app = True  # Load app once and fork workers (saves memory)

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
    """Log when a worker starts and optimize memory settings."""
    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker spawned (pid: {worker.pid})")
    
    # Force garbage collection in new worker
    gc.collect()
    
    # Try to log initial memory usage
    try:
        import psutil
        process = psutil.Process(worker.pid)
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Worker {worker.pid} initial memory: {memory_mb:.2f} MB")
    except ImportError:
        pass


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

    # Log final memory usage if possible
    try:
        import psutil
        process = psutil.Process(worker.pid)
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Worker {worker.pid} final memory: {memory_mb:.2f} MB")
    except (ImportError, psutil.NoSuchProcess):
        pass

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
    
    # Force final garbage collection
    gc.collect()


def pre_request(worker, req):
    """Called before each request."""
    # Log memory usage for large requests (optional, can be disabled in production)
    if os.environ.get("DEBUG_MEMORY", "false").lower() == "true":
        try:
            import psutil
            process = psutil.Process(worker.pid)
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 500:  # Log if memory usage exceeds 500MB
                logger = logging.getLogger("gunicorn.error")
                logger.warning(f"Worker {worker.pid} high memory usage: {memory_mb:.2f} MB before request")
        except ImportError:
            pass


def post_request(worker, req, environ, resp):
    """Called after each request."""
    # Force garbage collection periodically
    if hasattr(worker, '_request_count'):
        worker._request_count += 1
    else:
        worker._request_count = 1
    
    # Force GC every 10 requests to prevent memory accumulation
    if worker._request_count % 10 == 0:
        collected = gc.collect()
        if os.environ.get("DEBUG_MEMORY", "false").lower() == "true":
            logger = logging.getLogger("gunicorn.error")
            logger.debug(f"Worker {worker.pid} forced GC after {worker._request_count} requests, collected {collected} objects")

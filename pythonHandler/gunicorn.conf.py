import os
import sys
import logging
import logging.handlers
from datetime import datetime

def access_log_format(now):
    """Custom access log format function."""
    return '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" request_time=%(M)s request_id=%(i)s'

# Logging configuration
logconfig_dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - [%(process)d] - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'access': {
            'format': '%(message)s',  # Raw message for access logs
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': sys.stdout,
        },
        'error_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': sys.stderr,
        },
        'access_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'access',
            'stream': sys.stdout,
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console'],
            'level': 'INFO',
        },
        'gunicorn.error': {
            'handlers': ['error_console'],
            'level': 'INFO',
            'propagate': False,
        },
        'gunicorn.access': {
            'handlers': ['access_console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Gunicorn config
bind = "0.0.0.0:" + str(os.environ.get("PORT", 5001))
workers = int(os.environ.get("WEB_CONCURRENCY", 4))
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Enable logging
capture_output = True
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Access log format
access_log_format = access_log_format(datetime.now())

# Add custom headers to access log
access_log_fields = [
    'h',  # Remote address
    'l',  # '-'
    'u',  # User name
    't',  # Date of the request
    'r',  # Status line (e.g. GET / HTTP/1.1)
    's',  # Status
    'b',  # Response length
    'f',  # Referer
    'a',  # User Agent
    'M',  # Request time in milliseconds
    ('i', 'X-Request-ID'),  # Custom Request ID header
]

def on_starting(server):
    """Log when Gunicorn starts."""
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Starting ===")
    logger.info(f"Workers: {workers}")
    logger.info(f"Bind: {bind}")
    logger.info(f"Worker class: {worker_class}")
    logger.info(f"Worker connections: {worker_connections}")
    logger.info(f"Timeout: {timeout}")
    logger.info(f"Keep-alive: {keepalive}")
    logger.info(f"Max requests: {max_requests}")
    logger.info(f"Max requests jitter: {max_requests_jitter}")

def post_fork(server, worker):
    """Log when a worker starts."""
    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker spawned (pid: {worker.pid})")

def pre_request(worker, req):
    """Log before processing a request."""
    worker.log.debug(f"Processing request: {req.method} {req.path}")

def post_request(worker, req, environ, resp):
    """Log after processing a request."""
    worker.log.debug(f"Completed request: {req.method} {req.path} - Status: {resp.status}")

def worker_abort(worker):
    """Log when a worker is aborted."""
    logger = logging.getLogger("gunicorn.error")
    logger.error(f"Worker aborted (pid: {worker.pid})")

def on_exit(server):
    """Log when Gunicorn exits."""
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Shutting Down ===") 
import os
import sys
import logging
import logging.handlers
from datetime import datetime

# Gunicorn config
bind = "0.0.0.0:" + str(os.environ.get("PORT", 5001))
workers = int(os.environ.get("WEB_CONCURRENCY", 4))
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 50
preload_app = True  # Preload the application code
reload = True  # Enable auto-reload on code changes

# Enable logging
capture_output = True
accesslog = "-"
errorlog = "-"
loglevel = "debug"

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" request_time=%(M)s request_id=%(i)s'

# Logging configuration
logconfig_dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s [%(process)d] [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'stream': sys.stdout,
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console']
    },
    'loggers': {
        'gunicorn.error': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': True,
        },
        'gunicorn.access': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': True,
        },
        '': {  # Root logger
            'level': 'DEBUG',
            'handlers': ['console'],
        },
        'app': {  # Application logger
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': True,
        }
    }
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
    """Log when a worker starts and configure worker logging."""
    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker spawned (pid: {worker.pid})")
    
    # Configure worker-specific logging
    worker_logger = logging.getLogger(f'worker.{worker.pid}')
    worker_logger.setLevel(logging.DEBUG)
    
    # Ensure all logs go to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(process)d] [%(levelname)s] %(name)s: %(message)s'
    ))
    worker_logger.addHandler(handler)

def pre_request(worker, req):
    """Log before processing a request."""
    logger = logging.getLogger(f'worker.{worker.pid}')
    logger.debug(f"Processing request: {req.method} {req.path}")
    logger.debug(f"Request headers: {dict(req.headers)}")

def post_request(worker, req, environ, resp):
    """Log after processing a request."""
    logger = logging.getLogger(f'worker.{worker.pid}')
    logger.debug(f"Request completed: {req.method} {req.path} - Status: {resp.status}")
    if hasattr(resp, 'headers'):
        logger.debug(f"Response headers: {dict(resp.headers)}")

def worker_abort(worker):
    """Log when a worker is aborted."""
    logger = logging.getLogger("gunicorn.error")
    logger.error(f"Worker aborted (pid: {worker.pid})")

def on_exit(server):
    """Log when Gunicorn exits."""
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Shutting Down ===") 
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

# Enable logging
capture_output = True
accesslog = "-"
errorlog = "-"
loglevel = "debug"  # Changed to debug for more verbose logging

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
            'qualname': 'gunicorn.error'
        },
        'gunicorn.access': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': True,
            'qualname': 'gunicorn.access'
        }
    }
}

def child_exit(server, worker):
    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker exited (pid: {worker.pid})")

def on_starting(server):
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Starting ===")
    logger.info(f"Workers: {workers}")
    logger.info(f"Bind: {bind}")

def post_fork(server, worker):
    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker spawned (pid: {worker.pid})")

def pre_request(worker, req):
    worker.log.debug(f"Processing request: {req.method} {req.path}")
    worker.log.debug(f"Request headers: {req.headers}")

def post_request(worker, req, environ, resp):
    worker.log.debug(f"Request completed: {req.method} {req.path} - Status: {resp.status}")

def worker_abort(worker):
    logger = logging.getLogger("gunicorn.error")
    logger.error(f"Worker aborted (pid: {worker.pid})")

def on_exit(server):
    logger = logging.getLogger("gunicorn.error")
    logger.info("=== Gunicorn Shutting Down ===") 
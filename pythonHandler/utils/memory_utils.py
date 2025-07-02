"""Memory monitoring utilities for performance optimization."""

import logging
import os
import gc
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Install with: pip install psutil")


def log_memory_usage(context: str, log_level: str = "info") -> Optional[float]:
    """
    Log current memory usage for debugging and optimization.
    
    Args:
        context: Description of the current operation context
        log_level: Logging level ('debug', 'info', 'warning', 'error')
        
    Returns:
        Current memory usage in MB, or None if psutil not available
    """
    if not PSUTIL_AVAILABLE:
        return None
        
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        log_message = f"[{context}] Memory usage: {memory_mb:.2f} MB ({memory_percent:.1f}%)"
        
        if log_level.lower() == "debug":
            logger.debug(log_message)
        elif log_level.lower() == "warning":
            logger.warning(log_message)
        elif log_level.lower() == "error":
            logger.error(log_message)
        else:  # default to info
            logger.info(log_message)
            
        return memory_mb
        
    except Exception as e:
        logger.warning(f"Error getting memory usage for {context}: {e}")
        return None


def force_garbage_collection(context: str = "manual") -> None:
    """
    Force garbage collection and log memory usage before/after.
    
    Args:
        context: Description of when garbage collection is being called
    """
    memory_before = log_memory_usage(f"{context} - before GC", "debug")
    
    # Force garbage collection
    collected = gc.collect()
    
    memory_after = log_memory_usage(f"{context} - after GC", "debug")
    
    if memory_before and memory_after:
        memory_freed = memory_before - memory_after
        logger.debug(f"[{context}] Garbage collection freed {memory_freed:.2f} MB, collected {collected} objects")
    else:
        logger.debug(f"[{context}] Garbage collection completed, collected {collected} objects")


def check_memory_threshold(threshold_mb: float = 1000, context: str = "memory check") -> bool:
    """
    Check if current memory usage exceeds a threshold.
    
    Args:
        threshold_mb: Memory threshold in MB
        context: Description of the current operation context
        
    Returns:
        True if memory usage exceeds threshold, False otherwise
    """
    if not PSUTIL_AVAILABLE:
        return False
        
    memory_mb = log_memory_usage(context, "debug")
    if memory_mb and memory_mb > threshold_mb:
        logger.warning(f"[{context}] Memory usage ({memory_mb:.2f} MB) exceeds threshold ({threshold_mb} MB)")
        return True
    
    return False


def get_system_memory_info() -> dict:
    """
    Get system memory information.
    
    Returns:
        Dictionary with system memory information, or empty dict if psutil not available
    """
    if not PSUTIL_AVAILABLE:
        return {}
        
    try:
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / 1024 / 1024,
            "available_mb": memory.available / 1024 / 1024,
            "used_mb": memory.used / 1024 / 1024,
            "free_mb": memory.free / 1024 / 1024,
            "percent_used": memory.percent
        }
    except Exception as e:
        logger.warning(f"Error getting system memory info: {e}")
        return {}


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""
    
    def __init__(self, operation_name: str, log_level: str = "info"):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_memory = None
        
    def __enter__(self):
        self.start_memory = log_memory_usage(f"{self.operation_name} - start", self.log_level)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = log_memory_usage(f"{self.operation_name} - end", self.log_level)
        
        if self.start_memory and end_memory:
            memory_delta = end_memory - self.start_memory
            if memory_delta > 0:
                logger.info(f"[{self.operation_name}] Memory increased by {memory_delta:.2f} MB")
            else:
                logger.info(f"[{self.operation_name}] Memory decreased by {abs(memory_delta):.2f} MB")

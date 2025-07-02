"""Performance and memory optimization configuration."""

import os
from typing import Optional


class PerformanceConfig:
    """Configuration for performance and memory optimizations."""
    
    def __init__(self):
        # Memory optimization settings
        self.enable_memory_monitoring = self._get_bool_env("ENABLE_MEMORY_MONITORING", True)
        self.memory_warning_threshold_mb = self._get_int_env("MEMORY_WARNING_THRESHOLD_MB", 1000)
        self.force_gc_after_batches = self._get_bool_env("FORCE_GC_AFTER_BATCHES", True)
        
        # Embedding generation settings
        self.embedding_batch_size = self._get_int_env("EMBEDDING_BATCH_SIZE", 10)
        self.embedding_max_dataset_size = self._get_int_env("EMBEDDING_MAX_DATASET_SIZE", 2000)
        
        # Worker timeout and resource limits
        self.worker_timeout_seconds = self._get_int_env("WORKER_TIMEOUT_SECONDS", 120)
        self.max_concurrent_embeddings = self._get_int_env("MAX_CONCURRENT_EMBEDDINGS", 1)
        
        # Deduplication and optimization
        self.enable_deduplication = self._get_bool_env("ENABLE_DEDUPLICATION", True)
        self.enable_early_fallback = self._get_bool_env("ENABLE_EARLY_FALLBACK", True)
        
        # Cache settings
        self.embedding_cache_size = self._get_int_env("EMBEDDING_CACHE_SIZE", 500)
        self.enable_embedding_cache = self._get_bool_env("ENABLE_EMBEDDING_CACHE", True)
        
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def should_use_batch_processing(self, dataset_size: int) -> bool:
        """Determine if batch processing should be used based on dataset size."""
        return dataset_size > 100  # Use batches for datasets larger than 100 items
    
    def get_optimal_batch_size(self, dataset_size: int) -> int:
        """Get optimal batch size based on dataset size and memory constraints."""
        if dataset_size <= 50:
            return min(10, dataset_size)
        elif dataset_size <= 500:
            return min(25, self.embedding_batch_size)
        else:
            return min(15, self.embedding_batch_size)  # Smaller batches for very large datasets
    
    def get_embedding_generation_batch_size(self) -> int:
        """Get batch size specifically for embedding generation (most memory intensive)."""
        return min(10, self.embedding_batch_size)


# Global instance
performance_config = PerformanceConfig()


def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration instance."""
    return performance_config


def update_cleaning_config_for_performance(cleaning_config, dataset_size: int) -> None:
    """Update CleaningConfig with performance optimizations based on dataset size."""
    perf_config = get_performance_config()
    
    # Adjust batch size based on dataset size
    cleaning_config.embedding_batch_size = perf_config.get_optimal_batch_size(dataset_size)
    
    # Enable/disable features based on performance config
    cleaning_config.embedding_use_cache = perf_config.enable_embedding_cache
    
    # For very large datasets, be more aggressive with optimizations
    if dataset_size > perf_config.embedding_max_dataset_size:
        cleaning_config.embedding_similarity_threshold = 0.90  # Higher threshold for faster processing
        cleaning_config.embedding_clustering_method = "similarity"  # Fastest method
        cleaning_config.embedding_fallback_to_fuzzy = True  # Enable fallback

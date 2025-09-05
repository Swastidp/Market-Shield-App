import hashlib
import pickle
import time
import os
from typing import Any, Optional, Dict
import streamlit as st

class CacheManager:
    """Advanced caching system for MarketShield."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.ttl_default = 3600  # 1 hour default TTL
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, content: str, analysis_type: str = "default") -> str:
        """Generate cache key for content."""
        content_hash = hashlib.sha256(f"{content}{analysis_type}".encode()).hexdigest()
        return f"{analysis_type}_{content_hash[:16]}"
    
    def get(self, content: str, analysis_type: str = "default") -> Optional[Any]:
        """Retrieve cached result."""
        cache_key = self._get_cache_key(content, analysis_type)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_item = self.memory_cache[cache_key]
            if time.time() - cached_item['timestamp'] < cached_item['ttl']:
                self.cache_stats['hits'] += 1
                return cached_item['data']
            else:
                del self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_item = pickle.load(f)
                
                if time.time() - cached_item['timestamp'] < cached_item['ttl']:
                    # Load back to memory cache
                    self.memory_cache[cache_key] = cached_item
                    self.cache_stats['hits'] += 1
                    return cached_item['data']
                else:
                    os.remove(cache_file)
            except Exception:
                pass
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, content: str, data: Any, analysis_type: str = "default", ttl: int = None) -> None:
        """Cache analysis result."""
        cache_key = self._get_cache_key(content, analysis_type)
        ttl = ttl or self.ttl_default
        
        cached_item = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
        
        # Store in memory cache
        self.memory_cache[cache_key] = cached_item
        
        # Store in disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_item, f)
        except Exception:
            pass  # Continue if disk cache fails
    
    def clear_expired(self) -> None:
        """Clear expired cache entries."""
        current_time = time.time()
        
        # Clear memory cache
        expired_keys = [
            key for key, item in self.memory_cache.items()
            if current_time - item['timestamp'] > item['ttl']
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        cached_item = pickle.load(f)
                    
                    if current_time - cached_item['timestamp'] > cached_item['ttl']:
                        os.remove(filepath)
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'memory_entries': len(self.memory_cache),
            'disk_entries': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        }

# Decorators for easy caching
def cached_analysis(analysis_type: str = "default", ttl: int = 3600):
    """Decorator for caching analysis results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = st.session_state.get('cache_manager')
            if not cache_manager:
                cache_manager = CacheManager()
                st.session_state.cache_manager = cache_manager
            
            # Extract content from args (assumes first arg is content)
            content = str(args[0]) if args else ""
            
            # Try cache first
            cached_result = cache_manager.get(content, analysis_type)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(content, result, analysis_type, ttl)
            
            return result
        return wrapper
    return decorator

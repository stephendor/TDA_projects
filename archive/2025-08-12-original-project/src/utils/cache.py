"""
Cache Management Utilities

Provides Redis-based caching for TDA computations and API responses.
"""

import os
import json
import logging
from typing import Any, Optional, Dict
import asyncio

import aioredis
from aioredis import Redis


logger = logging.getLogger(__name__)


class CacheManager:
    """Async Redis cache manager."""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = 3600  # 1 hour default TTL
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                max_connections=10
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Redis cache connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            # Continue without cache rather than failing
            self.redis = None
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis cache connection closed")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = None) -> bool:
        """Set value in cache with optional expiration."""
        if not self.redis:
            return False
        
        try:
            serialized_value = json.dumps(value, default=str)
            ttl = expire or self.default_ttl
            
            await self.redis.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.exists(key)
            return result > 0
            
        except Exception as e:
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for existing key."""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.expire(key, seconds)
            return result
            
        except Exception as e:
            logger.warning(f"Cache expire error for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for key (-1 if no expire, -2 if key doesn't exist)."""
        if not self.redis:
            return -2
        
        try:
            return await self.redis.ttl(key)
            
        except Exception as e:
            logger.warning(f"Cache TTL error for key {key}: {e}")
            return -2
    
    async def get_many(self, keys: list) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.redis or not keys:
            return {}
        
        try:
            values = await self.redis.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to deserialize cached value for key {key}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Cache get_many error: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], expire: int = None) -> bool:
        """Set multiple key-value pairs."""
        if not self.redis or not mapping:
            return False
        
        try:
            # Serialize all values
            serialized_mapping = {}
            for key, value in mapping.items():
                serialized_mapping[key] = json.dumps(value, default=str)
            
            # Set all values
            await self.redis.mset(serialized_mapping)
            
            # Set expiration if specified
            if expire:
                tasks = [self.redis.expire(key, expire) for key in mapping.keys()]
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set_many error: {e}")
            return False
    
    async def delete_many(self, keys: list) -> int:
        """Delete multiple keys from cache."""
        if not self.redis or not keys:
            return 0
        
        try:
            return await self.redis.delete(*keys)
            
        except Exception as e:
            logger.warning(f"Cache delete_many error: {e}")
            return 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self.redis:
            return 0
        
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.redis.delete(*keys)
            return 0
            
        except Exception as e:
            logger.warning(f"Cache clear_pattern error for pattern {pattern}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis:
            return {"status": "unavailable"}
        
        try:
            info = await self.redis.info()
            
            return {
                "status": "connected",
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
            
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100.0
    
    async def health_check(self) -> bool:
        """Perform cache health check."""
        if not self.redis:
            return False
        
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False


# Dependency function for FastAPI
async def get_cache_manager() -> CacheManager:
    """Get cache manager instance."""
    # In a real application, this would be managed by the application lifecycle
    cache_manager = CacheManager()
    if not cache_manager.redis:
        await cache_manager.initialize()
    return cache_manager
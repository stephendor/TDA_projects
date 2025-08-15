"""
Storage service for TDA backend.

This service handles point cloud data persistence, metadata storage,
data retrieval and caching, and cleanup/lifecycle management.
Currently implements in-memory storage with preparation for database integration.
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID, uuid4

from ..models import PointCloud, Point

# Configure logging
logger = logging.getLogger(__name__)

class StorageMetadata:
    """Metadata for stored point clouds."""
    
    def __init__(self, 
                 point_cloud_id: UUID,
                 name: str,
                 description: Optional[str] = None,
                 source: str = "unknown",
                 tags: List[str] = None,
                 upload_id: Optional[str] = None):
        self.point_cloud_id = point_cloud_id
        self.name = name
        self.description = description
        self.source = source
        self.tags = tags or []
        self.upload_id = upload_id
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.access_count = 0
        self.size_bytes = 0
        self.checksum: Optional[str] = None
        
    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
        
    def update_modified(self):
        """Update modification timestamp."""
        self.updated_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "point_cloud_id": str(self.point_cloud_id),
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "tags": self.tags,
            "upload_id": self.upload_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum
        }

class CacheEntry:
    """Cache entry for point cloud data."""
    
    def __init__(self, point_cloud: PointCloud, metadata: StorageMetadata):
        self.point_cloud = point_cloud
        self.metadata = metadata
        self.cached_at = datetime.utcnow()
        self.cache_hits = 0
        
    def hit(self):
        """Record cache hit."""
        self.cache_hits += 1
        self.metadata.update_access()

class StorageService:
    """Service for point cloud storage and retrieval."""
    
    def __init__(self, cache_max_size: int = 100, cache_ttl_hours: int = 24):
        """
        Initialize storage service.
        
        Args:
            cache_max_size: Maximum number of point clouds to keep in cache
            cache_ttl_hours: Time-to-live for cache entries in hours
        """
        # In-memory storage (will be replaced with database)
        self._storage: Dict[UUID, PointCloud] = {}
        self._metadata: Dict[UUID, StorageMetadata] = {}
        
        # Caching system
        self._cache: Dict[UUID, CacheEntry] = {}
        self.cache_max_size = cache_max_size
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Statistics
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "storage_size_bytes": 0
        }
        
        logger.info(f"Storage service initialized with cache size {cache_max_size}, TTL {cache_ttl_hours}h")
    
    async def store_point_cloud(self,
                               point_cloud: PointCloud,
                               name: str,
                               description: Optional[str] = None,
                               source: str = "api",
                               tags: List[str] = None,
                               upload_id: Optional[str] = None) -> UUID:
        """
        Store a point cloud with metadata.
        
        Args:
            point_cloud: Point cloud data to store
            name: Human-readable name
            description: Optional description
            source: Source of the data
            tags: Optional tags for categorization
            upload_id: Optional associated upload ID
            
        Returns:
            UUID of the stored point cloud
        """
        # Generate unique ID
        point_cloud_id = uuid4()
        
        # Create metadata
        metadata = StorageMetadata(
            point_cloud_id=point_cloud_id,
            name=name,
            description=description,
            source=source,
            tags=tags or [],
            upload_id=upload_id
        )
        
        # Calculate size and checksum
        try:
            serialized = pickle.dumps(point_cloud)
            metadata.size_bytes = len(serialized)
            metadata.checksum = self._calculate_checksum(serialized)
        except Exception as e:
            logger.warning(f"Could not calculate size/checksum: {e}")
            metadata.size_bytes = 0
        
        # Store in memory (will be database persistence)
        self._storage[point_cloud_id] = point_cloud
        self._metadata[point_cloud_id] = metadata
        
        # Add to cache
        self._add_to_cache(point_cloud_id, point_cloud, metadata)
        
        # Update statistics
        self.stats["total_stored"] += 1
        self.stats["storage_size_bytes"] += metadata.size_bytes
        
        logger.info(f"Stored point cloud {point_cloud_id} with {len(point_cloud.points)} points")
        return point_cloud_id
    
    async def retrieve_point_cloud(self, point_cloud_id: UUID) -> Optional[Tuple[PointCloud, StorageMetadata]]:
        """
        Retrieve a point cloud by ID.
        
        Args:
            point_cloud_id: Unique identifier
            
        Returns:
            Tuple of (PointCloud, StorageMetadata) or None if not found
        """
        # Check cache first
        if point_cloud_id in self._cache:
            cache_entry = self._cache[point_cloud_id]
            
            # Check if cache entry is still valid
            if datetime.utcnow() - cache_entry.cached_at < self.cache_ttl:
                cache_entry.hit()
                self.stats["cache_hits"] += 1
                self.stats["total_retrieved"] += 1
                logger.debug(f"Cache hit for point cloud {point_cloud_id}")
                return cache_entry.point_cloud, cache_entry.metadata
            else:
                # Cache entry expired
                del self._cache[point_cloud_id]
                logger.debug(f"Cache entry expired for {point_cloud_id}")
        
        # Cache miss - retrieve from storage
        self.stats["cache_misses"] += 1
        
        if point_cloud_id not in self._storage:
            logger.warning(f"Point cloud {point_cloud_id} not found in storage")
            return None
        
        point_cloud = self._storage[point_cloud_id]
        metadata = self._metadata[point_cloud_id]
        
        # Update access tracking
        metadata.update_access()
        
        # Add to cache
        self._add_to_cache(point_cloud_id, point_cloud, metadata)
        
        self.stats["total_retrieved"] += 1
        logger.debug(f"Retrieved point cloud {point_cloud_id} from storage")
        
        return point_cloud, metadata
    
    async def list_point_clouds(self,
                               page: int = 1,
                               size: int = 10,
                               name_filter: Optional[str] = None,
                               source_filter: Optional[str] = None,
                               tag_filter: Optional[str] = None,
                               sort_by: str = "created_at",
                               sort_desc: bool = True) -> Tuple[List[StorageMetadata], int]:
        """
        List stored point clouds with filtering and pagination.
        
        Args:
            page: Page number (1-based)
            size: Items per page
            name_filter: Filter by name substring
            source_filter: Filter by source
            tag_filter: Filter by tag
            sort_by: Sort field (created_at, updated_at, name, size_bytes)
            sort_desc: Sort descending
            
        Returns:
            Tuple of (metadata_list, total_count)
        """
        # Get all metadata
        all_metadata = list(self._metadata.values())
        
        # Apply filters
        if name_filter:
            all_metadata = [m for m in all_metadata if name_filter.lower() in m.name.lower()]
        
        if source_filter:
            all_metadata = [m for m in all_metadata if m.source == source_filter]
        
        if tag_filter:
            all_metadata = [m for m in all_metadata if tag_filter in m.tags]
        
        # Sort
        if sort_by == "created_at":
            all_metadata.sort(key=lambda m: m.created_at, reverse=sort_desc)
        elif sort_by == "updated_at":
            all_metadata.sort(key=lambda m: m.updated_at, reverse=sort_desc)
        elif sort_by == "name":
            all_metadata.sort(key=lambda m: m.name.lower(), reverse=sort_desc)
        elif sort_by == "size_bytes":
            all_metadata.sort(key=lambda m: m.size_bytes, reverse=sort_desc)
        elif sort_by == "access_count":
            all_metadata.sort(key=lambda m: m.access_count, reverse=sort_desc)
        
        total_count = len(all_metadata)
        
        # Paginate
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        page_metadata = all_metadata[start_idx:end_idx]
        
        return page_metadata, total_count
    
    async def update_point_cloud_metadata(self,
                                        point_cloud_id: UUID,
                                        name: Optional[str] = None,
                                        description: Optional[str] = None,
                                        tags: Optional[List[str]] = None) -> bool:
        """
        Update point cloud metadata.
        
        Args:
            point_cloud_id: Point cloud identifier
            name: New name
            description: New description
            tags: New tags list
            
        Returns:
            True if updated successfully, False if not found
        """
        if point_cloud_id not in self._metadata:
            return False
        
        metadata = self._metadata[point_cloud_id]
        
        if name is not None:
            metadata.name = name
        if description is not None:
            metadata.description = description
        if tags is not None:
            metadata.tags = tags
        
        metadata.update_modified()
        
        # Update cache if present
        if point_cloud_id in self._cache:
            self._cache[point_cloud_id].metadata = metadata
        
        logger.info(f"Updated metadata for point cloud {point_cloud_id}")
        return True
    
    async def delete_point_cloud(self, point_cloud_id: UUID) -> bool:
        """
        Delete a point cloud and its metadata.
        
        Args:
            point_cloud_id: Point cloud identifier
            
        Returns:
            True if deleted successfully, False if not found
        """
        if point_cloud_id not in self._storage:
            return False
        
        # Get size for statistics
        metadata = self._metadata.get(point_cloud_id)
        size_bytes = metadata.size_bytes if metadata else 0
        
        # Remove from storage
        del self._storage[point_cloud_id]
        del self._metadata[point_cloud_id]
        
        # Remove from cache
        if point_cloud_id in self._cache:
            del self._cache[point_cloud_id]
        
        # Update statistics
        self.stats["storage_size_bytes"] -= size_bytes
        
        logger.info(f"Deleted point cloud {point_cloud_id}")
        return True
    
    async def get_metadata(self, point_cloud_id: UUID) -> Optional[StorageMetadata]:
        """
        Get metadata for a point cloud without loading the data.
        
        Args:
            point_cloud_id: Point cloud identifier
            
        Returns:
            StorageMetadata or None if not found
        """
        metadata = self._metadata.get(point_cloud_id)
        if metadata:
            metadata.update_access()
        return metadata
    
    async def search_point_clouds(self, query: str, limit: int = 10) -> List[StorageMetadata]:
        """
        Search point clouds by name, description, or tags.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching metadata, sorted by relevance
        """
        query_lower = query.lower()
        results = []
        
        for metadata in self._metadata.values():
            score = 0
            
            # Name match (highest weight)
            if query_lower in metadata.name.lower():
                score += 10
                if metadata.name.lower().startswith(query_lower):
                    score += 5
            
            # Description match
            if metadata.description and query_lower in metadata.description.lower():
                score += 3
            
            # Tag match
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    score += 2
            
            # Source match
            if query_lower in metadata.source.lower():
                score += 1
            
            if score > 0:
                results.append((score, metadata))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [metadata for _, metadata in results[:limit]]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage service statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Calculate cache statistics
        cache_size = len(self._cache)
        cache_hit_rate = 0.0
        if self.stats["cache_hits"] + self.stats["cache_misses"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
        
        # Calculate storage statistics
        total_points = sum(len(pc.points) for pc in self._storage.values())
        avg_dimension = 0.0
        if self._storage:
            avg_dimension = sum(pc.dimension for pc in self._storage.values()) / len(self._storage)
        
        return {
            "storage": {
                "total_point_clouds": len(self._storage),
                "total_points": total_points,
                "average_dimension": round(avg_dimension, 2),
                "storage_size_bytes": self.stats["storage_size_bytes"],
                "storage_size_mb": round(self.stats["storage_size_bytes"] / (1024 * 1024), 2)
            },
            "cache": {
                "cache_size": cache_size,
                "cache_max_size": self.cache_max_size,
                "cache_hit_rate": round(cache_hit_rate * 100, 2),
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"]
            },
            "activity": {
                "total_stored": self.stats["total_stored"],
                "total_retrieved": self.stats["total_retrieved"]
            }
        }
    
    def cleanup_expired_cache(self):
        """Remove expired entries from cache."""
        current_time = datetime.utcnow()
        expired = []
        
        for pc_id, entry in self._cache.items():
            if current_time - entry.cached_at > self.cache_ttl:
                expired.append(pc_id)
        
        for pc_id in expired:
            del self._cache[pc_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired cache entries")
    
    def cleanup_old_data(self, max_age_days: int = 30, max_unused_days: int = 7):
        """
        Clean up old or unused data.
        
        Args:
            max_age_days: Maximum age for any data
            max_unused_days: Maximum days since last access
        """
        current_time = datetime.utcnow()
        max_age = timedelta(days=max_age_days)
        max_unused = timedelta(days=max_unused_days)
        
        to_delete = []
        
        for pc_id, metadata in self._metadata.items():
            age = current_time - metadata.created_at
            unused_time = current_time - metadata.last_accessed
            
            if age > max_age or unused_time > max_unused:
                to_delete.append(pc_id)
        
        for pc_id in to_delete:
            try:
                del self._storage[pc_id]
                del self._metadata[pc_id]
                if pc_id in self._cache:
                    del self._cache[pc_id]
            except KeyError:
                pass
        
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old/unused point clouds")
    
    def _add_to_cache(self, point_cloud_id: UUID, point_cloud: PointCloud, metadata: StorageMetadata):
        """Add entry to cache, managing size limits."""
        # Remove expired entries first
        self.cleanup_expired_cache()
        
        # If cache is full, remove least recently used entry
        if len(self._cache) >= self.cache_max_size:
            # Find LRU entry
            lru_id = min(self._cache.keys(), 
                        key=lambda k: self._cache[k].metadata.last_accessed)
            del self._cache[lru_id]
        
        # Add new entry
        self._cache[point_cloud_id] = CacheEntry(point_cloud, metadata)
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data integrity."""
        import hashlib
        return hashlib.md5(data).hexdigest()

# Global service instance
_storage_service: Optional[StorageService] = None

def get_storage_service() -> StorageService:
    """Get or create the global storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service

def configure_storage_service(cache_max_size: int = 100, cache_ttl_hours: int = 24) -> StorageService:
    """Configure and get the global storage service instance."""
    global _storage_service
    _storage_service = StorageService(cache_max_size, cache_ttl_hours)
    return _storage_service
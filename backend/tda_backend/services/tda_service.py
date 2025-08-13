"""
Comprehensive TDA Service - C++ Bridge and High-Level Operations.

This service provides the main interface between the FastAPI application and the
C++ TDA bindings. It handles all TDA computations, data conversions, caching,
error handling, and performance monitoring.

Features:
    - Point cloud creation and validation from multiple formats
    - TDA algorithm execution with C++ bindings (Vietoris-Rips, Alpha Complex, etc.)
    - Async/await support for FastAPI integration
    - Comprehensive error handling with FastAPI-compatible exceptions
    - Memory management and cleanup
    - Performance monitoring and statistics
    - Caching for frequently used computations
    - Multiple input/output format support
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from uuid import UUID, uuid4
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from pydantic import ValidationError

from ..models import (
    PointCloud, Point, PointCloudCreate,
    TDAComputationRequest, ComputationConfig, TDAAlgorithm, MetricType,
    FiltrationParameter, FiltrationMode,
    TDAResults, PersistenceDiagram, PersistencePair, BettiNumbers,
    Job, JobStatus, ExportFormat,
    ErrorResponse
)
from ..config import get_settings

# Type hints for mock C++ bindings
TDAArray = np.ndarray
TDAMatrix = np.ndarray


class TDAComputationError(Exception):
    """Custom exception for TDA computation errors."""
    
    def __init__(self, message: str, error_code: str = "TDA_COMPUTATION_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class TDAValidationError(Exception):
    """Custom exception for TDA data validation errors."""
    
    def __init__(self, message: str, error_code: str = "TDA_VALIDATION_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class TDAMemoryError(Exception):
    """Custom exception for TDA memory-related errors."""
    
    def __init__(self, message: str, error_code: str = "TDA_MEMORY_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class MockCPPBindings:
    """
    Mock implementation of C++ TDA bindings.
    
    This class simulates the interface of the real C++ bindings while the actual
    library is being developed. It provides realistic sample data and timing
    to test the service layer integration.
    """
    
    @staticmethod
    def create_point_cloud(points: np.ndarray, dimension: int) -> Dict[str, Any]:
        """Mock point cloud creation."""
        if points.shape[1] != dimension:
            raise ValueError(f"Point dimension mismatch: expected {dimension}, got {points.shape[1]}")
            
        return {
            "points": points,
            "dimension": dimension,
            "num_points": points.shape[0],
            "bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist()
            }
        }
    
    @staticmethod
    def compute_vietoris_rips(point_cloud: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Vietoris-Rips computation."""
        points = point_cloud["points"]
        max_dim = config.get("max_dimension", 2)
        
        # Simulate computation time
        time.sleep(0.1 + points.shape[0] * 0.001)
        
        # Generate mock persistence pairs
        diagrams = []
        
        # H0 (connected components) - always present
        h0_pairs = []
        num_components = min(4, points.shape[0])
        for i in range(num_components):
            if i == 0:  # One infinite component
                h0_pairs.append({"birth": 0.0, "death": float("inf"), "dimension": 0})
            else:
                death_time = np.random.exponential(0.5)
                h0_pairs.append({"birth": 0.0, "death": death_time, "dimension": 0})
        
        diagrams.append({"dimension": 0, "pairs": h0_pairs})
        
        # H1 (loops)
        if max_dim >= 1 and points.shape[0] >= 3:
            h1_pairs = []
            num_loops = np.random.poisson(2)
            for _ in range(num_loops):
                birth = np.random.exponential(0.3)
                persistence = np.random.exponential(0.4)
                death = birth + persistence
                h1_pairs.append({"birth": birth, "death": death, "dimension": 1})
            
            diagrams.append({"dimension": 1, "pairs": h1_pairs})
        
        # H2 (voids)
        if max_dim >= 2 and points.shape[0] >= 4:
            h2_pairs = []
            num_voids = np.random.poisson(1)
            for _ in range(num_voids):
                birth = np.random.exponential(0.5)
                persistence = np.random.exponential(0.2)
                death = birth + persistence
                h2_pairs.append({"birth": birth, "death": death, "dimension": 2})
            
            diagrams.append({"dimension": 2, "pairs": h2_pairs})
        
        # Generate mock Betti numbers
        filtration_steps = config.get("num_steps", 100)
        max_filtration = max([p["death"] for d in diagrams for p in d["pairs"] if p["death"] != float("inf")], default=2.0)
        
        filtration_values = np.linspace(0, max_filtration, filtration_steps).tolist()
        betti_numbers = {}
        
        for dim in range(max_dim + 1):
            betti_numbers[str(dim)] = []
            for filt_val in filtration_values:
                # Count features alive at this filtration value
                count = 0
                for d in diagrams:
                    if d["dimension"] == dim:
                        for pair in d["pairs"]:
                            if pair["birth"] <= filt_val < pair["death"]:
                                count += 1
                betti_numbers[str(dim)].append(count)
        
        return {
            "persistence_diagrams": diagrams,
            "betti_numbers": {
                "filtration_values": filtration_values,
                "betti_numbers": betti_numbers,
                "max_dimension": max_dim
            },
            "computation_time": time.time(),
            "metadata": {
                "algorithm": "vietoris_rips",
                "num_points": points.shape[0],
                "dimension": points.shape[1],
                "max_filtration": max_filtration
            }
        }
    
    @staticmethod
    def compute_alpha_complex(point_cloud: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Alpha Complex computation."""
        # Similar structure to Vietoris-Rips but with different characteristics
        result = MockCPPBindings.compute_vietoris_rips(point_cloud, config)
        result["metadata"]["algorithm"] = "alpha_complex"
        
        # Alpha complexes typically have fewer long-lived features
        for diagram in result["persistence_diagrams"]:
            for pair in diagram["pairs"]:
                if pair["death"] != float("inf"):
                    pair["death"] *= 0.8  # Shorter lifespans
        
        return result
    
    @staticmethod
    def compute_cech_complex(point_cloud: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Čech Complex computation."""
        result = MockCPPBindings.compute_vietoris_rips(point_cloud, config)
        result["metadata"]["algorithm"] = "cech_complex"
        
        # Čech complexes typically capture features at smaller scales
        for diagram in result["persistence_diagrams"]:
            for pair in diagram["pairs"]:
                if pair["death"] != float("inf"):
                    pair["birth"] *= 0.5
                    pair["death"] *= 0.6
        
        return result
    
    @staticmethod
    def compute_witness_complex(point_cloud: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Witness Complex computation."""
        result = MockCPPBindings.compute_vietoris_rips(point_cloud, config)
        result["metadata"]["algorithm"] = "witness_complex"
        return result


class PerformanceMonitor:
    """Performance monitoring and statistics collection."""
    
    def __init__(self):
        self.computation_stats = {
            "total_computations": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "algorithm_stats": {},
            "error_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.active_computations = {}
    
    def start_computation(self, job_id: str, algorithm: str) -> str:
        """Start tracking a computation."""
        computation_id = str(uuid4())
        self.active_computations[computation_id] = {
            "job_id": job_id,
            "algorithm": algorithm,
            "start_time": time.time(),
            "memory_start": 0  # Would track actual memory in real implementation
        }
        return computation_id
    
    def end_computation(self, computation_id: str, success: bool = True) -> Dict[str, Any]:
        """End tracking a computation and update statistics."""
        if computation_id not in self.active_computations:
            return {}
        
        comp = self.active_computations.pop(computation_id)
        duration = time.time() - comp["start_time"]
        
        # Update global stats
        self.computation_stats["total_computations"] += 1
        self.computation_stats["total_time"] += duration
        self.computation_stats["avg_time"] = (
            self.computation_stats["total_time"] / self.computation_stats["total_computations"]
        )
        
        if not success:
            self.computation_stats["error_count"] += 1
        
        # Update algorithm-specific stats
        alg = comp["algorithm"]
        if alg not in self.computation_stats["algorithm_stats"]:
            self.computation_stats["algorithm_stats"][alg] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "errors": 0
            }
        
        alg_stats = self.computation_stats["algorithm_stats"][alg]
        alg_stats["count"] += 1
        alg_stats["total_time"] += duration
        alg_stats["avg_time"] = alg_stats["total_time"] / alg_stats["count"]
        
        if not success:
            alg_stats["errors"] += 1
        
        return {
            "duration": duration,
            "algorithm": alg,
            "success": success,
            "memory_used": 0  # Would track actual memory
        }
    
    def cache_hit(self):
        """Record a cache hit."""
        self.computation_stats["cache_hits"] += 1
    
    def cache_miss(self):
        """Record a cache miss."""
        self.computation_stats["cache_misses"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        cache_total = self.computation_stats["cache_hits"] + self.computation_stats["cache_misses"]
        cache_rate = (
            self.computation_stats["cache_hits"] / cache_total if cache_total > 0 else 0.0
        )
        
        return {
            **self.computation_stats,
            "cache_hit_rate": cache_rate,
            "active_computations": len(self.active_computations)
        }


class TDAService:
    """
    Comprehensive TDA Service for C++ binding integration.
    
    This service provides high-level TDA operations, handles data conversion,
    manages computations, and integrates with FastAPI for async operations.
    """
    
    def __init__(self, settings = None):
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.max_concurrent_jobs
        )
        
        # In-memory cache for results (would use Redis in production)
        self.result_cache = {}
        self.cache_ttl = timedelta(seconds=self.settings.tda_result_cache_ttl)
        
        # Mock C++ bindings (replace with real bindings when available)
        self.cpp_bindings = MockCPPBindings()
        
        # Job tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("TDAService initialized with mock C++ bindings")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self.result_cache.clear()
        self.logger.info("TDAService cleaned up")
    
    def _validate_point_cloud(self, point_cloud: PointCloud) -> None:
        """Validate point cloud data."""
        try:
            if not point_cloud.points:
                raise TDAValidationError("Point cloud cannot be empty")
            
            if len(point_cloud.points) > self.settings.tda_max_points:
                raise TDAValidationError(
                    f"Point cloud exceeds maximum size: {len(point_cloud.points)} > {self.settings.tda_max_points}"
                )
            
            # Check dimension consistency
            expected_dim = point_cloud.dimension
            for i, point in enumerate(point_cloud.points):
                if len(point.coordinates) != expected_dim:
                    raise TDAValidationError(
                        f"Point {i} dimension mismatch: expected {expected_dim}, got {len(point.coordinates)}"
                    )
                
                # Check for NaN or infinite values
                for j, coord in enumerate(point.coordinates):
                    if not np.isfinite(coord):
                        raise TDAValidationError(
                            f"Point {i}, coordinate {j} is not finite: {coord}"
                        )
            
            self.logger.debug(f"Point cloud validation passed: {len(point_cloud.points)} points, dimension {expected_dim}")
            
        except Exception as e:
            if isinstance(e, TDAValidationError):
                raise
            raise TDAValidationError(f"Point cloud validation failed: {str(e)}")
    
    def _convert_point_cloud_to_numpy(self, point_cloud: PointCloud) -> np.ndarray:
        """Convert PointCloud model to NumPy array."""
        try:
            points_list = []
            for point in point_cloud.points:
                points_list.append(point.coordinates)
            
            return np.array(points_list, dtype=np.float64)
            
        except Exception as e:
            raise TDAComputationError(f"Failed to convert point cloud to NumPy array: {str(e)}")
    
    def _convert_cpp_result_to_model(self, cpp_result: Dict[str, Any], config: ComputationConfig) -> TDAResults:
        """Convert C++ computation result to Pydantic model."""
        try:
            # Convert persistence diagrams
            diagrams = []
            for diag_data in cpp_result["persistence_diagrams"]:
                pairs = []
                for pair_data in diag_data["pairs"]:
                    pair = PersistencePair(
                        dimension=pair_data["dimension"],
                        birth=pair_data["birth"],
                        death=pair_data["death"],
                        persistence=pair_data["death"] - pair_data["birth"] if pair_data["death"] != float("inf") else float("inf")
                    )
                    pairs.append(pair)
                
                diagram = PersistenceDiagram(
                    dimension=diag_data["dimension"],
                    pairs=pairs,
                    num_features=len(pairs)
                )
                diagrams.append(diagram)
            
            # Convert Betti numbers
            betti_data = cpp_result["betti_numbers"]
            betti_numbers = BettiNumbers(
                filtration_values=betti_data["filtration_values"],
                betti_numbers=betti_data["betti_numbers"],
                max_dimension=betti_data["max_dimension"]
            )
            
            # Create complete result
            result = TDAResults(
                persistence_diagrams=diagrams,
                betti_numbers=betti_numbers,
                computation_time=time.time() - cpp_result["computation_time"],
                algorithm_used=config.algorithm,
                parameters=config
            )
            
            return result
            
        except Exception as e:
            raise TDAComputationError(f"Failed to convert C++ result to model: {str(e)}")
    
    def _generate_cache_key(self, point_cloud: PointCloud, config: ComputationConfig) -> str:
        """Generate cache key for computation results."""
        # Create a deterministic hash of the input data
        import hashlib
        
        # Convert point cloud to consistent representation
        points_data = []
        for point in point_cloud.points:
            points_data.append(tuple(point.coordinates))
        
        # Create cache key components
        cache_data = {
            "points": points_data,
            "dimension": point_cloud.dimension,
            "algorithm": config.algorithm.value,
            "metric": config.metric.value,
            "max_dimension": config.filtration.max_dimension,
            "max_edge_length": config.filtration.max_edge_length,
            "num_steps": config.filtration.num_steps,
            "mode": config.filtration.mode.value
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[TDAResults]:
        """Get result from cache if available and not expired."""
        if cache_key not in self.result_cache:
            self.performance_monitor.cache_miss()
            return None
        
        cached_item = self.result_cache[cache_key]
        
        # Check if cache entry has expired
        if datetime.utcnow() - cached_item["timestamp"] > self.cache_ttl:
            del self.result_cache[cache_key]
            self.performance_monitor.cache_miss()
            return None
        
        self.performance_monitor.cache_hit()
        self.logger.debug(f"Cache hit for key: {cache_key[:16]}...")
        return cached_item["result"]
    
    def _cache_result(self, cache_key: str, result: TDAResults) -> None:
        """Cache computation result."""
        self.result_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.utcnow()
        }
        
        # Simple cache size management (would use LRU in production)
        if len(self.result_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.result_cache.keys(),
                key=lambda k: self.result_cache[k]["timestamp"]
            )[:100]
            
            for key in oldest_keys:
                del self.result_cache[key]
    
    async def create_point_cloud(self, request: PointCloudCreate) -> Dict[str, Any]:
        """Create and validate a point cloud."""
        try:
            self.logger.info(f"Creating point cloud: {request.name}")
            
            # Validate the point cloud
            self._validate_point_cloud(request.point_cloud)
            
            # Convert to NumPy for C++ bindings
            np_points = self._convert_point_cloud_to_numpy(request.point_cloud)
            
            # Create point cloud with C++ bindings
            cpp_point_cloud = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.cpp_bindings.create_point_cloud,
                np_points,
                request.point_cloud.dimension
            )
            
            result = {
                "id": str(uuid4()),
                "name": request.name,
                "description": request.description,
                "point_cloud": request.point_cloud,
                "created_at": datetime.utcnow(),
                "stats": {
                    "num_points": len(request.point_cloud.points),
                    "dimension": request.point_cloud.dimension,
                    "bounds": cpp_point_cloud["bounds"]
                }
            }
            
            self.logger.info(f"Point cloud created successfully: {result['id']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create point cloud: {str(e)}")
            if isinstance(e, (TDAValidationError, TDAComputationError)):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_code": e.error_code,
                        "message": str(e),
                        "details": getattr(e, 'details', {})
                    }
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "INTERNAL_ERROR",
                    "message": "Internal server error during point cloud creation"
                }
            )
    
    async def compute_tda(self, request: TDAComputationRequest) -> TDAResults:
        """Perform TDA computation on point cloud."""
        job_id = str(uuid4())
        
        try:
            self.logger.info(f"Starting TDA computation {job_id} with algorithm {request.config.algorithm}")
            
            # Get point cloud (from request or stored)
            point_cloud = request.point_cloud
            if point_cloud is None:
                # Would retrieve from database in real implementation
                raise TDAValidationError("Point cloud retrieval not implemented in mock service")
            
            # Validate inputs
            self._validate_point_cloud(point_cloud)
            
            # Check cache first
            cache_key = self._generate_cache_key(point_cloud, request.config)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached result for computation {job_id}")
                return cached_result
            
            # Start performance monitoring
            comp_id = self.performance_monitor.start_computation(job_id, request.config.algorithm.value)
            
            try:
                # Convert to format expected by C++ bindings
                np_points = self._convert_point_cloud_to_numpy(point_cloud)
                cpp_point_cloud = self.cpp_bindings.create_point_cloud(np_points, point_cloud.dimension)
                
                # Prepare configuration for C++ bindings
                cpp_config = {
                    "max_dimension": request.config.filtration.max_dimension,
                    "max_edge_length": request.config.filtration.max_edge_length,
                    "num_steps": request.config.filtration.num_steps,
                    "mode": request.config.filtration.mode.value,
                    "metric": request.config.metric.value,
                    "parallel": request.config.parallel,
                    "precision": request.config.precision
                }
                
                # Select appropriate algorithm and run computation
                algorithm_map = {
                    TDAAlgorithm.VIETORIS_RIPS: self.cpp_bindings.compute_vietoris_rips,
                    TDAAlgorithm.ALPHA_COMPLEX: self.cpp_bindings.compute_alpha_complex,
                    TDAAlgorithm.CECH_COMPLEX: self.cpp_bindings.compute_cech_complex,
                    TDAAlgorithm.WITNESS_COMPLEX: self.cpp_bindings.compute_witness_complex,
                }
                
                compute_func = algorithm_map[request.config.algorithm]
                
                # Run computation in thread pool
                cpp_result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    compute_func,
                    cpp_point_cloud,
                    cpp_config
                )
                
                # Convert result to Pydantic model
                result = self._convert_cpp_result_to_model(cpp_result, request.config)
                
                # Cache the result
                self._cache_result(cache_key, result)
                
                # End performance monitoring
                perf_stats = self.performance_monitor.end_computation(comp_id, success=True)
                
                self.logger.info(
                    f"TDA computation {job_id} completed successfully in {perf_stats['duration']:.2f}s"
                )
                
                return result
                
            except Exception as e:
                # End performance monitoring with error
                self.performance_monitor.end_computation(comp_id, success=False)
                raise
            
        except Exception as e:
            self.logger.error(f"TDA computation {job_id} failed: {str(e)}")
            
            if isinstance(e, (TDAValidationError, TDAComputationError, TDAMemoryError)):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_code": e.error_code,
                        "message": str(e),
                        "details": getattr(e, 'details', {})
                    }
                )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "COMPUTATION_FAILED",
                    "message": "TDA computation failed due to internal error"
                }
            )
    
    async def export_results(self, results: TDAResults, format: ExportFormat, 
                           include_metadata: bool = True) -> Union[str, bytes, Dict]:
        """Export TDA results to various formats."""
        try:
            self.logger.info(f"Exporting results to format: {format}")
            
            if format == ExportFormat.JSON:
                # Convert to JSON-serializable format
                export_data = results.dict()
                if not include_metadata:
                    export_data.pop('parameters', None)
                return json.dumps(export_data, indent=2, default=str)
            
            elif format == ExportFormat.CSV:
                # Convert persistence diagrams to CSV format
                csv_data = []
                for diagram in results.persistence_diagrams:
                    for pair in diagram.pairs:
                        csv_data.append({
                            'dimension': pair.dimension,
                            'birth': pair.birth,
                            'death': pair.death,
                            'persistence': pair.persistence
                        })
                
                df = pd.DataFrame(csv_data)
                return df.to_csv(index=False)
            
            elif format == ExportFormat.NUMPY:
                # Export as NumPy arrays in a dictionary
                export_dict = {}
                for diagram in results.persistence_diagrams:
                    dim = diagram.dimension
                    pairs_array = np.array([
                        [pair.birth, pair.death] for pair in diagram.pairs
                    ])
                    export_dict[f'dimension_{dim}'] = pairs_array
                
                return pickle.dumps(export_dict)
            
            elif format == ExportFormat.GUDHI:
                # Export in GUDHI-compatible format
                gudhi_format = []
                for diagram in results.persistence_diagrams:
                    dim_data = []
                    for pair in diagram.pairs:
                        dim_data.append([pair.birth, pair.death])
                    gudhi_format.append(np.array(dim_data))
                
                return pickle.dumps(gudhi_format)
            
            else:
                raise TDAComputationError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export results: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "EXPORT_FAILED",
                    "message": f"Failed to export results: {str(e)}"
                }
            )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_monitor.get_stats()
        
        # Add system information
        stats.update({
            "cache_size": len(self.result_cache),
            "active_jobs": len(self.active_jobs),
            "executor_threads": self.executor._max_workers,
            "memory_usage": {  # Would implement actual memory tracking
                "total_mb": 0,
                "available_mb": 0,
                "percent_used": 0
            }
        })
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "components": {},
            "performance": await self.get_performance_stats()
        }
        
        # Check C++ bindings
        try:
            # Test basic functionality
            test_points = np.array([[0.0, 0.0], [1.0, 1.0]])
            test_pc = self.cpp_bindings.create_point_cloud(test_points, 2)
            health["components"]["cpp_bindings"] = "healthy"
        except Exception as e:
            health["components"]["cpp_bindings"] = f"error: {str(e)}"
            health["status"] = "degraded"
        
        # Check thread pool
        if self.executor._shutdown:
            health["components"]["thread_pool"] = "shutdown"
            health["status"] = "unhealthy"
        else:
            health["components"]["thread_pool"] = "healthy"
        
        # Check cache
        health["components"]["result_cache"] = "healthy"
        
        return health
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'executor') and not self.executor._shutdown:
            self.executor.shutdown(wait=False)


# Global service instance (would be dependency-injected in production)
_tda_service: Optional[TDAService] = None


def get_tda_service() -> TDAService:
    """Get or create the global TDA service instance."""
    global _tda_service
    if _tda_service is None:
        _tda_service = TDAService()
    return _tda_service


@asynccontextmanager
async def tda_service_context():
    """Async context manager for TDA service."""
    service = get_tda_service()
    try:
        yield service
    finally:
        await service.cleanup()
"""
TDA Core API Routes

Provides endpoints for core topological data analysis computations
including persistent homology, mapper algorithm, and topology utilities.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field, validator

from ...core import PersistentHomologyAnalyzer, TopologyUtils
from ...core.mapper import MapperAnalyzer
from ...utils.database import get_db_manager
from ...utils.cache import get_cache_manager


router = APIRouter()


# Request/Response Models
class PointCloudRequest(BaseModel):
    """Request model for point cloud data."""
    data: List[List[float]] = Field(..., description="Point cloud data as array of coordinates")
    maxdim: int = Field(default=1, ge=0, le=3, description="Maximum homology dimension")
    distance_metric: str = Field(default="euclidean", description="Distance metric to use")
    
    @validator('data')
    def validate_data(cls, v):
        if len(v) < 3:
            raise ValueError("Point cloud must contain at least 3 points")
        if not all(len(point) == len(v[0]) for point in v):
            raise ValueError("All points must have the same dimension")
        return v


class PersistentHomologyRequest(PointCloudRequest):
    """Request model for persistent homology computation."""
    compute_features: bool = Field(default=True, description="Whether to compute feature vectors")
    cache_result: bool = Field(default=True, description="Whether to cache the result")


class MapperRequest(PointCloudRequest):
    """Request model for Mapper algorithm."""
    filter_function: str = Field(default="coordinate_0", description="Filter function to use")
    num_intervals: int = Field(default=10, ge=3, le=50, description="Number of intervals")
    overlap_percent: float = Field(default=0.3, ge=0.0, le=0.9, description="Overlap percentage")
    clustering_algorithm: str = Field(default="single", description="Clustering algorithm")


class TopologyAnalysisRequest(PointCloudRequest):
    """Request model for comprehensive topology analysis."""
    include_persistence: bool = Field(default=True, description="Include persistent homology")
    include_mapper: bool = Field(default=True, description="Include Mapper analysis")
    include_stats: bool = Field(default=True, description="Include topological statistics")


# Response Models
class PersistenceBar(BaseModel):
    """Model for a single persistence bar."""
    dimension: int
    birth: float
    death: float
    persistence: float


class PersistentHomologyResponse(BaseModel):
    """Response model for persistent homology results."""
    job_id: str
    status: str
    computation_time_ms: float
    persistence_diagram: List[List[PersistenceBar]]
    features: Optional[List[float]] = None
    summary_stats: Dict[str, Any]
    metadata: Dict[str, Any]


class MapperResponse(BaseModel):
    """Response model for Mapper algorithm results."""
    job_id: str
    status: str
    computation_time_ms: float
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    node_statistics: Dict[str, Any]
    graph_statistics: Dict[str, Any]
    metadata: Dict[str, Any]


class TopologyAnalysisResponse(BaseModel):
    """Response model for comprehensive topology analysis."""
    job_id: str
    status: str
    computation_time_ms: float
    persistent_homology: Optional[PersistentHomologyResponse] = None
    mapper: Optional[MapperResponse] = None
    topological_statistics: Dict[str, Any]
    data_summary: Dict[str, Any]


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_percent: Optional[float] = None
    error_message: Optional[str] = None


# Core TDA Endpoints
@router.post("/persistent-homology", response_model=PersistentHomologyResponse)
async def compute_persistent_homology(
    request: PersistentHomologyRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Compute persistent homology for point cloud data.
    
    This endpoint computes the persistent homology of the input point cloud,
    providing topological features that are stable under small perturbations.
    """
    import time
    start_time = time.time()
    
    # Generate job ID
    job_id = _generate_job_id(request.dict())
    
    try:
        # Convert input data to numpy array
        data = np.array(request.data)
        
        # Check cache first
        if request.cache_result:
            cached_result = await cache.get(f"ph:{job_id}")
            if cached_result:
                return PersistentHomologyResponse.parse_obj(cached_result)
        
        # Initialize analyzer
        analyzer = PersistentHomologyAnalyzer(
            maxdim=request.maxdim,
            distance_metric=request.distance_metric
        )
        
        # Compute persistent homology
        if request.compute_features:
            features = analyzer.fit_transform(data)
        else:
            analyzer.fit(data)
            features = None
        
        # Get persistence diagram
        persistence_diagram = analyzer.get_persistence_diagram()
        
        # Process persistence diagram into response format
        processed_diagram = []
        for dim, bars in enumerate(persistence_diagram):
            dim_bars = []
            for birth, death in bars:
                persistence = death - birth if death != np.inf else np.inf
                dim_bars.append(PersistenceBar(
                    dimension=dim,
                    birth=float(birth),
                    death=float(death) if death != np.inf else float('inf'),
                    persistence=float(persistence) if persistence != np.inf else float('inf')
                ))
            processed_diagram.append(dim_bars)
        
        # Compute summary statistics
        summary_stats = _compute_persistence_statistics(persistence_diagram)
        
        # Prepare response
        computation_time = (time.time() - start_time) * 1000
        
        response = PersistentHomologyResponse(
            job_id=job_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            persistence_diagram=processed_diagram,
            features=features.tolist() if features is not None else None,
            summary_stats=summary_stats,
            metadata={
                "input_points": len(data),
                "input_dimension": data.shape[1],
                "maxdim": request.maxdim,
                "distance_metric": request.distance_metric,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
        
        # Cache result
        if request.cache_result:
            background_tasks.add_task(
                cache.set, 
                f"ph:{job_id}", 
                response.dict(), 
                expire=3600  # 1 hour
            )
        
        # Store in database
        background_tasks.add_task(
            _store_computation_result,
            db, job_id, "persistent_homology", response.dict()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Persistent homology computation failed: {str(e)}"
        )


@router.post("/mapper", response_model=MapperResponse)
async def compute_mapper(
    request: MapperRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Compute Mapper algorithm for point cloud data.
    
    The Mapper algorithm creates a graph representation of the data
    that captures its topological structure and enables visualization
    of high-dimensional datasets.
    """
    import time
    start_time = time.time()
    
    job_id = _generate_job_id(request.dict())
    
    try:
        data = np.array(request.data)
        
        # Check cache
        cached_result = await cache.get(f"mapper:{job_id}")
        if cached_result:
            return MapperResponse.parse_obj(cached_result)
        
        # Initialize Mapper analyzer
        mapper = MapperAnalyzer(
            filter_function=request.filter_function,
            num_intervals=request.num_intervals,
            overlap_percent=request.overlap_percent,
            clustering_algorithm=request.clustering_algorithm
        )
        
        # Compute Mapper graph
        mapper_result = mapper.fit_transform(data)
        
        # Process results
        nodes = []
        for i, node in enumerate(mapper_result['nodes']):
            nodes.append({
                "id": i,
                "size": len(node['points']),
                "points": node['points'].tolist(),
                "filter_values": node['filter_values'].tolist(),
                "statistics": node.get('statistics', {})
            })
        
        edges = []
        for edge in mapper_result['edges']:
            edges.append({
                "source": int(edge['source']),
                "target": int(edge['target']),
                "weight": float(edge.get('weight', 1.0))
            })
        
        # Compute statistics
        node_stats = {
            "total_nodes": len(nodes),
            "avg_node_size": np.mean([node['size'] for node in nodes]),
            "max_node_size": max([node['size'] for node in nodes]),
            "min_node_size": min([node['size'] for node in nodes])
        }
        
        graph_stats = {
            "total_edges": len(edges),
            "connected_components": mapper_result.get('num_components', 1),
            "avg_degree": 2 * len(edges) / len(nodes) if nodes else 0
        }
        
        computation_time = (time.time() - start_time) * 1000
        
        response = MapperResponse(
            job_id=job_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            nodes=nodes,
            edges=edges,
            node_statistics=node_stats,
            graph_statistics=graph_stats,
            metadata={
                "input_points": len(data),
                "input_dimension": data.shape[1],
                "filter_function": request.filter_function,
                "num_intervals": request.num_intervals,
                "overlap_percent": request.overlap_percent,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
        
        # Cache and store
        background_tasks.add_task(cache.set, f"mapper:{job_id}", response.dict(), expire=3600)
        background_tasks.add_task(_store_computation_result, db, job_id, "mapper", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mapper computation failed: {str(e)}"
        )


@router.post("/analyze", response_model=TopologyAnalysisResponse)
async def comprehensive_topology_analysis(
    request: TopologyAnalysisRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Perform comprehensive topological analysis.
    
    This endpoint combines multiple TDA techniques to provide
    a complete topological characterization of the input data.
    """
    import time
    start_time = time.time()
    
    job_id = _generate_job_id(request.dict())
    
    try:
        data = np.array(request.data)
        
        # Initialize results
        ph_result = None
        mapper_result = None
        
        # Compute persistent homology if requested
        if request.include_persistence:
            ph_request = PersistentHomologyRequest(
                data=request.data,
                maxdim=request.maxdim,
                distance_metric=request.distance_metric,
                cache_result=False  # Handle caching at this level
            )
            ph_response = await compute_persistent_homology(
                ph_request, background_tasks, db, cache
            )
            ph_result = ph_response
        
        # Compute Mapper if requested
        if request.include_mapper:
            mapper_request = MapperRequest(
                data=request.data,
                maxdim=request.maxdim,
                distance_metric=request.distance_metric
            )
            mapper_response = await compute_mapper(
                mapper_request, background_tasks, db, cache
            )
            mapper_result = mapper_response
        
        # Compute topological statistics
        topo_stats = {}
        if request.include_stats:
            topo_stats = _compute_topological_statistics(data)
        
        # Data summary
        data_summary = {
            "num_points": len(data),
            "dimension": data.shape[1],
            "bounding_box": {
                "min": data.min(axis=0).tolist(),
                "max": data.max(axis=0).tolist()
            },
            "centroid": data.mean(axis=0).tolist(),
            "std_dev": data.std(axis=0).tolist()
        }
        
        computation_time = (time.time() - start_time) * 1000
        
        response = TopologyAnalysisResponse(
            job_id=job_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            persistent_homology=ph_result,
            mapper=mapper_result,
            topological_statistics=topo_stats,
            data_summary=data_summary
        )
        
        # Cache and store
        background_tasks.add_task(cache.set, f"analysis:{job_id}", response.dict(), expire=3600)
        background_tasks.add_task(_store_computation_result, db, job_id, "topology_analysis", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Topology analysis failed: {str(e)}"
        )


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db=Depends(get_db_manager)):
    """
    Get the status of a TDA computation job.
    
    Use this endpoint to check the progress of long-running computations.
    """
    try:
        # Query database for job status
        job_info = await db.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobStatusResponse(**job_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str, db=Depends(get_db_manager)):
    """
    Cancel a running TDA computation job.
    """
    try:
        success = await db.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or cannot be cancelled"
            )
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


# Utility functions
def _generate_job_id(request_data: dict) -> str:
    """Generate deterministic job ID from request data."""
    request_json = json.dumps(request_data, sort_keys=True)
    return hashlib.md5(request_json.encode()).hexdigest()


def _compute_persistence_statistics(persistence_diagram: List[np.ndarray]) -> Dict[str, Any]:
    """Compute summary statistics for persistence diagram."""
    stats = {}
    
    for dim, bars in enumerate(persistence_diagram):
        if len(bars) == 0:
            continue
            
        finite_bars = bars[bars[:, 1] != np.inf]
        
        if len(finite_bars) > 0:
            persistences = finite_bars[:, 1] - finite_bars[:, 0]
            stats[f"dim_{dim}"] = {
                "num_features": len(bars),
                "num_finite": len(finite_bars),
                "num_infinite": len(bars) - len(finite_bars),
                "max_persistence": float(persistences.max()) if len(persistences) > 0 else 0.0,
                "avg_persistence": float(persistences.mean()) if len(persistences) > 0 else 0.0,
                "total_persistence": float(persistences.sum()) if len(persistences) > 0 else 0.0
            }
    
    return stats


def _compute_topological_statistics(data: np.ndarray) -> Dict[str, Any]:
    """Compute basic topological statistics for point cloud."""
    from scipy.spatial.distance import pdist
    
    # Distance-based statistics
    distances = pdist(data)
    
    stats = {
        "distance_statistics": {
            "min_distance": float(distances.min()),
            "max_distance": float(distances.max()),
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std())
        },
        "data_characteristics": {
            "num_points": len(data),
            "dimension": data.shape[1],
            "density_estimate": len(data) / (distances.max() ** data.shape[1]) if distances.max() > 0 else 0
        }
    }
    
    return stats


async def _store_computation_result(db, job_id: str, job_type: str, result_data: dict):
    """Store computation result in database."""
    try:
        await db.store_computation_result(job_id, job_type, result_data)
    except Exception as e:
        # Log error but don't fail the request
        print(f"Failed to store computation result: {e}")  # Use proper logging
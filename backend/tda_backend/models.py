"""
Pydantic models for TDA backend API.

This module contains comprehensive data models for all TDA operations,
including point clouds, computation requests/responses, persistence pairs,
job management, and error handling.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


class JobStatus(str, Enum):
    """Enumeration of possible job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TDAAlgorithm(str, Enum):
    """Enumeration of available TDA algorithms."""
    VIETORIS_RIPS = "vietoris_rips"
    ALPHA_COMPLEX = "alpha_complex"
    CECH_COMPLEX = "cech_complex"
    WITNESS_COMPLEX = "witness_complex"


class MetricType(str, Enum):
    """Enumeration of distance metrics for point cloud analysis."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    MINKOWSKI = "minkowski"
    COSINE = "cosine"


class FiltrationMode(str, Enum):
    """Enumeration of filtration modes."""
    STANDARD = "standard"
    LOWER_STAR = "lower_star"
    UPPER_STAR = "upper_star"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(True, description="Indicates if the operation was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    message: Optional[str] = Field(None, description="Additional message or description")


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = Field(False, description="Always False for error responses")
    error_code: str = Field(..., description="Specific error code for client handling")
    error_type: str = Field(..., description="Type of error (validation, computation, system)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "timestamp": "2023-12-07T10:30:00Z",
                "message": "Invalid point cloud data format",
                "error_code": "INVALID_POINT_CLOUD",
                "error_type": "validation",
                "details": {"dimension": "Expected 2D or 3D points"}
            }
        }


# Point Cloud Models
class Point(BaseModel):
    """Individual point in n-dimensional space."""
    coordinates: List[float] = Field(..., description="Point coordinates", min_items=1, max_items=10)
    label: Optional[str] = Field(None, description="Optional point label")
    weight: Optional[float] = Field(1.0, description="Point weight for weighted complexes")

    @field_validator('coordinates')
    def validate_coordinates(cls, v):
        if len(v) < 1 or len(v) > 10:
            raise ValueError('Point must have between 1 and 10 dimensions')
        if any(not isinstance(coord, (int, float)) for coord in v):
            raise ValueError('All coordinates must be numeric')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "coordinates": [1.0, 2.5, -0.3],
                "label": "point_1",
                "weight": 1.0
            }
        }


class PointCloud(BaseModel):
    """Point cloud data structure."""
    points: List[Point] = Field(..., description="List of points in the cloud", min_items=1)
    dimension: int = Field(..., description="Dimensionality of the point cloud", ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @model_validator(mode='after')
    def validate_consistency(self):
        if not self.points:
            raise ValueError('Point cloud must contain at least one point')
        
        # Check dimension consistency
        for i, point in enumerate(self.points):
            if len(point.coordinates) != self.dimension:
                raise ValueError(f'Point {i} has dimension {len(point.coordinates)}, expected {self.dimension}')
        
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "points": [
                    {"coordinates": [0.0, 0.0], "label": "origin"},
                    {"coordinates": [1.0, 0.0], "label": "point1"},
                    {"coordinates": [0.0, 1.0], "label": "point2"}
                ],
                "dimension": 2,
                "metadata": {"source": "synthetic", "noise_level": 0.1}
            }
        }


class PointCloudCreate(BaseModel):
    """Request model for creating point clouds."""
    name: str = Field(..., description="Name identifier for the point cloud", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Optional description", max_length=500)
    point_cloud: PointCloud = Field(..., description="Point cloud data")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "sample_circle",
                "description": "Points sampled from a circle",
                "point_cloud": {
                    "points": [
                        {"coordinates": [1.0, 0.0]},
                        {"coordinates": [0.0, 1.0]},
                        {"coordinates": [-1.0, 0.0]},
                        {"coordinates": [0.0, -1.0]}
                    ],
                    "dimension": 2
                }
            }
        }


# TDA Computation Models
class FiltrationParameter(BaseModel):
    """Parameters for filtration construction."""
    max_dimension: int = Field(2, description="Maximum simplex dimension", ge=0, le=5)
    max_edge_length: Optional[float] = Field(None, description="Maximum edge length for filtration")
    num_steps: int = Field(100, description="Number of filtration steps", ge=10, le=1000)
    mode: FiltrationMode = Field(FiltrationMode.STANDARD, description="Filtration mode")

    class Config:
        json_schema_extra = {
            "example": {
                "max_dimension": 2,
                "max_edge_length": 2.0,
                "num_steps": 50,
                "mode": "standard"
            }
        }


class ComputationConfig(BaseModel):
    """Configuration for TDA computations."""
    algorithm: TDAAlgorithm = Field(..., description="TDA algorithm to use")
    metric: MetricType = Field(MetricType.EUCLIDEAN, description="Distance metric")
    filtration: FiltrationParameter = Field(default_factory=FiltrationParameter, description="Filtration parameters")
    parallel: bool = Field(True, description="Use parallel computation when available")
    precision: str = Field("double", description="Numerical precision", pattern="^(single|double)$")

    class Config:
        json_schema_extra = {
            "example": {
                "algorithm": "vietoris_rips",
                "metric": "euclidean",
                "filtration": {
                    "max_dimension": 2,
                    "max_edge_length": 1.5,
                    "num_steps": 100
                },
                "parallel": True,
                "precision": "double"
            }
        }


class TDAComputationRequest(BaseModel):
    """Request model for TDA computations."""
    point_cloud_id: Optional[UUID] = Field(None, description="ID of stored point cloud")
    point_cloud: Optional[PointCloud] = Field(None, description="Inline point cloud data")
    config: ComputationConfig = Field(..., description="Computation configuration")
    job_name: Optional[str] = Field(None, description="Optional job name", max_length=100)
    priority: int = Field(0, description="Job priority (higher = more priority)", ge=-10, le=10)

    @model_validator(mode='after')
    def validate_point_cloud_source(self):
        if not self.point_cloud_id and not self.point_cloud:
            raise ValueError('Either point_cloud_id or point_cloud must be provided')
        if self.point_cloud_id and self.point_cloud:
            raise ValueError('Provide either point_cloud_id or point_cloud, not both')
        
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "point_cloud": {
                    "points": [
                        {"coordinates": [0.0, 0.0]},
                        {"coordinates": [1.0, 1.0]},
                        {"coordinates": [2.0, 0.0]}
                    ],
                    "dimension": 2
                },
                "config": {
                    "algorithm": "vietoris_rips",
                    "metric": "euclidean",
                    "filtration": {"max_dimension": 1, "num_steps": 50}
                },
                "job_name": "triangle_analysis",
                "priority": 1
            }
        }


# Result Models
class PersistencePair(BaseModel):
    """Individual persistence pair."""
    dimension: int = Field(..., description="Homology dimension", ge=0)
    birth: float = Field(..., description="Birth time in filtration")
    death: float = Field(..., description="Death time in filtration")
    persistence: float = Field(..., description="Persistence (death - birth)")

    @model_validator(mode='after')
    def compute_persistence(self):
        self.persistence = self.death - self.birth if self.death != float('inf') else float('inf')
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "dimension": 1,
                "birth": 0.0,
                "death": 1.5,
                "persistence": 1.5
            }
        }


class PersistenceDiagram(BaseModel):
    """Persistence diagram for a specific dimension."""
    dimension: int = Field(..., description="Homology dimension", ge=0)
    pairs: List[PersistencePair] = Field(..., description="Persistence pairs")
    num_features: int = Field(..., description="Number of topological features", ge=0)

    @model_validator(mode='after')
    def validate_num_features(self):
        self.num_features = len(self.pairs)
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "dimension": 1,
                "pairs": [
                    {"dimension": 1, "birth": 0.0, "death": 1.5, "persistence": 1.5},
                    {"dimension": 1, "birth": 0.5, "death": 2.0, "persistence": 1.5}
                ],
                "num_features": 2
            }
        }


class BettiNumbers(BaseModel):
    """Betti numbers across filtration."""
    filtration_values: List[float] = Field(..., description="Filtration parameter values")
    betti_numbers: Dict[int, List[int]] = Field(..., description="Betti numbers by dimension")
    max_dimension: int = Field(..., description="Maximum computed dimension", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "filtration_values": [0.0, 0.5, 1.0, 1.5, 2.0],
                "betti_numbers": {
                    "0": [4, 3, 2, 1, 1],
                    "1": [0, 0, 1, 1, 0],
                    "2": [0, 0, 0, 0, 0]
                },
                "max_dimension": 2
            }
        }


class TDAResults(BaseModel):
    """Complete TDA computation results."""
    persistence_diagrams: List[PersistenceDiagram] = Field(..., description="Persistence diagrams by dimension")
    betti_numbers: BettiNumbers = Field(..., description="Betti numbers across filtration")
    computation_time: float = Field(..., description="Computation time in seconds", ge=0)
    algorithm_used: TDAAlgorithm = Field(..., description="Algorithm that was used")
    parameters: ComputationConfig = Field(..., description="Configuration parameters used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "persistence_diagrams": [
                    {
                        "dimension": 0,
                        "pairs": [{"dimension": 0, "birth": 0.0, "death": float('inf'), "persistence": float('inf')}],
                        "num_features": 1
                    },
                    {
                        "dimension": 1,
                        "pairs": [{"dimension": 1, "birth": 0.5, "death": 2.0, "persistence": 1.5}],
                        "num_features": 1
                    }
                ],
                "betti_numbers": {
                    "filtration_values": [0.0, 0.5, 1.0, 1.5, 2.0],
                    "betti_numbers": {"0": [4, 3, 2, 1, 1], "1": [0, 0, 1, 1, 0]},
                    "max_dimension": 1
                },
                "computation_time": 0.142,
                "algorithm_used": "vietoris_rips",
                "parameters": {
                    "algorithm": "vietoris_rips",
                    "metric": "euclidean",
                    "filtration": {"max_dimension": 1, "num_steps": 50}
                }
            }
        }


# Job Management Models
class Job(BaseModel):
    """Job information model."""
    id: UUID = Field(default_factory=uuid4, description="Unique job identifier")
    name: Optional[str] = Field(None, description="Optional job name")
    status: JobStatus = Field(JobStatus.PENDING, description="Current job status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    progress: float = Field(0.0, description="Job progress percentage", ge=0.0, le=100.0)
    priority: int = Field(0, description="Job priority", ge=-10, le=10)
    computation_request: TDAComputationRequest = Field(..., description="Original computation request")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    result_url: Optional[str] = Field(None, description="URL to retrieve results when completed")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "circle_analysis",
                "status": "running",
                "created_at": "2023-12-07T10:00:00Z",
                "started_at": "2023-12-07T10:01:00Z",
                "progress": 45.5,
                "priority": 1,
                "computation_request": {
                    "config": {
                        "algorithm": "vietoris_rips",
                        "filtration": {"max_dimension": 2}
                    }
                }
            }
        }


class JobCreate(BaseModel):
    """Request model for job creation."""
    computation_request: TDAComputationRequest = Field(..., description="TDA computation request")

    class Config:
        json_schema_extra = {
            "example": {
                "computation_request": {
                    "point_cloud": {
                        "points": [{"coordinates": [0.0, 0.0]}, {"coordinates": [1.0, 1.0]}],
                        "dimension": 2
                    },
                    "config": {
                        "algorithm": "vietoris_rips",
                        "filtration": {"max_dimension": 1}
                    }
                }
            }
        }


class JobResponse(BaseResponse):
    """Response model for job operations."""
    job: Job = Field(..., description="Job information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "message": "Job created successfully",
                "job": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "pending",
                    "created_at": "2023-12-07T10:30:00Z",
                    "progress": 0.0
                }
            }
        }


class JobListResponse(BaseResponse):
    """Response model for job listing."""
    jobs: List[Job] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs", ge=0)
    page: int = Field(1, description="Current page number", ge=1)
    size: int = Field(10, description="Page size", ge=1, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "jobs": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "completed",
                        "progress": 100.0
                    }
                ],
                "total": 1,
                "page": 1,
                "size": 10
            }
        }


# Results Response Models
class TDAResultsResponse(BaseResponse):
    """Response model for TDA computation results."""
    results: TDAResults = Field(..., description="TDA computation results")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "message": "Results retrieved successfully",
                "results": {
                    "persistence_diagrams": [
                        {
                            "dimension": 1,
                            "pairs": [{"dimension": 1, "birth": 0.5, "death": 2.0, "persistence": 1.5}],
                            "num_features": 1
                        }
                    ],
                    "computation_time": 0.142,
                    "algorithm_used": "vietoris_rips"
                }
            }
        }


# Export Models
class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    NUMPY = "numpy"
    GUDHI = "gudhi"


class ExportRequest(BaseModel):
    """Request model for exporting results."""
    format: ExportFormat = Field(..., description="Export format")
    include_metadata: bool = Field(True, description="Include computation metadata")
    compress: bool = Field(False, description="Compress the exported file")

    class Config:
        json_schema_extra = {
            "example": {
                "format": "json",
                "include_metadata": True,
                "compress": False
            }
        }


# Health and Status Models
class SystemHealth(BaseModel):
    """System health status."""
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str] = Field(..., description="Component status")
    active_jobs: int = Field(..., description="Number of active jobs", ge=0)
    queue_length: int = Field(..., description="Jobs in queue", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2023-12-07T10:30:00Z",
                "components": {
                    "database": "healthy",
                    "computation_engine": "healthy",
                    "storage": "healthy"
                },
                "active_jobs": 3,
                "queue_length": 7
            }
        }


class SystemHealthResponse(BaseResponse):
    """Response model for system health."""
    health: SystemHealth = Field(..., description="System health information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "health": {
                    "status": "healthy",
                    "version": "1.0.0",
                    "components": {"database": "healthy"},
                    "active_jobs": 2,
                    "queue_length": 5
                }
            }
        }


# Configuration Models  
class DatabaseConfig(BaseModel):
    """Database configuration model."""
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port", ge=1, le=65535)
    database: str = Field("tda_db", description="Database name")
    pool_size: int = Field(10, description="Connection pool size", ge=1, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "host": "localhost",
                "port": 5432,
                "database": "tda_backend",
                "pool_size": 20
            }
        }


class ComputationEngineConfig(BaseModel):
    """Computation engine configuration."""
    max_workers: int = Field(4, description="Maximum worker threads", ge=1, le=32)
    job_timeout: int = Field(3600, description="Job timeout in seconds", ge=60)
    max_queue_size: int = Field(100, description="Maximum queue size", ge=10, le=1000)
    enable_gpu: bool = Field(False, description="Enable GPU acceleration")

    class Config:
        json_schema_extra = {
            "example": {
                "max_workers": 8,
                "job_timeout": 1800,
                "max_queue_size": 50,
                "enable_gpu": True
            }
        }


class APIConfig(BaseModel):
    """API configuration model."""
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port", ge=1, le=65535)
    debug: bool = Field(False, description="Debug mode")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")

    class Config:
        json_schema_extra = {
            "example": {
                "host": "0.0.0.0",
                "port": 8080,
                "debug": False,
                "cors_origins": ["http://localhost:3000", "https://tda-frontend.com"]
            }
        }


class AppConfig(BaseModel):
    """Complete application configuration."""
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    computation: ComputationEngineConfig = Field(default_factory=ComputationEngineConfig)
    
    class Config:
        json_schema_extra = {
            "example": {
                "api": {"host": "0.0.0.0", "port": 8000, "debug": False},
                "database": {"host": "localhost", "port": 5432, "database": "tda_db"},
                "computation": {"max_workers": 4, "job_timeout": 3600}
            }
        }
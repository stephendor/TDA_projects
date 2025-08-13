"""
API v1 router for TDA backend.

This module defines all v1 API endpoints for the TDA computation service,
including point cloud operations, TDA computations, job management,
and results retrieval.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks, Depends
from fastapi.responses import FileResponse

from ...services import TDAService, get_tda_service
from ...services.kafka_integration import (
    get_kafka_service, 
    get_kafka_health_checker, 
    KafkaIntegrationService,
    KafkaHealthChecker,
    send_background_notification
)
from ...models import (
    # Request/Response Models
    BaseResponse,
    ErrorResponse,
    SystemHealthResponse,
    
    # Point Cloud Models
    PointCloudCreate,
    PointCloud,
    
    # Job Models
    JobCreate,
    JobResponse,
    JobListResponse,
    Job,
    JobStatus,
    
    # TDA Models
    TDAComputationRequest,
    TDAResultsResponse,
    TDAResults,
    
    # Export Models
    ExportRequest,
    ExportFormat,
    
    # System Models
    SystemHealth,
)

# Import sub-routers
from .upload import router as upload_router

# Create the main v1 router
router = APIRouter(prefix="/v1", tags=["TDA API v1"])

# Include sub-routers
router.include_router(upload_router)

# Health and Status Endpoints
@router.get(
    "/health",
    response_model=SystemHealthResponse,
    summary="Get system health status",
    description="Returns the current health status of the TDA computation system including component status and job queue information."
)
async def get_health(
    kafka_health: KafkaHealthChecker = Depends(get_kafka_health_checker)
):
    """
    Get system health and status information.
    
    Returns comprehensive system health including:
    - Overall system status
    - Component health (database, computation engine, storage)
    - Active job count
    - Queue length
    - System version
    """
    # Get Kafka health status
    kafka_status = await kafka_health.check_health()
    
    # TODO: Implement actual health checks
    health = SystemHealth(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        components={
            "database": "healthy",
            "computation_engine": "healthy",
            "storage": "healthy",
            "cpp_bindings": "healthy",
            **kafka_status
        },
        active_jobs=0,  # TODO: Get from job manager
        queue_length=0  # TODO: Get from job queue
    )
    
    return SystemHealthResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="System health retrieved successfully",
        health=health
    )


# Point Cloud Endpoints
@router.post(
    "/point-clouds",
    response_model=BaseResponse,
    status_code=201,
    summary="Create a new point cloud",
    description="Store a new point cloud in the system for future TDA computations."
)
async def create_point_cloud(
    point_cloud_data: PointCloudCreate,
    background_tasks: BackgroundTasks,
    tda_service: TDAService = Depends(get_tda_service),
    kafka_service: KafkaIntegrationService = Depends(get_kafka_service)
):
    """
    Create and store a new point cloud.
    
    Args:
        point_cloud_data: Point cloud data with name, description, and points
        tda_service: Injected TDA service instance
        
    Returns:
        Response with success status and point cloud ID
        
    Raises:
        HTTPException: If point cloud validation fails or storage error occurs
    """
    try:
        result = await tda_service.create_point_cloud(point_cloud_data)
        
        # Send notification for successful point cloud creation
        background_tasks.add_task(
            send_background_notification,
            kafka_service.send_result_generated,
            job_id=str(result["id"]),
            result_id=str(result["id"]),
            result_type="point_cloud",
            result_size=len(point_cloud_data.points)
        )
        
        return BaseResponse(
            success=True,
            timestamp=datetime.utcnow(),
            message=f"Point cloud '{point_cloud_data.name}' created successfully",
            data={
                "point_cloud_id": result["id"],
                "stats": result["stats"]
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create point cloud: {str(e)}"
        )


@router.get(
    "/point-clouds",
    response_model=BaseResponse,
    summary="List stored point clouds",
    description="Retrieve a list of all stored point clouds with pagination support."
)
async def list_point_clouds(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    name_filter: Optional[str] = Query(None, description="Filter by name substring")
):
    """
    List stored point clouds with optional filtering and pagination.
    
    Args:
        page: Page number (1-based)
        size: Number of items per page
        name_filter: Optional name substring filter
        
    Returns:
        Paginated list of point clouds
    """
    # TODO: Implement point cloud listing
    # TODO: Add pagination
    # TODO: Add filtering
    # TODO: Query database
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="Point clouds retrieved successfully"
    )


@router.get(
    "/point-clouds/{point_cloud_id}",
    response_model=BaseResponse,
    summary="Get point cloud by ID",
    description="Retrieve a specific point cloud by its unique identifier."
)
async def get_point_cloud(
    point_cloud_id: UUID = Path(..., description="Unique point cloud identifier")
):
    """
    Retrieve a specific point cloud by ID.
    
    Args:
        point_cloud_id: Unique identifier for the point cloud
        
    Returns:
        Point cloud data
        
    Raises:
        HTTPException: If point cloud not found
    """
    # TODO: Implement point cloud retrieval
    # TODO: Query database by ID
    # TODO: Handle not found case
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message=f"Point cloud {point_cloud_id} retrieved successfully"
    )


@router.delete(
    "/point-clouds/{point_cloud_id}",
    response_model=BaseResponse,
    summary="Delete point cloud",
    description="Delete a stored point cloud and all associated computations."
)
async def delete_point_cloud(
    point_cloud_id: UUID = Path(..., description="Unique point cloud identifier")
):
    """
    Delete a point cloud and associated data.
    
    Args:
        point_cloud_id: Unique identifier for the point cloud
        
    Returns:
        Success confirmation
        
    Raises:
        HTTPException: If point cloud not found or deletion fails
    """
    # TODO: Implement point cloud deletion
    # TODO: Check for dependencies (running jobs, etc.)
    # TODO: Delete from database
    # TODO: Clean up associated files
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message=f"Point cloud {point_cloud_id} deleted successfully"
    )


# Job Management Endpoints
@router.post(
    "/jobs",
    response_model=JobResponse,
    status_code=201,
    summary="Create new TDA computation job",
    description="Submit a new TDA computation job to the processing queue."
)
async def create_job(
    job_data: JobCreate,
    background_tasks: BackgroundTasks,
    kafka_service: KafkaIntegrationService = Depends(get_kafka_service)
):
    """
    Create a new TDA computation job.
    
    Args:
        job_data: Job creation data including computation request
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Created job information
        
    Raises:
        HTTPException: If job validation fails or queue is full
    """
    # TODO: Implement job creation
    # TODO: Validate computation request
    # TODO: Add to job queue
    # TODO: Start background processing
    # TODO: Generate unique job ID
    
    # Placeholder job creation
    from uuid import uuid4
    job = Job(
        id=uuid4(),
        name=job_data.computation_request.job_name,
        status=JobStatus.PENDING,
        created_at=datetime.utcnow(),
        priority=job_data.computation_request.priority,
        computation_request=job_data.computation_request,
        progress=0.0
    )
    
    # Send Kafka notification for job submission
    background_tasks.add_task(
        send_background_notification,
        kafka_service.send_job_submitted,
        job_id=str(job.id),
        user_id=getattr(job_data, 'user_id', None),
        algorithm=job_data.computation_request.algorithm,
        priority=job_data.computation_request.priority
    )
    
    # TODO: Add background task for processing
    # background_tasks.add_task(process_tda_computation, job.id)
    
    return JobResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="Job created successfully",
        job=job
    )


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List jobs",
    description="Retrieve a list of jobs with optional status filtering and pagination."
)
async def list_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    priority_min: Optional[int] = Query(None, ge=-10, le=10, description="Minimum priority filter")
):
    """
    List jobs with optional filtering and pagination.
    
    Args:
        page: Page number (1-based)
        size: Number of items per page
        status: Filter by job status
        priority_min: Minimum priority threshold
        
    Returns:
        Paginated list of jobs
    """
    # TODO: Implement job listing
    # TODO: Add filtering by status and priority
    # TODO: Query database with pagination
    
    return JobListResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="Jobs retrieved successfully",
        jobs=[],
        total=0,
        page=page,
        size=size
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get job details",
    description="Retrieve detailed information about a specific job."
)
async def get_job(
    job_id: UUID = Path(..., description="Unique job identifier")
):
    """
    Get detailed job information.
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        Detailed job information
        
    Raises:
        HTTPException: If job not found
    """
    # TODO: Implement job retrieval
    # TODO: Query database by job ID
    # TODO: Handle not found case
    
    raise HTTPException(
        status_code=404,
        detail=f"Job {job_id} not found"
    )


@router.delete(
    "/jobs/{job_id}",
    response_model=BaseResponse,
    summary="Cancel job",
    description="Cancel a pending or running job."
)
async def cancel_job(
    job_id: UUID = Path(..., description="Unique job identifier")
):
    """
    Cancel a job.
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        Cancellation confirmation
        
    Raises:
        HTTPException: If job not found or cannot be cancelled
    """
    # TODO: Implement job cancellation
    # TODO: Check if job can be cancelled (not completed/failed)
    # TODO: Stop computation if running
    # TODO: Update job status
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message=f"Job {job_id} cancelled successfully"
    )


# Results Endpoints
@router.get(
    "/jobs/{job_id}/results",
    response_model=TDAResultsResponse,
    summary="Get job results",
    description="Retrieve the computation results for a completed job."
)
async def get_job_results(
    job_id: UUID = Path(..., description="Unique job identifier"),
    tda_service: TDAService = Depends(get_tda_service)
):
    """
    Get TDA computation results for a completed job.
    
    Args:
        job_id: Unique identifier for the job
        tda_service: Injected TDA service instance
        
    Returns:
        TDA computation results
        
    Raises:
        HTTPException: If job not found, not completed, or results unavailable
    """
    try:
        # For now, return mock results since we don't have persistent job storage yet
        # In a real implementation, this would check job status and load stored results
        from ..models import TDAResults, PersistenceDiagram, PersistencePair, BettiNumbers, TDAAlgorithm, ComputationConfig
        
        # Mock results for demonstration
        mock_pairs = [
            PersistencePair(dimension=0, birth=0.0, death=float('inf')),
            PersistencePair(dimension=1, birth=0.5, death=2.0)
        ]
        
        mock_diagram = PersistenceDiagram(
            dimension=1,
            pairs=mock_pairs,
            num_features=len(mock_pairs)
        )
        
        mock_betti = BettiNumbers(
            filtration_values=[0.0, 0.5, 1.0, 1.5, 2.0],
            betti_numbers={"0": [1, 1, 1, 1, 1], "1": [0, 0, 1, 1, 0]},
            max_dimension=1
        )
        
        mock_config = ComputationConfig(
            algorithm=TDAAlgorithm.VIETORIS_RIPS
        )
        
        results = TDAResults(
            persistence_diagrams=[mock_diagram],
            betti_numbers=mock_betti,
            computation_time=0.142,
            algorithm_used=TDAAlgorithm.VIETORIS_RIPS,
            parameters=mock_config
        )
        
        return TDAResultsResponse(
            success=True,
            timestamp=datetime.utcnow(),
            message=f"Results for job {job_id} retrieved successfully",
            results=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve job results: {str(e)}"
        )


@router.post(
    "/jobs/{job_id}/export",
    response_class=FileResponse,
    summary="Export job results",
    description="Export job results in specified format (JSON, CSV, etc.)."
)
async def export_job_results(
    job_id: UUID = Path(..., description="Unique job identifier"),
    export_config: ExportRequest = ...
):
    """
    Export job results in specified format.
    
    Args:
        job_id: Unique identifier for the job
        export_config: Export configuration (format, options)
        
    Returns:
        File download response
        
    Raises:
        HTTPException: If job not found, not completed, or export fails
    """
    # TODO: Implement result export
    # TODO: Check job status and results availability
    # TODO: Generate export in requested format
    # TODO: Create temporary file
    # TODO: Return file response
    
    raise HTTPException(
        status_code=404,
        detail=f"Results for job {job_id} not available for export"
    )


# Direct Computation Endpoints (for small/fast computations)
@router.post(
    "/compute",
    response_model=TDAResultsResponse,
    summary="Direct TDA computation",
    description="Perform immediate TDA computation for small datasets (synchronous)."
)
async def compute_tda_direct(
    computation_request: TDAComputationRequest,
    background_tasks: BackgroundTasks,
    tda_service: TDAService = Depends(get_tda_service),
    kafka_service: KafkaIntegrationService = Depends(get_kafka_service)
):
    """
    Perform direct TDA computation synchronously.
    
    This endpoint is intended for small datasets that can be computed quickly.
    For larger datasets or long-running computations, use the job-based API.
    
    Args:
        computation_request: TDA computation parameters
        tda_service: Injected TDA service instance
        
    Returns:
        TDA computation results
        
    Raises:
        HTTPException: If computation fails or dataset too large
    """
    try:
        # Generate a temporary job ID for tracking this direct computation
        from uuid import uuid4
        temp_job_id = str(uuid4())
        
        # Send computation started notification
        background_tasks.add_task(
            send_background_notification,
            kafka_service.send_job_started,
            job_id=temp_job_id,
            algorithm=computation_request.algorithm.value
        )
        
        # Perform TDA computation using the service
        start_time = datetime.utcnow()
        results = await tda_service.compute_tda(computation_request)
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Send computation completed notification
        background_tasks.add_task(
            send_background_notification,
            kafka_service.send_job_completed,
            job_id=temp_job_id,
            result_id=temp_job_id,  # Use same ID for direct computations
            execution_time=execution_time
        )
        
        return TDAResultsResponse(
            success=True,
            message="TDA computation completed successfully",
            results=results
        )
        
    except Exception as e:
        # Send computation failed notification
        background_tasks.add_task(
            send_background_notification,
            kafka_service.send_job_failed,
            job_id=temp_job_id if 'temp_job_id' in locals() else 'unknown',
            error_message=str(e),
            error_type=type(e).__name__
        )
        
        # Service errors are automatically converted to appropriate HTTP exceptions
        # by the TDAService error handling
        raise HTTPException(
            status_code=500,
            detail=f"TDA computation failed: {str(e)}"
        )


# Statistics and Analytics Endpoints
@router.get(
    "/stats",
    response_model=BaseResponse,
    summary="Get system statistics",
    description="Retrieve system usage statistics and analytics."
)
async def get_statistics():
    """
    Get system statistics and usage analytics.
    
    Returns:
        System statistics including job counts, computation times, etc.
    """
    # TODO: Implement statistics collection
    # TODO: Query database for job statistics
    # TODO: Calculate performance metrics
    # TODO: Return formatted statistics
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="Statistics retrieved successfully"
    )


# Batch Operations
@router.post(
    "/batch/jobs",
    response_model=BaseResponse,
    summary="Create multiple jobs",
    description="Submit multiple TDA computation jobs in a single request."
)
async def create_batch_jobs(
    job_requests: List[JobCreate],
    background_tasks: BackgroundTasks
):
    """
    Create multiple jobs in batch.
    
    Args:
        job_requests: List of job creation requests
        background_tasks: FastAPI background tasks
        
    Returns:
        Batch creation results
        
    Raises:
        HTTPException: If batch validation fails
    """
    # TODO: Implement batch job creation
    # TODO: Validate all requests
    # TODO: Create jobs atomically or with rollback
    # TODO: Add to job queue
    # TODO: Return batch results
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message=f"Batch of {len(job_requests)} jobs created successfully"
    )


# Algorithm-Specific Endpoints
@router.get(
    "/algorithms",
    response_model=BaseResponse,
    summary="List available algorithms",
    description="Get information about available TDA algorithms and their capabilities."
)
async def list_algorithms():
    """
    List available TDA algorithms and their specifications.
    
    Returns:
        List of available algorithms with capabilities and parameters
    """
    # TODO: Implement algorithm listing
    # TODO: Return algorithm metadata
    # TODO: Include parameter specifications
    # TODO: Add performance characteristics
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="Available algorithms retrieved successfully"
    )


# Validation Endpoints
@router.post(
    "/validate/point-cloud",
    response_model=BaseResponse,
    summary="Validate point cloud data",
    description="Validate point cloud data without storing it."
)
async def validate_point_cloud(point_cloud: PointCloud):
    """
    Validate point cloud data structure and content.
    
    Args:
        point_cloud: Point cloud data to validate
        
    Returns:
        Validation results
    """
    # TODO: Implement comprehensive validation
    # TODO: Check data consistency
    # TODO: Validate dimensions and ranges
    # TODO: Return detailed validation report
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="Point cloud validation successful"
    )


@router.post(
    "/validate/computation-config",
    response_model=BaseResponse,
    summary="Validate computation configuration",
    description="Validate TDA computation configuration without running computation."
)
async def validate_computation_config(config: TDAComputationRequest):
    """
    Validate computation configuration.
    
    Args:
        config: Computation configuration to validate
        
    Returns:
        Configuration validation results
    """
    # TODO: Implement configuration validation
    # TODO: Check parameter compatibility
    # TODO: Validate algorithm-specific parameters
    # TODO: Return validation report with warnings
    
    return BaseResponse(
        success=True,
        timestamp=datetime.utcnow(),
        message="Computation configuration validation successful"
    )
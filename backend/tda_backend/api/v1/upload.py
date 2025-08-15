"""
Upload router for TDA backend API.

This module provides endpoints for uploading point cloud data files,
tracking upload progress, batch operations, and file management.
Supports multiple file formats with async processing.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Path, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...services.upload_service import get_upload_service, FileUploadService, UploadProgress, UploadStatus
from ...services.kafka_integration import get_kafka_service, KafkaIntegrationService, send_background_notification
from ...models import BaseResponse, ErrorResponse, PointCloud

# Configure logging
logger = logging.getLogger(__name__)

# Create upload router
router = APIRouter(prefix="/upload", tags=["File Upload"])

# Request/Response Models
class UploadConfigRequest(BaseModel):
    """Configuration for file upload processing."""
    has_header: bool = Field(True, description="Whether CSV/TSV files have header row")
    delimiter: str = Field(",", description="Delimiter for CSV/TSV files")
    dataset_name: str = Field("points", description="Dataset name for HDF5 files")
    coordinate_columns: Optional[List[str]] = Field(None, description="Specific columns to use as coordinates")
    label_column: Optional[str] = Field(None, description="Column to use for point labels")

    class Config:
        schema_extra = {
            "example": {
                "has_header": True,
                "delimiter": ",",
                "dataset_name": "coordinates",
                "coordinate_columns": ["x", "y", "z"],
                "label_column": "name"
            }
        }

class UploadResponse(BaseResponse):
    """Response for upload initiation."""
    upload_id: str = Field(..., description="Unique identifier for the upload")
    status: str = Field(..., description="Current upload status")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "message": "Upload started successfully",
                "upload_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing"
            }
        }

class UploadProgressResponse(BaseResponse):
    """Response for upload progress queries."""
    upload_id: str = Field(..., description="Upload identifier")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    created_at: datetime = Field(..., description="Upload creation time")
    updated_at: datetime = Field(..., description="Last update time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    file_info: Dict[str, Any] = Field(default_factory=dict, description="File information")
    point_cloud_info: Optional[Dict[str, Any]] = Field(None, description="Parsed point cloud information")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "upload_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "progress": 100.0,
                "created_at": "2023-12-07T10:25:00Z",
                "updated_at": "2023-12-07T10:30:00Z",
                "file_info": {
                    "filename": "points.csv",
                    "format": "csv"
                },
                "point_cloud_info": {
                    "dimension": 3,
                    "num_points": 1000
                }
            }
        }

class BatchUploadResponse(BaseResponse):
    """Response for batch upload operations."""
    upload_ids: List[str] = Field(..., description="List of upload identifiers")
    successful_uploads: int = Field(..., description="Number of successful uploads")
    failed_uploads: int = Field(..., description="Number of failed uploads")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "message": "Batch upload completed",
                "upload_ids": ["id1", "id2", "id3"],
                "successful_uploads": 2,
                "failed_uploads": 1
            }
        }

class UploadListResponse(BaseResponse):
    """Response for upload history listing."""
    uploads: List[UploadProgressResponse] = Field(..., description="List of uploads")
    total: int = Field(..., description="Total number of uploads")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "uploads": [],
                "total": 5,
                "page": 1,
                "size": 10
            }
        }

class FileValidationResponse(BaseResponse):
    """Response for file validation."""
    is_valid: bool = Field(..., description="Whether the file is valid")
    file_info: Dict[str, Any] = Field(..., description="File information")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2023-12-07T10:30:00Z",
                "message": "File validation completed",
                "is_valid": True,
                "file_info": {
                    "filename": "data.csv",
                    "format": "csv",
                    "estimated_points": 500
                },
                "validation_warnings": ["Header row detected but not specified in config"]
            }
        }

# Single File Upload Endpoints

@router.post(
    "/file",
    response_model=UploadResponse,
    status_code=202,
    summary="Upload single file",
    description="Upload a single point cloud data file for processing. Supports CSV, JSON, NumPy, and HDF5 formats."
)
async def upload_file(
    file: UploadFile = File(..., description="Point cloud data file"),
    config: Optional[str] = Form(None, description="JSON string of upload configuration"),
    upload_service: FileUploadService = Depends(get_upload_service),
    background_tasks: BackgroundTasks,
    kafka_service: KafkaIntegrationService = Depends(get_kafka_service)
):
    """
    Upload a single file containing point cloud data.
    
    The file will be validated, processed asynchronously, and converted to the
    internal PointCloud format. Use the returned upload_id to track progress.
    
    Args:
        file: Uploaded file (CSV, JSON, NumPy .npy, HDF5)
        config: Optional JSON configuration for parsing
        upload_service: Injected upload service
        
    Returns:
        Upload response with tracking ID
        
    Raises:
        HTTPException: If file validation fails or upload cannot be started
    """
    try:
        # Parse configuration if provided
        options = {}
        if config:
            import json
            try:
                config_dict = json.loads(config)
                upload_config = UploadConfigRequest(**config_dict)
                options = upload_config.dict()
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration format: {str(e)}"
                )
        
        # Start the upload
        upload_id, progress = await upload_service.start_upload(file, options)
        
        logger.info(f"Started upload {upload_id} for file {file.filename}")
        
        # Send Kafka notification for file upload
        background_tasks.add_task(
            send_background_notification,
            kafka_service.send_file_uploaded,
            file_id=str(upload_id),
            filename=file.filename,
            file_size=file.size or 0,
            content_type=file.content_type or "unknown"
        )
        
        return UploadResponse(
            success=True,
            message=f"Upload started for file '{file.filename}'",
            upload_id=upload_id,
            status=progress.status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@router.post(
    "/batch",
    response_model=BatchUploadResponse,
    status_code=202,
    summary="Upload multiple files",
    description="Upload multiple point cloud files in a single request."
)
async def upload_batch(
    files: List[UploadFile] = File(..., description="List of point cloud data files"),
    config: Optional[str] = Form(None, description="JSON string of upload configuration"),
    upload_service: FileUploadService = Depends(get_upload_service)
):
    """
    Upload multiple files in batch.
    
    All files will be processed with the same configuration. Each file
    gets its own upload_id for individual progress tracking.
    
    Args:
        files: List of uploaded files
        config: Optional JSON configuration applied to all files
        upload_service: Injected upload service
        
    Returns:
        Batch upload response with all upload IDs
    """
    if len(files) > 10:  # Reasonable batch size limit
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 10 files per batch."
        )
    
    try:
        # Parse configuration
        options = {}
        if config:
            import json
            try:
                config_dict = json.loads(config)
                upload_config = UploadConfigRequest(**config_dict)
                options = upload_config.dict()
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration format: {str(e)}"
                )
        
        upload_ids = []
        successful_uploads = 0
        failed_uploads = 0
        
        for file in files:
            try:
                upload_id, progress = await upload_service.start_upload(file, options)
                upload_ids.append(upload_id)
                
                if progress.status != UploadStatus.FAILED:
                    successful_uploads += 1
                else:
                    failed_uploads += 1
                    
            except Exception as e:
                logger.error(f"Failed to start upload for {file.filename}: {e}")
                failed_uploads += 1
        
        logger.info(f"Batch upload started: {successful_uploads} successful, {failed_uploads} failed")
        
        return BatchUploadResponse(
            success=True,
            message=f"Batch upload completed: {successful_uploads} successful, {failed_uploads} failed",
            upload_ids=upload_ids,
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch upload failed: {str(e)}"
        )

# Progress Tracking Endpoints

@router.get(
    "/status/{upload_id}",
    response_model=UploadProgressResponse,
    summary="Get upload progress",
    description="Get the current status and progress of a file upload."
)
async def get_upload_status(
    upload_id: str = Path(..., description="Upload identifier"),
    upload_service: FileUploadService = Depends(get_upload_service)
):
    """
    Get upload progress and status.
    
    Args:
        upload_id: Unique upload identifier
        upload_service: Injected upload service
        
    Returns:
        Upload progress information
        
    Raises:
        HTTPException: If upload ID not found
    """
    progress = upload_service.get_upload_progress(upload_id)
    if not progress:
        raise HTTPException(
            status_code=404,
            detail=f"Upload {upload_id} not found"
        )
    
    # Extract point cloud info from completed uploads
    point_cloud_info = None
    if progress.status == UploadStatus.COMPLETED and hasattr(progress, 'data') and progress.data:
        point_cloud_data = progress.data.get('point_cloud')
        if point_cloud_data:
            point_cloud_info = {
                'dimension': point_cloud_data.dimension,
                'num_points': len(point_cloud_data.points),
                'metadata': point_cloud_data.metadata
            }
    
    return UploadProgressResponse(
        success=True,
        message="Upload status retrieved successfully",
        upload_id=upload_id,
        status=progress.status,
        progress=progress.progress,
        created_at=progress.created_at,
        updated_at=progress.updated_at,
        error_message=progress.error_message,
        file_info=progress.file_info,
        point_cloud_info=point_cloud_info
    )

@router.get(
    "/list",
    response_model=UploadListResponse,
    summary="List uploads",
    description="Get a paginated list of uploads with optional status filtering."
)
async def list_uploads(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    status: Optional[str] = Query(None, description="Filter by upload status"),
    upload_service: FileUploadService = Depends(get_upload_service)
):
    """
    List uploads with pagination and filtering.
    
    Args:
        page: Page number (1-based)
        size: Items per page
        status: Optional status filter
        upload_service: Injected upload service
        
    Returns:
        Paginated list of uploads
    """
    all_uploads = list(upload_service.upload_progress.values())
    
    # Filter by status if specified
    if status:
        all_uploads = [u for u in all_uploads if u.status == status]
    
    # Sort by creation time (newest first)
    all_uploads.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    page_uploads = all_uploads[start_idx:end_idx]
    
    # Convert to response format
    upload_responses = []
    for progress in page_uploads:
        point_cloud_info = None
        if progress.status == UploadStatus.COMPLETED and hasattr(progress, 'data') and progress.data:
            point_cloud_data = progress.data.get('point_cloud')
            if point_cloud_data:
                point_cloud_info = {
                    'dimension': point_cloud_data.dimension,
                    'num_points': len(point_cloud_data.points)
                }
        
        upload_responses.append(UploadProgressResponse(
            success=True,
            upload_id=progress.upload_id,
            status=progress.status,
            progress=progress.progress,
            created_at=progress.created_at,
            updated_at=progress.updated_at,
            error_message=progress.error_message,
            file_info=progress.file_info,
            point_cloud_info=point_cloud_info
        ))
    
    return UploadListResponse(
        success=True,
        message=f"Retrieved {len(upload_responses)} uploads",
        uploads=upload_responses,
        total=len(all_uploads),
        page=page,
        size=size
    )

@router.delete(
    "/cancel/{upload_id}",
    response_model=BaseResponse,
    summary="Cancel upload",
    description="Cancel a pending or processing upload."
)
async def cancel_upload(
    upload_id: str = Path(..., description="Upload identifier"),
    upload_service: FileUploadService = Depends(get_upload_service)
):
    """
    Cancel an upload.
    
    Args:
        upload_id: Unique upload identifier
        upload_service: Injected upload service
        
    Returns:
        Cancellation confirmation
        
    Raises:
        HTTPException: If upload not found or cannot be cancelled
    """
    progress = upload_service.get_upload_progress(upload_id)
    if not progress:
        raise HTTPException(
            status_code=404,
            detail=f"Upload {upload_id} not found"
        )
    
    if progress.status in [UploadStatus.COMPLETED, UploadStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel upload with status '{progress.status}'"
        )
    
    success = await upload_service.cancel_upload(upload_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel upload"
        )
    
    return BaseResponse(
        success=True,
        message=f"Upload {upload_id} cancelled successfully"
    )

# File Validation Endpoints

@router.post(
    "/validate",
    response_model=FileValidationResponse,
    summary="Validate file",
    description="Validate a file without uploading it for processing."
)
async def validate_file(
    file: UploadFile = File(..., description="File to validate"),
    upload_service: FileUploadService = Depends(get_upload_service)
):
    """
    Validate a file without processing it.
    
    This endpoint performs validation checks on the uploaded file
    without storing it or converting it to PointCloud format.
    
    Args:
        file: File to validate
        upload_service: Injected upload service
        
    Returns:
        Validation results and file information
    """
    try:
        result = await upload_service.validate_file(file)
        
        file_info = {}
        validation_warnings = []
        
        if result.success:
            file_info = result.data or {}
            
            # Add estimated file analysis
            try:
                await file.seek(0)
                first_lines = await file.read(1024)
                await file.seek(0)
                
                if file_info.get('detected_format') == 'csv':
                    # Estimate number of lines/points
                    line_count = first_lines.decode('utf-8', errors='ignore').count('\n')
                    if line_count > 0:
                        file_info['estimated_points_in_sample'] = line_count
                        validation_warnings.append(f"Detected approximately {line_count} lines in first 1KB")
                
            except Exception as e:
                logger.warning(f"Could not analyze file content: {e}")
                validation_warnings.append("Could not analyze file content")
        
        return FileValidationResponse(
            success=True,
            message="File validation completed",
            is_valid=result.success,
            file_info=file_info,
            validation_warnings=validation_warnings
        )
        
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return FileValidationResponse(
            success=False,
            message=f"Validation failed: {str(e)}",
            is_valid=False,
            file_info={},
            validation_warnings=[str(e)]
        )

@router.get(
    "/formats",
    response_model=BaseResponse,
    summary="Get supported formats",
    description="Get information about supported file formats and their configurations."
)
async def get_supported_formats():
    """
    Get information about supported file formats.
    
    Returns:
        Information about supported formats and their configuration options
    """
    format_info = {
        "csv": {
            "description": "Comma-separated values format",
            "extensions": [".csv", ".txt"],
            "configuration_options": {
                "has_header": "Whether the first row contains column headers",
                "delimiter": "Column separator character (default: comma)",
                "coordinate_columns": "Specific columns to use as coordinates",
                "label_column": "Column to use for point labels"
            },
            "example_config": {
                "has_header": True,
                "delimiter": ",",
                "coordinate_columns": ["x", "y", "z"],
                "label_column": "name"
            }
        },
        "tsv": {
            "description": "Tab-separated values format",
            "extensions": [".tsv"],
            "configuration_options": {
                "has_header": "Whether the first row contains column headers",
                "coordinate_columns": "Specific columns to use as coordinates",
                "label_column": "Column to use for point labels"
            },
            "example_config": {
                "has_header": True,
                "coordinate_columns": ["x", "y", "z"]
            }
        },
        "json": {
            "description": "JavaScript Object Notation format",
            "extensions": [".json"],
            "supported_structures": [
                "Array of coordinate arrays: [[x1,y1,z1], [x2,y2,z2], ...]",
                "Array of point objects: [{coordinates: [x,y,z], label: 'name'}, ...]",
                "Object with 'points' or 'coordinates' field"
            ],
            "configuration_options": {},
            "example_structure": {
                "points": [
                    {"coordinates": [1.0, 2.0, 3.0], "label": "point1"},
                    {"coordinates": [4.0, 5.0, 6.0], "label": "point2"}
                ]
            }
        },
        "numpy": {
            "description": "NumPy binary array format",
            "extensions": [".npy"],
            "requirements": [
                "Array must be 1D (single point) or 2D (multiple points)",
                "Shape: (n_points, n_dimensions)",
                "All values must be numeric"
            ],
            "configuration_options": {}
        },
        "hdf5": {
            "description": "Hierarchical Data Format 5",
            "extensions": [".h5", ".hdf5"],
            "configuration_options": {
                "dataset_name": "Name of dataset containing point data (default: 'points')"
            },
            "common_dataset_names": ["points", "coordinates", "data", "point_cloud"],
            "example_config": {
                "dataset_name": "coordinates"
            }
        }
    }
    
    return BaseResponse(
        success=True,
        message="Supported file formats retrieved successfully",
        data={
            "formats": format_info,
            "max_file_size": "100MB",
            "max_batch_size": 10,
            "general_requirements": [
                "All coordinate values must be numeric",
                "Point dimensions must be consistent within a file",
                "Files must not exceed 100MB in size"
            ]
        }
    )

# Point Cloud Data Retrieval

@router.get(
    "/data/{upload_id}",
    response_model=BaseResponse,
    summary="Get parsed point cloud",
    description="Retrieve the parsed point cloud data from a completed upload."
)
async def get_upload_data(
    upload_id: str = Path(..., description="Upload identifier"),
    format: str = Query("json", description="Response format (json, summary)"),
    upload_service: FileUploadService = Depends(get_upload_service)
):
    """
    Get the parsed point cloud data from a completed upload.
    
    Args:
        upload_id: Upload identifier
        format: Response format ('json' for full data, 'summary' for overview)
        upload_service: Injected upload service
        
    Returns:
        Point cloud data or summary
        
    Raises:
        HTTPException: If upload not found or not completed
    """
    progress = upload_service.get_upload_progress(upload_id)
    if not progress:
        raise HTTPException(
            status_code=404,
            detail=f"Upload {upload_id} not found"
        )
    
    if progress.status != UploadStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Upload not completed. Current status: {progress.status}"
        )
    
    if not hasattr(progress, 'data') or not progress.data or 'point_cloud' not in progress.data:
        raise HTTPException(
            status_code=404,
            detail="Point cloud data not available"
        )
    
    point_cloud = progress.data['point_cloud']
    
    if format == "summary":
        # Return summary information only
        return BaseResponse(
            success=True,
            message="Point cloud summary retrieved successfully",
            data={
                "upload_id": upload_id,
                "dimension": point_cloud.dimension,
                "num_points": len(point_cloud.points),
                "metadata": point_cloud.metadata,
                "coordinate_ranges": {
                    f"dim_{i}": {
                        "min": min(p.coordinates[i] for p in point_cloud.points),
                        "max": max(p.coordinates[i] for p in point_cloud.points)
                    }
                    for i in range(point_cloud.dimension)
                }
            }
        )
    else:
        # Return full point cloud data
        return BaseResponse(
            success=True,
            message="Point cloud data retrieved successfully",
            data={
                "upload_id": upload_id,
                "point_cloud": point_cloud.dict()
            }
        )

# Management Endpoints

@router.post(
    "/cleanup",
    response_model=BaseResponse,
    summary="Cleanup old uploads",
    description="Clean up old upload records and temporary files."
)
async def cleanup_uploads(
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age in hours"),
    upload_service: FileUploadService = Depends(get_upload_service)
):
    """
    Clean up old upload records.
    
    Args:
        max_age_hours: Maximum age in hours for keeping upload records
        upload_service: Injected upload service
        
    Returns:
        Cleanup results
    """
    try:
        upload_service.cleanup_old_uploads(max_age_hours)
        return BaseResponse(
            success=True,
            message=f"Cleanup completed for uploads older than {max_age_hours} hours"
        )
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )
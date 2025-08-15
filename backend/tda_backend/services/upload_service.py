"""
Upload service for TDA backend.

This service handles file uploads for point cloud data with support for multiple
file formats, validation, security checks, and async processing. It includes
temporary file management, data parsing and conversion to PointCloud format,
and comprehensive error handling.
"""

import asyncio
import json
import logging
import mimetypes
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import aiofiles
import h5py
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

from ..models import PointCloud, Point
from .storage_service import get_storage_service

# Configure logging
logger = logging.getLogger(__name__)

# File upload configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.csv', '.json', '.npy', '.h5', '.hdf5', '.txt', '.tsv'}
ALLOWED_MIME_TYPES = {
    'text/csv', 'application/json', 'text/plain', 
    'application/octet-stream', 'application/x-hdf'
}

# Supported file formats
class FileFormat:
    CSV = 'csv'
    JSON = 'json'
    NUMPY = 'numpy'
    HDF5 = 'hdf5'
    TSV = 'tsv'

class UploadStatus:
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

class UploadResult:
    """Result container for upload operations."""
    
    def __init__(self, success: bool, message: str, data: Optional[Dict] = None, error: Optional[str] = None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.error = error
        self.timestamp = datetime.utcnow()

class UploadProgress:
    """Progress tracking for upload operations."""
    
    def __init__(self, upload_id: str):
        self.upload_id = upload_id
        self.status = UploadStatus.PENDING
        self.progress = 0.0
        self.message = "Upload initiated"
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.error_message: Optional[str] = None
        self.file_info: Dict[str, Any] = {}
        
    def update(self, status: str = None, progress: float = None, message: str = None, error: str = None):
        """Update progress information."""
        if status:
            self.status = status
        if progress is not None:
            self.progress = progress
        if message:
            self.message = message
        if error:
            self.error_message = error
        self.updated_at = datetime.utcnow()

class FileUploadService:
    """Service for handling file uploads and processing."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tda_uploads_")
        self.upload_progress: Dict[str, UploadProgress] = {}
        self.active_uploads: Dict[str, asyncio.Task] = {}
        logger.info(f"Upload service initialized with temp directory: {self.temp_dir}")
    
    async def validate_file(self, file: UploadFile) -> UploadResult:
        """
        Validate uploaded file for security and format compatibility.
        
        Args:
            file: Uploaded file object
            
        Returns:
            UploadResult with validation status
        """
        try:
            # Check file size
            if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
                return UploadResult(
                    success=False,
                    message=f"File size {file.size} exceeds maximum allowed size {MAX_FILE_SIZE}",
                    error="FILE_TOO_LARGE"
                )
            
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                return UploadResult(
                    success=False,
                    message=f"File extension '{file_ext}' not allowed. Supported: {ALLOWED_EXTENSIONS}",
                    error="INVALID_FILE_EXTENSION"
                )
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file.filename)
            if mime_type and mime_type not in ALLOWED_MIME_TYPES:
                return UploadResult(
                    success=False,
                    message=f"MIME type '{mime_type}' not allowed",
                    error="INVALID_MIME_TYPE"
                )
            
            # Basic security checks - scan for potentially malicious patterns
            await file.seek(0)
            first_chunk = await file.read(1024)
            await file.seek(0)
            
            # Check for binary executable signatures
            if first_chunk.startswith(b'\x7fELF') or first_chunk.startswith(b'MZ'):
                return UploadResult(
                    success=False,
                    message="File appears to be an executable binary",
                    error="SECURITY_RISK"
                )
            
            return UploadResult(
                success=True,
                message="File validation successful",
                data={
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'detected_format': self._detect_file_format(file.filename)
                }
            )
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return UploadResult(
                success=False,
                message=f"Validation error: {str(e)}",
                error="VALIDATION_ERROR"
            )
    
    def _detect_file_format(self, filename: str) -> str:
        """Detect file format from filename extension."""
        ext = Path(filename).suffix.lower()
        format_map = {
            '.csv': FileFormat.CSV,
            '.tsv': FileFormat.TSV,
            '.json': FileFormat.JSON,
            '.npy': FileFormat.NUMPY,
            '.h5': FileFormat.HDF5,
            '.hdf5': FileFormat.HDF5,
            '.txt': FileFormat.CSV  # Assume text files are CSV-like
        }
        return format_map.get(ext, FileFormat.CSV)
    
    async def save_temporary_file(self, file: UploadFile, upload_id: str) -> str:
        """
        Save uploaded file to temporary location.
        
        Args:
            file: Uploaded file object
            upload_id: Unique upload identifier
            
        Returns:
            Path to saved temporary file
        """
        file_ext = Path(file.filename).suffix
        temp_filename = f"{upload_id}_{file.filename}"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        
        try:
            async with aiofiles.open(temp_path, 'wb') as temp_file:
                await file.seek(0)
                while chunk := await file.read(8192):  # 8KB chunks
                    await temp_file.write(chunk)
            
            logger.info(f"File saved to temporary location: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temporary file: {e}")
            # Clean up on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    async def parse_file_to_point_cloud(self, file_path: str, format_type: str, options: Dict = None) -> UploadResult:
        """
        Parse uploaded file and convert to PointCloud format.
        
        Args:
            file_path: Path to uploaded file
            format_type: Detected file format
            options: Additional parsing options
            
        Returns:
            UploadResult with parsed PointCloud or error
        """
        options = options or {}
        
        try:
            if format_type == FileFormat.CSV:
                return await self._parse_csv_file(file_path, options)
            elif format_type == FileFormat.TSV:
                return await self._parse_csv_file(file_path, {**options, 'delimiter': '\t'})
            elif format_type == FileFormat.JSON:
                return await self._parse_json_file(file_path, options)
            elif format_type == FileFormat.NUMPY:
                return await self._parse_numpy_file(file_path, options)
            elif format_type == FileFormat.HDF5:
                return await self._parse_hdf5_file(file_path, options)
            else:
                return UploadResult(
                    success=False,
                    message=f"Unsupported file format: {format_type}",
                    error="UNSUPPORTED_FORMAT"
                )
                
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return UploadResult(
                success=False,
                message=f"File parsing error: {str(e)}",
                error="PARSING_ERROR"
            )
    
    async def _parse_csv_file(self, file_path: str, options: Dict) -> UploadResult:
        """Parse CSV file into PointCloud."""
        delimiter = options.get('delimiter', ',')
        has_header = options.get('has_header', True)
        
        try:
            # Use pandas for robust CSV parsing
            df = pd.read_csv(
                file_path, 
                delimiter=delimiter,
                header=0 if has_header else None,
                dtype=float  # Try to parse all data as float
            )
            
            if df.empty:
                return UploadResult(
                    success=False,
                    message="CSV file is empty",
                    error="EMPTY_FILE"
                )
            
            # Convert DataFrame to points
            points = []
            dimension = len(df.columns)
            
            for idx, row in df.iterrows():
                coordinates = row.values.tolist()
                # Handle NaN values
                if any(pd.isna(coord) for coord in coordinates):
                    continue
                
                points.append(Point(
                    coordinates=coordinates,
                    label=f"point_{idx}"
                ))
            
            if not points:
                return UploadResult(
                    success=False,
                    message="No valid points found in CSV file",
                    error="NO_VALID_POINTS"
                )
            
            point_cloud = PointCloud(
                points=points,
                dimension=dimension,
                metadata={
                    'source': 'csv_upload',
                    'original_filename': Path(file_path).name,
                    'total_points': len(points),
                    'columns': list(df.columns) if has_header else [f"dim_{i}" for i in range(dimension)]
                }
            )
            
            return UploadResult(
                success=True,
                message=f"Successfully parsed CSV with {len(points)} points in {dimension}D",
                data={'point_cloud': point_cloud}
            )
            
        except Exception as e:
            return UploadResult(
                success=False,
                message=f"CSV parsing error: {str(e)}",
                error="CSV_PARSING_ERROR"
            )
    
    async def _parse_json_file(self, file_path: str, options: Dict) -> UploadResult:
        """Parse JSON file into PointCloud."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of coordinates or points
                points_data = data
            elif isinstance(data, dict):
                if 'points' in data:
                    points_data = data['points']
                elif 'coordinates' in data:
                    points_data = data['coordinates']
                else:
                    return UploadResult(
                        success=False,
                        message="JSON must contain 'points' or 'coordinates' field",
                        error="INVALID_JSON_STRUCTURE"
                    )
            else:
                return UploadResult(
                    success=False,
                    message="JSON must be array or object",
                    error="INVALID_JSON_TYPE"
                )
            
            points = []
            dimension = None
            
            for idx, point_data in enumerate(points_data):
                if isinstance(point_data, list):
                    coordinates = point_data
                    label = f"point_{idx}"
                elif isinstance(point_data, dict):
                    coordinates = point_data.get('coordinates', point_data.get('coords', []))
                    label = point_data.get('label', point_data.get('name', f"point_{idx}"))
                else:
                    continue
                
                if not coordinates:
                    continue
                
                # Validate dimension consistency
                if dimension is None:
                    dimension = len(coordinates)
                elif len(coordinates) != dimension:
                    logger.warning(f"Skipping point {idx}: dimension mismatch ({len(coordinates)} vs {dimension})")
                    continue
                
                points.append(Point(
                    coordinates=coordinates,
                    label=label
                ))
            
            if not points:
                return UploadResult(
                    success=False,
                    message="No valid points found in JSON file",
                    error="NO_VALID_POINTS"
                )
            
            point_cloud = PointCloud(
                points=points,
                dimension=dimension,
                metadata={
                    'source': 'json_upload',
                    'original_filename': Path(file_path).name,
                    'total_points': len(points)
                }
            )
            
            return UploadResult(
                success=True,
                message=f"Successfully parsed JSON with {len(points)} points in {dimension}D",
                data={'point_cloud': point_cloud}
            )
            
        except json.JSONDecodeError as e:
            return UploadResult(
                success=False,
                message=f"Invalid JSON format: {str(e)}",
                error="INVALID_JSON"
            )
        except Exception as e:
            return UploadResult(
                success=False,
                message=f"JSON parsing error: {str(e)}",
                error="JSON_PARSING_ERROR"
            )
    
    async def _parse_numpy_file(self, file_path: str, options: Dict) -> UploadResult:
        """Parse NumPy .npy file into PointCloud."""
        try:
            data = np.load(file_path)
            
            # Ensure data is 2D (points x dimensions)
            if data.ndim == 1:
                # Single point, reshape to 2D
                data = data.reshape(1, -1)
            elif data.ndim != 2:
                return UploadResult(
                    success=False,
                    message=f"NumPy array must be 1D or 2D, got {data.ndim}D",
                    error="INVALID_NUMPY_SHAPE"
                )
            
            num_points, dimension = data.shape
            
            points = []
            for idx in range(num_points):
                coordinates = data[idx].tolist()
                points.append(Point(
                    coordinates=coordinates,
                    label=f"point_{idx}"
                ))
            
            point_cloud = PointCloud(
                points=points,
                dimension=dimension,
                metadata={
                    'source': 'numpy_upload',
                    'original_filename': Path(file_path).name,
                    'total_points': num_points,
                    'original_shape': list(data.shape)
                }
            )
            
            return UploadResult(
                success=True,
                message=f"Successfully parsed NumPy array with {num_points} points in {dimension}D",
                data={'point_cloud': point_cloud}
            )
            
        except Exception as e:
            return UploadResult(
                success=False,
                message=f"NumPy parsing error: {str(e)}",
                error="NUMPY_PARSING_ERROR"
            )
    
    async def _parse_hdf5_file(self, file_path: str, options: Dict) -> UploadResult:
        """Parse HDF5 file into PointCloud."""
        dataset_name = options.get('dataset_name', 'points')
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Try to find the dataset
                if dataset_name in f:
                    data = f[dataset_name][:]
                else:
                    # Try to find any dataset that looks like point data
                    possible_names = ['points', 'coordinates', 'data', 'point_cloud']
                    found_dataset = None
                    
                    for name in possible_names:
                        if name in f:
                            found_dataset = name
                            break
                    
                    if found_dataset:
                        data = f[found_dataset][:]
                    else:
                        # List available datasets
                        available = list(f.keys())
                        return UploadResult(
                            success=False,
                            message=f"Could not find point data. Available datasets: {available}",
                            error="DATASET_NOT_FOUND",
                            data={'available_datasets': available}
                        )
                
                # Process similar to NumPy
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif data.ndim != 2:
                    return UploadResult(
                        success=False,
                        message=f"HDF5 dataset must be 1D or 2D, got {data.ndim}D",
                        error="INVALID_HDF5_SHAPE"
                    )
                
                num_points, dimension = data.shape
                
                points = []
                for idx in range(num_points):
                    coordinates = data[idx].tolist()
                    points.append(Point(
                        coordinates=coordinates,
                        label=f"point_{idx}"
                    ))
                
                point_cloud = PointCloud(
                    points=points,
                    dimension=dimension,
                    metadata={
                        'source': 'hdf5_upload',
                        'original_filename': Path(file_path).name,
                        'total_points': num_points,
                        'dataset_name': dataset_name,
                        'original_shape': list(data.shape)
                    }
                )
                
                return UploadResult(
                    success=True,
                    message=f"Successfully parsed HDF5 with {num_points} points in {dimension}D",
                    data={'point_cloud': point_cloud}
                )
                
        except Exception as e:
            return UploadResult(
                success=False,
                message=f"HDF5 parsing error: {str(e)}",
                error="HDF5_PARSING_ERROR"
            )
    
    async def process_upload_async(self, upload_id: str, file_path: str, format_type: str, options: Dict = None, auto_store: bool = True) -> None:
        """
        Process file upload asynchronously.
        
        Args:
            upload_id: Unique upload identifier
            file_path: Path to uploaded file
            format_type: Detected file format
            options: Additional processing options
            auto_store: Whether to automatically store parsed point cloud
        """
        progress = self.upload_progress.get(upload_id)
        if not progress:
            logger.error(f"No progress tracking found for upload {upload_id}")
            return
        
        try:
            # Update status to processing
            progress.update(status=UploadStatus.PROCESSING, progress=10.0, message="Starting file processing")
            
            # Parse the file
            progress.update(progress=30.0, message="Parsing file data")
            result = await self.parse_file_to_point_cloud(file_path, format_type, options)
            
            if result.success:
                # Optionally store the point cloud in storage service
                point_cloud_id = None
                if auto_store and 'point_cloud' in result.data:
                    try:
                        progress.update(progress=80.0, message="Storing point cloud data")
                        storage_service = get_storage_service()
                        point_cloud = result.data['point_cloud']
                        
                        # Generate name from filename or upload ID
                        filename = progress.file_info.get('filename', f'upload_{upload_id}')
                        name = Path(filename).stem
                        
                        point_cloud_id = await storage_service.store_point_cloud(
                            point_cloud,
                            name=name,
                            description=f"Point cloud from uploaded file: {filename}",
                            source="file_upload",
                            tags=["uploaded", format_type],
                            upload_id=upload_id
                        )
                        
                        # Add storage info to result data
                        result.data['point_cloud_id'] = str(point_cloud_id)
                        logger.info(f"Stored point cloud {point_cloud_id} from upload {upload_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to store point cloud for upload {upload_id}: {e}")
                        # Don't fail the upload, just log the error
                
                progress.update(
                    status=UploadStatus.COMPLETED,
                    progress=100.0,
                    message=f"Upload completed successfully: {result.message}"
                )
                # Store the result in progress data
                progress.data = result.data
            else:
                progress.update(
                    status=UploadStatus.FAILED,
                    message="Upload processing failed",
                    error=result.error or result.message
                )
                
        except Exception as e:
            logger.error(f"Async upload processing error for {upload_id}: {e}")
            progress.update(
                status=UploadStatus.FAILED,
                message="Upload processing failed with unexpected error",
                error=str(e)
            )
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {e}")
            
            # Remove from active uploads
            if upload_id in self.active_uploads:
                del self.active_uploads[upload_id]
    
    async def start_upload(self, file: UploadFile, options: Dict = None, auto_store: bool = True) -> Tuple[str, UploadProgress]:
        """
        Start an upload process.
        
        Args:
            file: Uploaded file object
            options: Processing options
            auto_store: Whether to automatically store parsed point cloud
            
        Returns:
            Tuple of (upload_id, progress_tracker)
        """
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Validate file
        validation_result = await self.validate_file(file)
        if not validation_result.success:
            # Create failed progress entry
            progress = UploadProgress(upload_id)
            progress.update(
                status=UploadStatus.FAILED,
                error=validation_result.error,
                message=validation_result.message
            )
            self.upload_progress[upload_id] = progress
            return upload_id, progress
        
        # Create progress tracker
        progress = UploadProgress(upload_id)
        progress.file_info = {
            'filename': file.filename,
            'content_type': file.content_type,
            'format': validation_result.data['detected_format']
        }
        self.upload_progress[upload_id] = progress
        
        try:
            # Save temporary file
            progress.update(progress=5.0, message="Saving uploaded file")
            temp_path = await self.save_temporary_file(file, upload_id)
            
            # Start async processing
            format_type = validation_result.data['detected_format']
            task = asyncio.create_task(
                self.process_upload_async(upload_id, temp_path, format_type, options, auto_store)
            )
            self.active_uploads[upload_id] = task
            
            progress.update(
                status=UploadStatus.PROCESSING,
                progress=10.0,
                message="Upload queued for processing"
            )
            
        except Exception as e:
            logger.error(f"Error starting upload {upload_id}: {e}")
            progress.update(
                status=UploadStatus.FAILED,
                error=str(e),
                message="Failed to start upload processing"
            )
        
        return upload_id, progress
    
    def get_upload_progress(self, upload_id: str) -> Optional[UploadProgress]:
        """Get upload progress by ID."""
        return self.upload_progress.get(upload_id)
    
    async def cancel_upload(self, upload_id: str) -> bool:
        """
        Cancel an active upload.
        
        Args:
            upload_id: Upload identifier
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        progress = self.upload_progress.get(upload_id)
        if not progress:
            return False
        
        # Cancel the task if it's running
        if upload_id in self.active_uploads:
            task = self.active_uploads[upload_id]
            task.cancel()
            del self.active_uploads[upload_id]
        
        # Update progress
        progress.update(
            status=UploadStatus.CANCELLED,
            message="Upload cancelled by user"
        )
        
        return True
    
    def cleanup_old_uploads(self, max_age_hours: int = 24):
        """Clean up old upload records and temporary files."""
        current_time = datetime.utcnow()
        to_remove = []
        
        for upload_id, progress in self.upload_progress.items():
            age_hours = (current_time - progress.created_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_remove.append(upload_id)
        
        for upload_id in to_remove:
            del self.upload_progress[upload_id]
            if upload_id in self.active_uploads:
                self.active_uploads[upload_id].cancel()
                del self.active_uploads[upload_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old upload records")
    
    def __del__(self):
        """Cleanup temporary directory on service destruction."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {self.temp_dir}: {e}")

# Global service instance
_upload_service: Optional[FileUploadService] = None

def get_upload_service() -> FileUploadService:
    """Get or create the global upload service instance."""
    global _upload_service
    if _upload_service is None:
        _upload_service = FileUploadService()
    return _upload_service
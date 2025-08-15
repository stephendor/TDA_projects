# TDA Backend Upload System

This document describes the comprehensive data upload system implemented for the TDA backend, including file upload handlers, data validation, async processing, and storage integration.

## Overview

The upload system provides production-ready file upload capabilities for point cloud data with support for multiple formats, security validation, async processing, and seamless integration with the TDA computation pipeline.

### Key Features

- **Multiple Format Support**: CSV, JSON, NumPy .npy, HDF5
- **Security Validation**: File size limits, extension checking, MIME type validation
- **Async Processing**: Background file processing with progress tracking
- **Auto-Storage**: Automatic integration with storage service
- **Batch Operations**: Support for multiple file uploads
- **Progress Tracking**: Real-time upload status and progress monitoring
- **Error Handling**: Comprehensive error reporting and recovery
- **Cleanup Management**: Automatic cleanup of temporary files and old records

## Architecture

### Components

1. **Upload Service** (`upload_service.py`)
   - File validation and security checks
   - Multi-format parsing (CSV, JSON, NumPy, HDF5)
   - Async processing with progress tracking
   - Temporary file management

2. **Upload Router** (`api/v1/upload.py`)
   - RESTful API endpoints for upload operations
   - Single and batch file upload
   - Progress monitoring and status tracking
   - File validation endpoints

3. **Storage Service** (`storage_service.py`)
   - Point cloud persistence and metadata storage
   - Caching and retrieval optimization
   - Search and filtering capabilities
   - Lifecycle management

### Workflow

```
File Upload → Validation → Temporary Storage → Async Processing → Point Cloud Parsing → Storage Service → Cache
     ↓              ↓              ↓                    ↓                      ↓                  ↓           ↓
Progress Track  Security Check  File Save      Background Process     Format Convert    Metadata Store  Ready for TDA
```

## API Endpoints

### Upload Operations

- `POST /v1/upload/file` - Upload single file
- `POST /v1/upload/batch` - Upload multiple files
- `POST /v1/upload/validate` - Validate file without processing

### Progress Tracking

- `GET /v1/upload/status/{upload_id}` - Get upload progress
- `GET /v1/upload/list` - List uploads with filtering
- `DELETE /v1/upload/cancel/{upload_id}` - Cancel upload

### Data Retrieval

- `GET /v1/upload/data/{upload_id}` - Get parsed point cloud data
- `GET /v1/upload/formats` - Get supported format information

### Management

- `POST /v1/upload/cleanup` - Clean up old uploads

## Supported File Formats

### CSV/TSV Files
```
Configuration Options:
- has_header: Whether first row contains headers
- delimiter: Column separator (default: comma)
- coordinate_columns: Specific columns for coordinates
- label_column: Column for point labels

Example:
x,y,z,label
1.0,2.0,3.0,point1
4.0,5.0,6.0,point2
```

### JSON Files
```json
{
  "points": [
    {"coordinates": [1.0, 2.0, 3.0], "label": "point1"},
    {"coordinates": [4.0, 5.0, 6.0], "label": "point2"}
  ]
}
```

### NumPy Arrays (.npy)
```
Requirements:
- 1D (single point) or 2D (multiple points) arrays
- Shape: (n_points, n_dimensions)
- All numeric values
```

### HDF5 Files (.h5, .hdf5)
```
Configuration Options:
- dataset_name: Name of dataset containing points (default: 'points')

Common dataset names: 'points', 'coordinates', 'data', 'point_cloud'
```

## Usage Examples

### Basic File Upload

```python
import httpx

# Upload CSV file
with open("points.csv", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/v1/upload/file",
        files={"file": f},
        data={"config": '{"has_header": true, "delimiter": ","}'}
    )
    
upload_id = response.json()["upload_id"]
```

### Progress Monitoring

```python
# Check upload progress
response = httpx.get(f"http://localhost:8000/v1/upload/status/{upload_id}")
status = response.json()

print(f"Status: {status['status']}")
print(f"Progress: {status['progress']}%")
```

### Batch Upload

```python
# Upload multiple files
files = [
    ("files", open("data1.csv", "rb")),
    ("files", open("data2.json", "rb")),
    ("files", open("data3.npy", "rb"))
]

response = httpx.post(
    "http://localhost:8000/v1/upload/batch",
    files=files
)

upload_ids = response.json()["upload_ids"]
```

## Configuration

### File Limits

- **Maximum file size**: 100MB
- **Maximum batch size**: 10 files
- **Allowed extensions**: .csv, .json, .npy, .h5, .hdf5, .txt, .tsv
- **Cache TTL**: 24 hours (configurable)
- **Cleanup age**: 24 hours (configurable)

### Security Features

- File extension validation
- MIME type checking
- Binary executable detection
- Size limit enforcement
- Temporary file isolation

## Error Handling

### Error Types

- `FILE_TOO_LARGE`: File exceeds size limit
- `INVALID_FILE_EXTENSION`: Unsupported file type
- `SECURITY_RISK`: Potentially malicious file
- `PARSING_ERROR`: File format parsing failed
- `NO_VALID_POINTS`: No parseable point data found

### Error Response Format

```json
{
  "success": false,
  "timestamp": "2023-12-07T10:30:00Z",
  "message": "File validation failed",
  "error_code": "INVALID_FILE_EXTENSION",
  "error_type": "validation",
  "details": {"extension": ".exe", "allowed": [".csv", ".json", ".npy", ".h5"]}
}
```

## Integration with TDA Pipeline

### Automatic Storage Integration

```python
# Upload automatically stores point clouds
upload_id, progress = await upload_service.start_upload(file, auto_store=True)

# Retrieve stored point cloud ID
final_progress = upload_service.get_upload_progress(upload_id)
point_cloud_id = final_progress.data['point_cloud_id']

# Use in TDA computation
computation_request = TDAComputationRequest(
    point_cloud_id=UUID(point_cloud_id),
    config=ComputationConfig(algorithm="vietoris_rips")
)
```

### Storage Service Features

- **Metadata Management**: Name, description, tags, source tracking
- **Caching System**: LRU cache with configurable TTL
- **Search Capabilities**: Full-text search across metadata
- **Lifecycle Management**: Automatic cleanup of old data
- **Statistics Tracking**: Storage usage, access patterns, cache performance

## Testing

### Integration Test

Run the integration test to verify the complete upload workflow:

```bash
cd /home/stephen-dorman/dev/TDA_projects/backend
python test_upload_integration.py
```

### Manual Testing

```bash
# Start the server
uvicorn tda_backend.main:app --reload

# Test upload via curl
curl -X POST "http://localhost:8000/v1/upload/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data.csv"
```

## Dependencies

The upload system requires these additional dependencies (already included in pyproject.toml):

```
aiofiles>=23.2.1     # Async file operations
h5py>=3.9.0          # HDF5 file support  
pandas>=2.1.0        # CSV parsing
numpy>=1.24.0        # NumPy array support
```

## Future Enhancements

### Planned Features

1. **Database Persistence**: Replace in-memory storage with PostgreSQL
2. **Cloud Storage**: S3/Azure Blob Storage integration
3. **Streaming Upload**: Support for large file streaming
4. **Format Auto-detection**: Improved format detection
5. **Compression Support**: Gzip, zip file handling
6. **Webhook Integration**: Progress notifications
7. **Authentication**: User-based upload tracking
8. **Rate Limiting**: Upload frequency controls

### Performance Optimizations

1. **Chunked Processing**: Process large files in chunks
2. **Parallel Parsing**: Multi-threaded file processing
3. **Memory Management**: Streaming for large files
4. **Cache Optimization**: Smarter cache eviction policies

## Security Considerations

### Current Security Measures

1. **File Validation**: Extension and MIME type checking
2. **Size Limits**: Prevent resource exhaustion
3. **Binary Detection**: Reject executable files
4. **Temporary Isolation**: Isolated processing environment
5. **Error Sanitization**: No sensitive information in errors

### Additional Security Recommendations

1. **Virus Scanning**: Integrate antivirus scanning
2. **Content Filtering**: Deep content analysis
3. **Rate Limiting**: Prevent abuse
4. **User Authentication**: Track uploads by user
5. **Audit Logging**: Complete upload audit trail

## Monitoring and Observability

### Metrics Tracked

- Upload count and success rate
- File format distribution
- Processing time statistics
- Cache hit rates
- Storage utilization
- Error frequency by type

### Logging

- All upload attempts logged
- Processing progress logged
- Error conditions logged with context
- Performance metrics logged
- Security events logged

This comprehensive upload system provides a robust foundation for point cloud data ingestion in the TDA backend, with production-ready features for security, performance, and reliability.
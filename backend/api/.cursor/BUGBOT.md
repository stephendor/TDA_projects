# TDA Platform - API Layer BugBot Rules

## API Architecture Overview
The API layer provides **RESTful endpoints** for TDA analysis, user management, and real-time data processing. All endpoints follow **OpenAPI 3.0** specification with comprehensive documentation, proper error handling, and security enforcement.

## API Design Principles

### 1. RESTful Endpoint Design
```python
# ✅ DO: Follow REST conventions
@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis by ID"""
    pass

@router.post("/analyses")
async def create_analysis(analysis: AnalysisCreate):
    """Create new analysis"""
    pass

@router.put("/analyses/{analysis_id}")
async def update_analysis(analysis_id: str, analysis: AnalysisUpdate):
    """Update analysis"""
    pass

@router.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete analysis"""
    pass

# ❌ DON'T: Non-RESTful patterns
@router.post("/getAnalysis")  # Bad: should be GET
@router.get("/createAnalysis")  # Bad: should be POST
@router.post("/updateAnalysis")  # Bad: should be PUT
```

### 2. Request/Response Models
```python
# ✅ DO: Use Pydantic models with validation
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class AnalysisRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Analysis name")
    points: List[List[float]] = Field(..., min_items=1, description="Point cloud data")
    filtration_type: str = Field(..., regex="^(vietoris_rips|alpha_complex|cech|dtm)$")
    max_dimension: int = Field(..., ge=0, le=3, description="Maximum homology dimension")
    
    @validator('points')
    def validate_points(cls, v):
        if not all(len(point) == len(v[0]) for point in v):
            raise ValueError("All points must have the same dimension")
        return v

class AnalysisResponse(BaseModel):
    id: str
    status: str
    created_at: datetime
    result: Optional[Dict] = None
    error: Optional[str] = None

# ❌ DON'T: Use raw dictionaries or missing validation
@router.post("/analyze")
async def analyze_data(data: dict):  # Bad: no validation
    return {"result": "success"}
```

### 3. Error Handling & Status Codes
```python
# ✅ DO: Use appropriate HTTP status codes and detailed error messages
from fastapi import HTTPException, status

@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    analysis = await analysis_service.get_by_id(analysis_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found"
        )
    
    if analysis.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Analysis {analysis_id} failed: {analysis.error_message}"
        )
    
    return analysis

# ❌ DON'T: Generic error messages or wrong status codes
@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    analysis = await analysis_service.get_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=500, detail="Error")  # Bad: wrong status code
```

## TDA-Specific API Patterns

### 1. Analysis Endpoints
```python
# ✅ DO: Comprehensive analysis workflow endpoints
@router.post("/analyses", response_model=AnalysisResponse)
async def create_analysis(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    tda_service: TDAService = Depends(get_tda_service)
):
    """Create and start a new TDA analysis"""
    
    # Validate user permissions
    if not current_user.has_permission("can_run_analysis"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to run analysis"
        )
    
    # Validate input data
    if len(request.points) > 1_000_000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Point cloud exceeds maximum size of 1,000,000 points"
        )
    
    # Create analysis record
    analysis = await analysis_service.create(
        user_id=current_user.id,
        name=request.name,
        parameters=request.dict()
    )
    
    # Submit to processing queue
    await tda_service.submit_analysis(analysis.id, request)
    
    return AnalysisResponse(
        id=analysis.id,
        status="queued",
        created_at=analysis.created_at
    )

@router.get("/analyses/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get current status of analysis"""
    status_info = await analysis_service.get_status(analysis_id)
    return status_info

@router.get("/analyses/{analysis_id}/result")
async def get_analysis_result(analysis_id: str):
    """Get analysis results (persistence diagram, feature vectors)"""
    result = await analysis_service.get_result(analysis_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis result not found or not yet completed"
        )
    return result
```

### 2. Data Upload Endpoints
```python
# ✅ DO: Handle file uploads with proper validation
from fastapi import UploadFile, File

@router.post("/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Upload dataset file for analysis"""
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.txt', '.xyz')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Use CSV, TXT, or XYZ files"
        )
    
    # Validate file size (e.g., 100MB limit)
    if file.size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 100MB limit"
        )
    
    # Process and store file
    dataset_id = await dataset_service.store_file(
        file=file,
        user_id=current_user.id,
        name=name,
        description=description
    )
    
    return {"dataset_id": dataset_id, "status": "uploaded"}

# ❌ DON'T: Accept any file without validation
@router.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    # Bad: no file validation
    return {"status": "uploaded"}
```

### 3. Real-time Processing Endpoints
```python
# ✅ DO: WebSocket endpoints for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws/analysis/{analysis_id}")
async def analysis_websocket(
    websocket: WebSocket,
    analysis_id: str,
    token: str
):
    """WebSocket connection for real-time analysis updates"""
    
    # Validate authentication
    try:
        user = await authenticate_websocket(token)
    except AuthenticationError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    await websocket.accept()
    
    try:
        # Subscribe to analysis updates
        async for update in analysis_service.stream_updates(analysis_id):
            await websocket.send_json(update)
    except WebSocketDisconnect:
        # Handle client disconnect
        await analysis_service.unsubscribe(analysis_id, user.id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
```

## Security & Authentication

### 1. Endpoint Protection
```python
# ✅ DO: Protect all sensitive endpoints
@router.post("/analyses", dependencies=[Depends(require_permission("can_run_analysis"))])
async def create_analysis(request: AnalysisRequest):
    pass

@router.get("/users", dependencies=[Depends(require_permission("can_manage_users"))])
async def list_users():
    pass

# ❌ DON'T: Leave endpoints unprotected
@router.post("/analyses")  # Bad: no authentication
async def create_analysis(request: AnalysisRequest):
    pass
```

### 2. Input Validation & Sanitization
```python
# ✅ DO: Validate and sanitize all inputs
from pydantic import validator
import re

class UserInput(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v

# ❌ DON'T: Accept raw input without validation
@router.post("/users")
async def create_user(username: str, email: str):  # Bad: no validation
    pass
```

## Performance & Scalability

### 1. Pagination & Filtering
```python
# ✅ DO: Implement proper pagination and filtering
from fastapi import Query

@router.get("/analyses")
async def list_analyses(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[str] = Query(None, description="Filter by status"),
    user_id: Optional[str] = Query(None, description="Filter by user"),
    created_after: Optional[datetime] = Query(None, description="Filter by creation date")
):
    """List analyses with pagination and filtering"""
    
    filters = {}
    if status:
        filters["status"] = status
    if user_id:
        filters["user_id"] = user_id
    if created_after:
        filters["created_at__gte"] = created_after
    
    total, analyses = await analysis_service.list_paginated(
        page=page,
        size=size,
        filters=filters
    )
    
    return {
        "items": analyses,
        "pagination": {
            "page": page,
            "size": size,
            "total": total,
            "pages": (total + size - 1) // size
        }
    }

# ❌ DON'T: Return unlimited results
@router.get("/analyses")
async def list_analyses():  # Bad: no pagination
    return await analysis_service.list_all()
```

### 2. Response Optimization
```python
# ✅ DO: Optimize response size and structure
@router.get("/analyses/{analysis_id}/summary")
async def get_analysis_summary(analysis_id: str):
    """Get lightweight analysis summary"""
    summary = await analysis_service.get_summary(analysis_id)
    return summary

@router.get("/analyses/{analysis_id}/full")
async def get_analysis_full(analysis_id: str):
    """Get full analysis with all details"""
    full_analysis = await analysis_service.get_full(analysis_id)
    return full_analysis

# ❌ DON'T: Always return full objects
@router.get("/analyses")
async def list_analyses():
    # Bad: returns full analysis objects in list
    return await analysis_service.list_all()
```

## API Documentation

### 1. OpenAPI Documentation
```python
# ✅ DO: Comprehensive endpoint documentation
@router.post(
    "/analyses",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new TDA analysis",
    description="""
    Create and submit a new Topological Data Analysis job.
    
    The analysis will be queued for processing and the user will receive
    real-time updates on the progress.
    
    **Required permissions**: can_run_analysis
    """,
    responses={
        201: {"description": "Analysis created successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        413: {"description": "Request entity too large"},
        422: {"description": "Validation error"}
    }
)
async def create_analysis(request: AnalysisRequest):
    pass

# ❌ DON'T: Missing or incomplete documentation
@router.post("/analyses")
async def create_analysis(request: AnalysisRequest):  # Bad: no documentation
    pass
```

### 2. Example Requests/Responses
```python
# ✅ DO: Provide comprehensive examples
class AnalysisRequest(BaseModel):
    name: str = Field(..., example="Market Regime Analysis Q1 2024")
    points: List[List[float]] = Field(..., example=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    filtration_type: str = Field(..., example="vietoris_rips")
    max_dimension: int = Field(..., example=2)

# ❌ DON'T: Missing examples
class AnalysisRequest(BaseModel):
    name: str
    points: List[List[float]]
    filtration_type: str
    max_dimension: int
```

## Testing API Endpoints

### 1. Test Client Setup
```python
# ✅ DO: Proper test client setup with authentication
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

@pytest.fixture
def authenticated_client():
    with patch('tda_backend.auth.get_current_user') as mock_auth:
        mock_auth.return_value = User(id="test-user", permissions=["can_run_analysis"])
        client = TestClient(app)
        client.headers = {"Authorization": "Bearer test-token"}
        return client

@pytest.mark.asyncio
async def test_create_analysis_success(authenticated_client):
    response = authenticated_client.post(
        "/api/v1/analyses",
        json={
            "name": "Test Analysis",
            "points": [[1.0, 2.0], [3.0, 4.0]],
            "filtration_type": "vietoris_rips",
            "max_dimension": 2
        }
    )
    assert response.status_code == 201
    assert "id" in response.json()

# ❌ DON'T: Test without proper authentication
def test_create_analysis_success():
    client = TestClient(app)  # Bad: no authentication
    response = client.post("/api/v1/analyses", json={...})
    assert response.status_code == 201
```

### 2. Error Case Testing
```python
# ✅ DO: Test error conditions and edge cases
@pytest.mark.asyncio
async def test_create_analysis_invalid_points(authenticated_client):
    response = authenticated_client.post(
        "/api/v1/analyses",
        json={
            "name": "Test Analysis",
            "points": [],  # Empty point cloud
            "filtration_type": "vietoris_rips",
            "max_dimension": 2
        }
    )
    assert response.status_code == 422
    assert "points" in response.json()["detail"][0]["loc"]

@pytest.mark.asyncio
async def test_create_analysis_insufficient_permissions():
    with patch('tda_backend.auth.get_current_user') as mock_auth:
        mock_auth.return_value = User(id="test-user", permissions=[])
        client = TestClient(app)
        client.headers = {"Authorization": "Bearer test-token"}
        
        response = client.post("/api/v1/analyses", json={...})
        assert response.status_code == 403
        assert "Insufficient permissions" in response.json()["detail"]
```

## Common API Issues

### 1. Authentication & Authorization
- **Missing Authentication**: Ensure all endpoints require valid JWT tokens
- **Permission Checking**: Verify user permissions before allowing operations
- **Token Expiration**: Handle expired tokens gracefully

### 2. Input Validation
- **Data Type Validation**: Ensure proper data types for all inputs
- **Business Logic Validation**: Validate business rules (e.g., point cloud size limits)
- **SQL Injection Prevention**: Use parameterized queries, never string concatenation

### 3. Error Handling
- **Generic Error Messages**: Provide specific, actionable error messages
- **Status Code Consistency**: Use appropriate HTTP status codes
- **Error Logging**: Log errors with sufficient context for debugging

### 4. Performance Issues
- **N+1 Queries**: Avoid multiple database queries in loops
- **Large Response Payloads**: Implement pagination and filtering
- **Missing Caching**: Cache frequently accessed data when appropriate

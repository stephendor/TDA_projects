# TDA Platform - Backend Services BugBot Rules

## Backend Architecture Overview
The backend consists of **FastAPI services** that orchestrate C++ TDA algorithms, manage data persistence, and provide real-time streaming capabilities. The system uses a **hybrid database approach** with PostgreSQL for metadata and MongoDB for large binary data.

## Core Backend Components

### 1. FastAPI Application Structure
- **Main App**: `tda_backend/` package with modular service architecture
- **API Versioning**: RESTful endpoints with OpenAPI documentation
- **Middleware**: Authentication, CORS, logging, error handling
- **Dependencies**: Database connections, service injections

### 2. Service Layer Architecture
- **TDA Service**: Core algorithm orchestration and C++ integration
- **Storage Service**: Dual-database operations (PostgreSQL + MongoDB)
- **Streaming Service**: Kafka producer/consumer management
- **Authentication Service**: JWT validation and RBAC enforcement

### 3. Database Integration
- **PostgreSQL**: User management, metadata, audit trails, versioning
- **MongoDB**: Persistence diagrams, feature vectors, large datasets
- **Connection Pooling**: Efficient database connection management
- **Transaction Management**: ACID compliance for critical operations

## Backend-Specific Rules

### 1. FastAPI Best Practices
```python
# ✅ DO: Use proper dependency injection
from fastapi import Depends, HTTPException
from .services import get_tda_service

@router.post("/analyze")
async def analyze_data(
    data: AnalysisRequest,
    tda_service: TDAService = Depends(get_tda_service)
):
    return await tda_service.process_analysis(data)

# ❌ DON'T: Create services in route handlers
@router.post("/analyze")
async def analyze_data(data: AnalysisRequest):
    tda_service = TDAService()  # Bad: creates new instance each time
    return await tda_service.process_analysis(data)
```

### 2. C++ Integration Patterns
```python
# ✅ DO: Proper pybind11 binding usage
import tda_core

def compute_persistence_diagram(points: np.ndarray) -> Dict:
    try:
        # Ensure proper data type conversion
        points_cpp = points.astype(np.float64, copy=False)
        result = tda_core.compute_vietoris_rips(points_cpp)
        return result
    except Exception as e:
        logger.error(f"C++ algorithm failed: {e}")
        raise HTTPException(status_code=500, detail="Algorithm computation failed")

# ❌ DON'T: Ignore C++ exceptions or memory management
def compute_persistence_diagram(points: np.ndarray) -> Dict:
    result = tda_core.compute_vietoris_rips(points)  # Bad: no error handling
    return result
```

### 3. Database Operations
```python
# ✅ DO: Proper transaction management
async def store_analysis_result(analysis_data: Dict, user_id: str):
    async with postgres_db.transaction():
        # Store metadata in PostgreSQL
        metadata_id = await store_metadata(analysis_data, user_id)
        
        # Store large data in MongoDB
        await mongo_db.analysis_results.insert_one({
            "metadata_id": metadata_id,
            "persistence_diagram": analysis_data["diagram"],
            "feature_vectors": analysis_data["vectors"]
        })

# ❌ DON'T: Mix database operations without transactions
async def store_analysis_result(analysis_data: Dict, user_id: str):
    metadata_id = await store_metadata(analysis_data, user_id)  # Bad: no transaction
    await mongo_db.analysis_results.insert_one({...})  # Could fail after metadata stored
```

### 4. Async/Await Patterns
```python
# ✅ DO: Proper async operation handling
async def process_streaming_data(kafka_message: Dict):
    # Process message asynchronously
    analysis_result = await process_analysis(kafka_message["data"])
    
    # Store result asynchronously
    await store_result(analysis_result)
    
    # Send notification asynchronously
    await notify_user(analysis_result["user_id"])

# ❌ DON'T: Block async operations
async def process_streaming_data(kafka_message: Dict):
    # Bad: blocks the event loop
    analysis_result = process_analysis_sync(kafka_message["data"])
    store_result_sync(analysis_result)
```

### 5. Error Handling & Logging
```python
# ✅ DO: Comprehensive error handling with logging
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

async def run_tda_analysis(analysis_request: AnalysisRequest):
    try:
        # Validate input data
        if not analysis_request.points.size > 0:
            raise ValueError("Point cloud cannot be empty")
        
        # Run analysis
        result = await tda_service.analyze(analysis_request)
        
        logger.info(f"Analysis completed for user {analysis_request.user_id}")
        return result
        
    except ValueError as e:
        logger.warning(f"Invalid analysis request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# ❌ DON'T: Generic error handling or missing logging
async def run_tda_analysis(analysis_request: AnalysisRequest):
    try:
        result = await tda_service.analyze(analysis_request)
        return result
    except Exception as e:  # Bad: too generic
        raise HTTPException(status_code=500, detail="Error")  # Bad: no logging
```

## Security & Authentication

### 1. JWT Token Validation
```python
# ✅ DO: Proper JWT validation with dependency injection
from fastapi import Depends, HTTPException, status
from .auth import get_current_user

@router.get("/protected-endpoint")
async def protected_operation(
    current_user: User = Depends(get_current_user)
):
    if not current_user.has_permission("can_run_analysis"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return {"message": "Access granted"}

# ❌ DON'T: Manual token validation in route handlers
@router.get("/protected-endpoint")
async def protected_operation(token: str = Header(...)):
    # Bad: manual token handling
    if not validate_token_manually(token):
        raise HTTPException(status_code=401)
```

### 2. RBAC Implementation
```python
# ✅ DO: Use decorators for permission checking
from .auth import require_permission

@router.post("/run-analysis")
@require_permission("can_run_analysis")
async def run_analysis(analysis_request: AnalysisRequest):
    return await tda_service.run_analysis(analysis_request)

# ❌ DON'T: Check permissions inline
@router.post("/run-analysis")
async def run_analysis(analysis_request: AnalysisRequest, user: User):
    if user.role != "analyst":  # Bad: hardcoded role checking
        raise HTTPException(status_code=403)
```

## Performance & Scalability

### 1. Database Query Optimization
```python
# ✅ DO: Use proper indexing and query optimization
async def get_user_analyses(user_id: str, limit: int = 100):
    # Use indexed query with pagination
    query = """
        SELECT id, name, created_at, status 
        FROM analyses 
        WHERE user_id = $1 
        ORDER BY created_at DESC 
        LIMIT $2
    """
    return await postgres_db.fetch(query, user_id, limit)

# ❌ DON'T: N+1 queries or missing pagination
async def get_user_analyses(user_id: str):
    # Bad: could return unlimited results
    query = "SELECT * FROM analyses WHERE user_id = $1"
    return await postgres_db.fetch(query, user_id)
```

### 2. Streaming & Real-time Processing
```python
# ✅ DO: Efficient Kafka message handling
from kafka import KafkaProducer
import json

class StreamingService:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    async def publish_analysis_event(self, event_data: Dict):
        future = self.producer.send('analysis-events', event_data)
        # Handle delivery reports asynchronously
        future.add_callback(self.on_send_success)
        future.add_errback(self.on_send_error)

# ❌ DON'T: Block on Kafka operations
async def publish_analysis_event(self, event_data: Dict):
    # Bad: blocks until message is sent
    self.producer.send('analysis-events', event_data).get()
```

## Testing & Quality

### 1. Unit Testing Patterns
```python
# ✅ DO: Comprehensive unit testing with mocks
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_tda_analysis_success():
    with patch('tda_backend.services.tda_service.TDAService') as mock_service:
        mock_service.return_value.analyze = AsyncMock(return_value={"result": "success"})
        
        response = await client.post("/api/v1/analyze", json={"points": [[1, 2], [3, 4]]})
        assert response.status_code == 200
        assert response.json()["result"] == "success"

# ❌ DON'T: Test without proper mocking
@pytest.mark.asyncio
async def test_tda_analysis_success():
    # Bad: depends on external services
    response = await client.post("/api/v1/analyze", json={"points": [[1, 2], [3, 4]]})
    assert response.status_code == 200
```

### 2. Integration Testing
```python
# ✅ DO: Use test databases and proper setup/teardown
@pytest.fixture
async def test_db():
    # Setup test database
    test_db = await setup_test_database()
    yield test_db
    # Cleanup
    await cleanup_test_database(test_db)

@pytest.mark.asyncio
async def test_database_integration(test_db):
    # Test with isolated test database
    result = await test_db.execute("SELECT 1")
    assert result == 1
```

## Common Backend Issues

### 1. Memory Management
- **C++ Integration**: Ensure proper cleanup of C++ objects
- **Database Connections**: Use connection pooling, avoid connection leaks
- **Large Data**: Stream large datasets, avoid loading everything into memory

### 2. Error Propagation
- **C++ Exceptions**: Properly translate to Python exceptions
- **Database Errors**: Handle connection failures, constraint violations
- **External Services**: Implement circuit breakers, retry logic

### 3. Configuration Management
- **Environment Variables**: Use for sensitive data (API keys, database URLs)
- **Configuration Files**: Use for non-sensitive settings
- **Validation**: Validate configuration at startup

### 4. Monitoring & Observability
- **Logging**: Structured logging with correlation IDs
- **Metrics**: Performance metrics, business metrics
- **Tracing**: Distributed tracing for complex operations
- **Health Checks**: Endpoint health, dependency health

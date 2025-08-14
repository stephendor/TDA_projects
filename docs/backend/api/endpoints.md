# API Endpoints Reference

Complete reference for the TDA Platform Backend REST API endpoints.

## 🌐 Base URL and Versioning

```
Base URL: https://api.tda-platform.com/api/v1
Development: http://localhost:8000/api/v1
```

All endpoints are versioned using URL path versioning (`/api/v1/`). The API follows REST conventions and returns JSON responses.

## 🔐 Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <jwt_token>
```

### Authentication Endpoints

#### **POST /auth/login**
Authenticate user and receive JWT token.

**Request:**
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "123",
    "username": "user@example.com",
    "roles": ["user"]
  }
}
```

#### **POST /auth/refresh**
Refresh JWT token.

**Request:**
```http
POST /api/v1/auth/refresh
Authorization: Bearer <current_token>
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## 📊 TDA Computation Endpoints

### **POST /tda/compute/vietoris-rips**
Compute Vietoris-Rips persistence homology.

**Request:**
```http
POST /api/v1/tda/compute/vietoris-rips
Authorization: Bearer <token>
Content-Type: application/json

{
  "point_cloud": {
    "points": [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ],
    "dimension": 3
  },
  "parameters": {
    "max_edge_length": 2.0,
    "max_dimension": 2,
    "num_threads": 4
  },
  "job_options": {
    "async": false,
    "cache_results": true,
    "notify_completion": true
  }
}
```

**Response (Synchronous):**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "job_id": "vr_20231201_123456_abc123",
  "algorithm": "vietoris_rips",
  "status": "completed",
  "results": {
    "persistence_pairs": [
      {
        "dimension": 0,
        "birth": 0.0,
        "death": 1.414,
        "persistence": 1.414
      },
      {
        "dimension": 1,
        "birth": 1.0,
        "death": 1.732,
        "persistence": 0.732
      }
    ],
    "betti_numbers": {
      "beta_0": 1,
      "beta_1": 0,
      "beta_2": 0
    },
    "statistics": {
      "computation_time_ms": 156,
      "memory_peak_mb": 24.5,
      "num_simplices": 1247,
      "num_edges": 6
    }
  },
  "metadata": {
    "computed_at": "2023-12-01T12:34:56Z",
    "cache_hit": false,
    "version": "1.0.0"
  }
}
```

**Response (Asynchronous):**
```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "job_id": "vr_20231201_123456_abc123",
  "status": "submitted",
  "message": "Computation job submitted successfully",
  "estimated_completion": "2023-12-01T12:36:56Z",
  "status_url": "/api/v1/tda/jobs/vr_20231201_123456_abc123",
  "webhook_url": null
}
```

### **POST /tda/compute/alpha-complex**
Compute Alpha complex persistence homology.

**Request:**
```http
POST /api/v1/tda/compute/alpha-complex
Authorization: Bearer <token>
Content-Type: application/json

{
  "point_cloud": {
    "points": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]],
    "dimension": 2
  },
  "parameters": {
    "max_alpha_value": 1.0,
    "max_dimension": 1
  }
}
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "job_id": "alpha_20231201_123457_def456",
  "algorithm": "alpha_complex",
  "status": "completed",
  "results": {
    "persistence_pairs": [
      {
        "dimension": 0,
        "birth": 0.0,
        "death": "infinity",
        "persistence": "infinity"
      },
      {
        "dimension": 1,
        "birth": 0.433,
        "death": "infinity",
        "persistence": "infinity"
      }
    ],
    "alpha_shape": {
      "vertices": 3,
      "edges": 3,
      "triangles": 1
    }
  }
}
```

### **POST /tda/compute/sparse-rips**
Compute sparse Vietoris-Rips approximation.

**Request:**
```http
POST /api/v1/tda/compute/sparse-rips
Authorization: Bearer <token>
Content-Type: application/json

{
  "point_cloud": {
    "points": "... large point cloud ...",
    "dimension": 3
  },
  "parameters": {
    "sparsity_factor": 0.1,
    "max_dimension": 2,
    "use_landmarks": true,
    "num_landmarks": 1000,
    "quality_target": 0.85
  }
}
```

## 📁 Data Management Endpoints

### **POST /data/upload**
Upload point cloud data for analysis.

**Request:**
```http
POST /api/v1/data/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

Content-Disposition: form-data; name="file"; filename="pointcloud.csv"
Content-Type: text/csv

x,y,z
0.0,0.0,0.0
1.0,0.0,0.0
0.0,1.0,0.0
...

Content-Disposition: form-data; name="metadata"
Content-Type: application/json

{
  "name": "Sample Point Cloud",
  "description": "Test dataset for TDA analysis",
  "format": "csv",
  "dimension": 3,
  "tags": ["test", "synthetic"]
}
```

**Response:**
```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "file_id": "pc_20231201_123458_ghi789",
  "name": "Sample Point Cloud",
  "size_bytes": 15728,
  "num_points": 1000,
  "dimension": 3,
  "format": "csv",
  "checksum": "sha256:a1b2c3...",
  "uploaded_at": "2023-12-01T12:34:58Z",
  "status": "uploaded",
  "validation": {
    "is_valid": true,
    "warnings": [],
    "errors": []
  }
}
```

### **GET /data/{file_id}**
Retrieve uploaded data information.

**Request:**
```http
GET /api/v1/data/pc_20231201_123458_ghi789
Authorization: Bearer <token>
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "file_id": "pc_20231201_123458_ghi789",
  "name": "Sample Point Cloud",
  "description": "Test dataset for TDA analysis",
  "size_bytes": 15728,
  "num_points": 1000,
  "dimension": 3,
  "format": "csv",
  "uploaded_at": "2023-12-01T12:34:58Z",
  "status": "processed",
  "metadata": {
    "tags": ["test", "synthetic"],
    "bounding_box": {
      "min": [0.0, 0.0, 0.0],
      "max": [10.0, 10.0, 10.0]
    },
    "statistics": {
      "mean": [5.0, 5.0, 5.0],
      "std": [2.89, 2.89, 2.89]
    }
  }
}
```

### **DELETE /data/{file_id}**
Delete uploaded data.

**Request:**
```http
DELETE /api/v1/data/pc_20231201_123458_ghi789
Authorization: Bearer <token>
```

**Response:**
```http
HTTP/1.1 204 No Content
```

## ⚙️ Job Management Endpoints

### **GET /jobs/{job_id}**
Get job status and results.

**Request:**
```http
GET /api/v1/jobs/vr_20231201_123456_abc123
Authorization: Bearer <token>
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "job_id": "vr_20231201_123456_abc123",
  "algorithm": "vietoris_rips",
  "status": "running",
  "progress": {
    "percentage": 65,
    "stage": "computing_persistence",
    "estimated_completion": "2023-12-01T12:36:30Z"
  },
  "created_at": "2023-12-01T12:34:56Z",
  "started_at": "2023-12-01T12:35:02Z",
  "parameters": {
    "max_edge_length": 2.0,
    "max_dimension": 2,
    "num_threads": 4
  },
  "input_data": {
    "file_id": "pc_20231201_123458_ghi789",
    "num_points": 1000,
    "dimension": 3
  }
}
```

### **POST /jobs/{job_id}/cancel**
Cancel a running job.

**Request:**
```http
POST /api/v1/jobs/vr_20231201_123456_abc123/cancel
Authorization: Bearer <token>
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "job_id": "vr_20231201_123456_abc123",
  "status": "cancelled",
  "message": "Job cancellation requested",
  "cancelled_at": "2023-12-01T12:35:45Z"
}
```

### **GET /jobs**
List user's jobs with filtering.

**Request:**
```http
GET /api/v1/jobs?status=completed&algorithm=vietoris_rips&limit=10&offset=0
Authorization: Bearer <token>
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "jobs": [
    {
      "job_id": "vr_20231201_123456_abc123",
      "algorithm": "vietoris_rips",
      "status": "completed",
      "created_at": "2023-12-01T12:34:56Z",
      "completed_at": "2023-12-01T12:36:15Z",
      "duration_ms": 79000
    }
  ],
  "pagination": {
    "total": 25,
    "limit": 10,
    "offset": 0,
    "has_next": true
  }
}
```

## 🌊 Streaming Endpoints

### **POST /stream/publish**
Publish event to streaming pipeline.

**Request:**
```http
POST /api/v1/stream/publish
Authorization: Bearer <token>
Content-Type: application/json

{
  "topic": "tda_events",
  "event": {
    "type": "point_cloud_update",
    "data": {
      "stream_id": "sensor_001",
      "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      "timestamp": "2023-12-01T12:37:00Z"
    }
  },
  "options": {
    "partition_key": "sensor_001",
    "ensure_ordering": true
  }
}
```

**Response:**
```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "message_id": "msg_20231201_123700_jkl012",
  "topic": "tda_events",
  "partition": 2,
  "offset": 15847,
  "timestamp": "2023-12-01T12:37:00.123Z",
  "status": "published"
}
```

### **GET /stream/status**
Get streaming pipeline status.

**Request:**
```http
GET /api/v1/stream/status
Authorization: Bearer <token>
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "kafka": {
    "status": "healthy",
    "brokers": 3,
    "topics": 5,
    "total_partitions": 15,
    "throughput_msgs_per_sec": 1250
  },
  "flink": {
    "status": "healthy",
    "jobs": [
      {
        "job_id": "flink_tda_streaming",
        "status": "running",
        "parallelism": 4,
        "uptime": "2d 4h 15m",
        "processed_records": 2847592
      }
    ]
  },
  "overall_health": "healthy"
}
```

## 🔍 Monitoring Endpoints

### **GET /health**
Health check endpoint.

**Request:**
```http
GET /api/v1/health
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "timestamp": "2023-12-01T12:37:30Z",
  "version": "1.0.0",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 5,
      "connections": {
        "active": 3,
        "idle": 7,
        "max": 20
      }
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 2,
      "memory_usage": "45.2MB"
    },
    "kafka": {
      "status": "healthy",
      "response_time_ms": 12,
      "broker_count": 3
    },
    "cpp_engine": {
      "status": "healthy",
      "version": "1.0.0",
      "last_computation": "2023-12-01T12:35:15Z"
    }
  }
}
```

### **GET /metrics**
Prometheus metrics endpoint.

**Request:**
```http
GET /api/v1/metrics
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: text/plain

# HELP tda_api_requests_total Total API requests
# TYPE tda_api_requests_total counter
tda_api_requests_total{method="POST",endpoint="/tda/compute/vietoris-rips",status="200"} 145

# HELP tda_computation_duration_seconds TDA computation duration
# TYPE tda_computation_duration_seconds histogram
tda_computation_duration_seconds_bucket{algorithm="vietoris_rips",dimension="2",le="0.1"} 23
tda_computation_duration_seconds_bucket{algorithm="vietoris_rips",dimension="2",le="1.0"} 89
tda_computation_duration_seconds_bucket{algorithm="vietoris_rips",dimension="2",le="+Inf"} 145
tda_computation_duration_seconds_sum{algorithm="vietoris_rips",dimension="2"} 67.4
tda_computation_duration_seconds_count{algorithm="vietoris_rips",dimension="2"} 145
```

## 📝 Request/Response Schemas

### **Point Cloud Schema**
```json
{
  "type": "object",
  "properties": {
    "points": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {"type": "number"}
      },
      "minItems": 2
    },
    "dimension": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10
    }
  },
  "required": ["points", "dimension"]
}
```

### **Job Status Values**
- `submitted` - Job has been submitted to the queue
- `running` - Job is currently being processed
- `completed` - Job completed successfully
- `failed` - Job failed with error
- `cancelled` - Job was cancelled by user
- `timeout` - Job exceeded maximum execution time

### **Algorithm Types**
- `vietoris_rips` - Standard Vietoris-Rips filtration
- `alpha_complex` - Alpha complex filtration
- `sparse_rips` - Sparse Vietoris-Rips approximation
- `cech_complex` - Čech complex filtration
- `witness_complex` - Witness complex filtration

## ⚠️ Error Handling

### **Error Response Format**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid point cloud dimension",
    "details": {
      "field": "point_cloud.dimension",
      "provided": 0,
      "expected": "integer >= 1"
    },
    "timestamp": "2023-12-01T12:37:45Z",
    "request_id": "req_20231201_123745_mno345"
  }
}
```

### **Common Error Codes**
- `AUTHENTICATION_REQUIRED` (401) - Missing or invalid authentication
- `PERMISSION_DENIED` (403) - Insufficient permissions
- `VALIDATION_ERROR` (400) - Invalid request data
- `NOT_FOUND` (404) - Resource not found
- `RATE_LIMIT_EXCEEDED` (429) - Too many requests
- `COMPUTATION_FAILED` (422) - TDA computation failed
- `SERVICE_UNAVAILABLE` (503) - Service temporarily unavailable

## 🔧 Request Examples

### **cURL Examples**

```bash
# Authenticate
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user@example.com","password":"password"}'

# Compute Vietoris-Rips
curl -X POST http://localhost:8000/api/v1/tda/compute/vietoris-rips \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "point_cloud": {
      "points": [[0,0],[1,0],[0,1],[1,1]],
      "dimension": 2
    },
    "parameters": {
      "max_edge_length": 2.0,
      "max_dimension": 1
    }
  }'

# Upload data
curl -X POST http://localhost:8000/api/v1/data/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@pointcloud.csv" \
  -F 'metadata={"name":"Test Data","dimension":3}'

# Check job status
curl -X GET http://localhost:8000/api/v1/jobs/vr_20231201_123456_abc123 \
  -H "Authorization: Bearer $TOKEN"
```

### **Python Client Example**

```python
import requests
import json

# Authentication
auth_response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    json={"username": "user@example.com", "password": "password"}
)
token = auth_response.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# Compute TDA
tda_response = requests.post(
    "http://localhost:8000/api/v1/tda/compute/vietoris-rips",
    headers=headers,
    json={
        "point_cloud": {
            "points": [[0,0],[1,0],[0,1],[1,1]],
            "dimension": 2
        },
        "parameters": {
            "max_edge_length": 2.0,
            "max_dimension": 1
        }
    }
)

results = tda_response.json()
print(f"Found {len(results['results']['persistence_pairs'])} persistence pairs")
```

---

*Next: [Kafka Integration](../kafka/architecture.md) | [Development Guide](../development/getting-started.md)*
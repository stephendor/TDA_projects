# End-to-End Integration Testing

Comprehensive testing framework for the TDA Platform Backend covering the complete data flow from API requests through Kafka messaging to Flink stream processing.

## 🧪 Testing Strategy Overview

### **Test Architecture**

```mermaid
graph TB
    subgraph "Test Environment"
        TC[Test Controller]
        TD[Test Data Generator]
        TV[Test Validator]
    end
    
    subgraph "System Under Test"
        API[FastAPI Backend]
        KAFKA[Kafka Cluster]
        FLINK[Flink Stream Processing]
        DB[(PostgreSQL)]
        REDIS[(Redis Cache)]
    end
    
    subgraph "Test Types"
        UNIT[Unit Tests]
        INT[Integration Tests]
        E2E[End-to-End Tests]
        PERF[Performance Tests]
        CHAOS[Chaos Tests]
    end
    
    TC --> API
    TC --> KAFKA
    TC --> FLINK
    TD --> TC
    TV --> TC
    
    API --> KAFKA
    KAFKA --> FLINK
    API --> DB
    API --> REDIS
```

### **Testing Pyramid**

| Test Level | Coverage | Speed | Scope | Tools |
|------------|----------|-------|-------|-------|
| **Unit** | 70% | Fast (ms) | Individual functions | pytest, unittest |
| **Integration** | 20% | Medium (seconds) | Service interactions | pytest, testcontainers |
| **End-to-End** | 10% | Slow (minutes) | Full system workflow | pytest, docker-compose |

## 🔧 Test Environment Setup

### **Docker Test Environment**

```yaml
# /tests/docker-compose.test.yml
version: '3.8'

services:
  # Test Database
  test-postgres:
    image: postgres:16-alpine
    container_name: test-postgres
    environment:
      POSTGRES_DB: tda_test_db
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5433:5432"
    volumes:
      - test_postgres_data:/var/lib/postgresql/data
      - ./test-data/init.sql:/docker-entrypoint-initdb.d/init.sql

  # Test Redis
  test-redis:
    image: redis:7-alpine
    container_name: test-redis
    ports:
      - "6380:6379"
    command: redis-server --appendonly yes

  # Test Kafka Cluster
  test-zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: test-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  test-kafka:
    image: confluentinc/cp-kafka:7.4.0
    container_name: test-kafka
    depends_on:
      - test-zookeeper
    ports:
      - "9093:9093"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: test-zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://test-kafka:29092,PLAINTEXT_HOST://localhost:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: true

  # Test Schema Registry
  test-schema-registry:
    image: confluentinc/cp-schema-registry:7.4.0
    container_name: test-schema-registry
    depends_on:
      - test-kafka
    ports:
      - "8082:8081"
    environment:
      SCHEMA_REGISTRY_HOST_NAME: test-schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: test-kafka:29092

  # Test Flink
  test-flink-jobmanager:
    image: flink:1.18.0
    container_name: test-flink-jobmanager
    ports:
      - "8082:8081"
    command: jobmanager
    environment:
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: test-flink-jobmanager
        taskmanager.numberOfTaskSlots: 2

  test-flink-taskmanager:
    image: flink:1.18.0
    container_name: test-flink-taskmanager
    depends_on:
      - test-flink-jobmanager
    command: taskmanager
    environment:
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: test-flink-jobmanager
        taskmanager.numberOfTaskSlots: 2

  # TDA Backend API (Test Mode)
  test-api:
    build:
      context: ..
      dockerfile: Dockerfile.test
    container_name: test-api
    depends_on:
      - test-postgres
      - test-redis
      - test-kafka
    ports:
      - "8001:8000"
    environment:
      ENVIRONMENT: test
      DATABASE_URL: postgresql://test_user:test_password@test-postgres:5432/tda_test_db
      REDIS_URL: redis://test-redis:6379/0
      KAFKA_BOOTSTRAP_SERVERS: test-kafka:29092
      SCHEMA_REGISTRY_URL: http://test-schema-registry:8081
    volumes:
      - ../tests:/app/tests
      - test_uploads:/app/uploads

volumes:
  test_postgres_data:
  test_uploads:
```

### **Test Configuration**

```python
# /tests/conftest.py
import asyncio
import pytest
import docker
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient
from testcontainers.compose import DockerCompose

from backend.core.config import get_settings
from backend.core.database import get_database
from backend.services.kafka_producer import KafkaProducerService
from backend.services.tda_service import TDAService

@pytest.fixture(scope="session")
def docker_compose_environment():
    """Start the complete test environment using Docker Compose."""
    compose_path = Path(__file__).parent / "docker-compose.test.yml"
    
    with DockerCompose(str(compose_path.parent), compose_file_name="docker-compose.test.yml") as compose:
        # Wait for services to be ready
        _wait_for_services_ready()
        yield compose

def _wait_for_services_ready(max_wait: int = 60):
    """Wait for all services to be ready."""
    client = docker.from_env()
    services = [
        "test-postgres", "test-redis", "test-kafka", 
        "test-schema-registry", "test-flink-jobmanager", "test-api"
    ]
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        all_ready = True
        for service in services:
            try:
                container = client.containers.get(service)
                if container.status != "running":
                    all_ready = False
                    break
                
                # Additional health checks
                if service == "test-api":
                    # Check API health endpoint
                    import requests
                    try:
                        response = requests.get("http://localhost:8001/api/v1/health", timeout=5)
                        if response.status_code != 200:
                            all_ready = False
                    except:
                        all_ready = False
                        
            except docker.errors.NotFound:
                all_ready = False
                break
        
        if all_ready:
            break
        time.sleep(2)
    
    if not all_ready:
        raise RuntimeError("Services failed to start within timeout")

@pytest.fixture(scope="session")
def test_settings():
    """Test environment settings."""
    return get_settings(environment="test")

@pytest.fixture(scope="session")
async def test_client(docker_compose_environment, test_settings) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for API testing."""
    async with AsyncClient(
        base_url="http://localhost:8001/api/v1",
        timeout=30.0
    ) as client:
        yield client

@pytest.fixture(scope="session")
async def test_database(test_settings):
    """Test database connection."""
    db = get_database(test_settings.database_url)
    yield db
    await db.disconnect()

@pytest.fixture(scope="session") 
async def test_kafka_producer(test_settings):
    """Test Kafka producer."""
    producer = KafkaProducerService(test_settings)
    await producer.start()
    yield producer
    await producer.stop()

@pytest.fixture(scope="function")
async def clean_database(test_database):
    """Clean database before each test."""
    # Reset database state
    await test_database.execute("TRUNCATE TABLE jobs, results, files CASCADE")
    yield
    # Additional cleanup if needed

@pytest.fixture(scope="function") 
def test_point_cloud():
    """Sample point cloud data for testing."""
    return {
        "points": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ],
        "dimension": 3
    }

@pytest.fixture(scope="function")
def test_computation_params():
    """Standard computation parameters."""
    return {
        "max_edge_length": 2.0,
        "max_dimension": 2,
        "num_threads": 2
    }
```

## 🔬 Unit Testing Framework

### **API Endpoint Tests**

```python
# /tests/unit/test_api_endpoints.py
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
class TestTDAComputationEndpoints:
    """Test TDA computation endpoints."""
    
    async def test_vietoris_rips_computation_success(
        self, 
        test_client: AsyncClient,
        test_point_cloud: dict,
        test_computation_params: dict
    ):
        """Test successful Vietoris-Rips computation."""
        # Arrange
        request_data = {
            "point_cloud": test_point_cloud,
            "parameters": test_computation_params,
            "job_options": {
                "async": False,
                "cache_results": True
            }
        }
        
        # Act
        response = await test_client.post(
            "/tda/compute/vietoris-rips",
            json=request_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        
        assert "job_id" in result
        assert result["algorithm"] == "vietoris_rips"
        assert result["status"] == "completed"
        assert "results" in result
        assert "persistence_pairs" in result["results"]
        assert "betti_numbers" in result["results"]
        assert len(result["results"]["persistence_pairs"]) > 0
    
    async def test_vietoris_rips_async_computation(
        self,
        test_client: AsyncClient,
        test_point_cloud: dict,
        test_computation_params: dict
    ):
        """Test asynchronous Vietoris-Rips computation."""
        # Arrange
        request_data = {
            "point_cloud": test_point_cloud,
            "parameters": test_computation_params,
            "job_options": {"async": True}
        }
        
        # Act
        response = await test_client.post(
            "/tda/compute/vietoris-rips",
            json=request_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert
        assert response.status_code == 202
        result = response.json()
        
        assert "job_id" in result
        assert result["status"] == "submitted"
        assert "status_url" in result
        assert "estimated_completion" in result
    
    async def test_invalid_point_cloud_validation(
        self,
        test_client: AsyncClient,
        test_computation_params: dict
    ):
        """Test validation of invalid point cloud data."""
        # Arrange
        invalid_point_cloud = {
            "points": [[0.0], [1.0]],  # Inconsistent dimensions
            "dimension": 2
        }
        request_data = {
            "point_cloud": invalid_point_cloud,
            "parameters": test_computation_params
        }
        
        # Act
        response = await test_client.post(
            "/tda/compute/vietoris-rips",
            json=request_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert
        assert response.status_code == 400
        error = response.json()
        assert "error" in error
        assert error["error"]["code"] == "VALIDATION_ERROR"
    
    async def test_large_point_cloud_handling(
        self,
        test_client: AsyncClient,
        test_computation_params: dict
    ):
        """Test handling of large point clouds."""
        # Arrange
        import numpy as np
        large_points = np.random.rand(10000, 3).tolist()
        large_point_cloud = {
            "points": large_points,
            "dimension": 3
        }
        request_data = {
            "point_cloud": large_point_cloud,
            "parameters": {**test_computation_params, "async": True}
        }
        
        # Act
        response = await test_client.post(
            "/tda/compute/vietoris-rips",
            json=request_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert - Should auto-switch to async for large datasets
        assert response.status_code == 202
        result = response.json()
        assert result["status"] == "submitted"

@pytest.mark.asyncio
class TestJobManagementEndpoints:
    """Test job management endpoints."""
    
    async def test_job_status_retrieval(
        self,
        test_client: AsyncClient,
        test_database
    ):
        """Test job status retrieval."""
        # Arrange - Create a test job
        job_id = "test_job_123"
        await test_database.execute(
            "INSERT INTO jobs (job_id, algorithm, status, created_at) "
            "VALUES (:job_id, :algorithm, :status, NOW())",
            {"job_id": job_id, "algorithm": "vietoris_rips", "status": "running"}
        )
        
        # Act
        response = await test_client.get(
            f"/jobs/{job_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == job_id
        assert result["algorithm"] == "vietoris_rips"
        assert result["status"] == "running"
    
    async def test_job_cancellation(
        self,
        test_client: AsyncClient,
        test_database
    ):
        """Test job cancellation."""
        # Arrange
        job_id = "test_job_cancel_123"
        await test_database.execute(
            "INSERT INTO jobs (job_id, algorithm, status, created_at) "
            "VALUES (:job_id, :algorithm, :status, NOW())",
            {"job_id": job_id, "algorithm": "vietoris_rips", "status": "running"}
        )
        
        # Act
        response = await test_client.post(
            f"/jobs/{job_id}/cancel",
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "cancelled"
        
        # Verify database update
        job_record = await test_database.fetch_one(
            "SELECT status FROM jobs WHERE job_id = :job_id",
            {"job_id": job_id}
        )
        assert job_record["status"] == "cancelled"

@pytest.mark.asyncio
class TestDataUploadEndpoints:
    """Test data upload endpoints."""
    
    async def test_csv_file_upload(
        self,
        test_client: AsyncClient
    ):
        """Test CSV file upload."""
        # Arrange
        csv_content = "x,y,z\n0.0,0.0,0.0\n1.0,0.0,0.0\n0.0,1.0,0.0\n"
        files = {"file": ("test.csv", csv_content, "text/csv")}
        data = {
            "metadata": '{"name": "Test Data", "dimension": 3, "format": "csv"}'
        }
        
        # Act
        response = await test_client.post(
            "/data/upload",
            files=files,
            data=data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert
        assert response.status_code == 201
        result = response.json()
        assert "file_id" in result
        assert result["format"] == "csv"
        assert result["dimension"] == 3
        assert result["validation"]["is_valid"] is True
    
    async def test_invalid_file_format(
        self,
        test_client: AsyncClient
    ):
        """Test upload of invalid file format."""
        # Arrange
        invalid_content = "This is not valid CSV data"
        files = {"file": ("test.txt", invalid_content, "text/plain")}
        data = {
            "metadata": '{"name": "Invalid Data", "dimension": 3, "format": "csv"}'
        }
        
        # Act
        response = await test_client.post(
            "/data/upload",
            files=files,
            data=data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert
        assert response.status_code == 400
        error = response.json()
        assert "validation" in error["error"]["details"]
```

### **Kafka Integration Tests**

```python
# /tests/unit/test_kafka_integration.py
import pytest
import json
import asyncio
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer
from backend.services.kafka_producer import KafkaProducerService

@pytest.mark.asyncio
class TestKafkaProducer:
    """Test Kafka producer functionality."""
    
    async def test_job_message_publishing(
        self,
        test_kafka_producer: KafkaProducerService,
        test_settings
    ):
        """Test publishing job lifecycle messages."""
        # Arrange
        job_data = {
            "job_id": "test_job_123",
            "algorithm": "vietoris_rips",
            "status": "submitted",
            "user_id": "test_user",
            "parameters": {"max_edge_length": 2.0}
        }
        
        # Act
        success = await test_kafka_producer.send_job_submitted(**job_data)
        
        # Assert
        assert success is True
        
        # Verify message was published by consuming it
        consumer = AIOKafkaConsumer(
            test_settings.jobs_topic,
            bootstrap_servers=test_settings.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="earliest"
        )
        
        await consumer.start()
        try:
            # Get the published message
            message = await asyncio.wait_for(consumer.getone(), timeout=10.0)
            
            assert message.value["metadata"]["message_type"] == "job.submitted"
            assert message.value["payload"]["job_id"] == "test_job_123"
            assert message.value["payload"]["algorithm"] == "vietoris_rips"
            
        finally:
            await consumer.stop()
    
    async def test_result_message_publishing(
        self,
        test_kafka_producer: KafkaProducerService,
        test_settings
    ):
        """Test publishing computation result messages."""
        # Arrange
        result_data = {
            "job_id": "test_job_123",
            "result_id": "result_456",
            "result_type": "persistence_diagram",
            "persistence_pairs": [
                {"dimension": 0, "birth": 0.0, "death": 1.414},
                {"dimension": 1, "birth": 1.0, "death": "infinity"}
            ],
            "betti_numbers": {"beta_0": 1, "beta_1": 1}
        }
        
        # Act
        success = await test_kafka_producer.send_result_generated(**result_data)
        
        # Assert
        assert success is True
        
        # Verify message content
        consumer = AIOKafkaConsumer(
            test_settings.results_topic,
            bootstrap_servers=test_settings.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest"
        )
        
        await consumer.start()
        try:
            message = await asyncio.wait_for(consumer.getone(), timeout=10.0)
            
            assert message.value["metadata"]["message_type"] == "result.generated"
            assert message.value["payload"]["result_type"] == "persistence_diagram"
            assert len(message.value["payload"]["persistence_pairs"]) == 2
            
        finally:
            await consumer.stop()
    
    async def test_streaming_event_publishing(
        self,
        test_kafka_producer: KafkaProducerService,
        test_settings
    ):
        """Test publishing streaming events."""
        # Arrange
        event_data = {
            "event_type": "point_cloud_update",
            "stream_id": "sensor_001",
            "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Act
        success = await test_kafka_producer.send_streaming_event(**event_data)
        
        # Assert
        assert success is True
        
        # Verify event was published
        consumer = AIOKafkaConsumer(
            test_settings.events_topic,
            bootstrap_servers=test_settings.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest"
        )
        
        await consumer.start()
        try:
            message = await asyncio.wait_for(consumer.getone(), timeout=10.0)
            
            assert message.value["payload"]["event_type"] == "point_cloud_update"
            assert message.value["payload"]["stream_id"] == "sensor_001"
            assert len(message.value["payload"]["points"]) == 2
            
        finally:
            await consumer.stop()
    
    async def test_message_serialization_validation(
        self,
        test_kafka_producer: KafkaProducerService
    ):
        """Test message schema validation."""
        # Arrange - Invalid message structure
        invalid_data = {
            "job_id": "",  # Empty job_id should be invalid
            "algorithm": "invalid_algorithm",  # Invalid algorithm
            "status": "unknown_status"  # Invalid status
        }
        
        # Act & Assert
        with pytest.raises((ValueError, ValidationError)):
            await test_kafka_producer.send_job_submitted(**invalid_data)
    
    async def test_producer_error_handling(
        self,
        test_settings
    ):
        """Test producer error handling with invalid configuration."""
        # Arrange - Invalid Kafka configuration
        invalid_producer = KafkaProducerService(
            bootstrap_servers="invalid:9092"
        )
        
        # Act & Assert
        with pytest.raises(Exception):
            await invalid_producer.start()
            await invalid_producer.send_job_submitted(
                job_id="test", algorithm="vietoris_rips", status="submitted"
            )
```

### **C++ Core Integration Tests**

```python
# /tests/unit/test_cpp_integration.py
import pytest
import numpy as np
from backend.services.tda_service import TDAService
from backend.core.exceptions import TDAComputationError

@pytest.mark.asyncio
class TestCppTDAIntegration:
    """Test C++ TDA core integration."""
    
    def setup_method(self):
        """Setup TDA service for testing."""
        self.tda_service = TDAService()
    
    async def test_vietoris_rips_computation(self):
        """Test Vietoris-Rips computation with C++ core."""
        # Arrange
        points = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ]
        parameters = {
            "max_edge_length": 2.0,
            "max_dimension": 1,
            "num_threads": 1
        }
        
        # Act
        result = await self.tda_service.compute_vietoris_rips(
            point_cloud=points,
            **parameters
        )
        
        # Assert
        assert "persistence_pairs" in result
        assert "betti_numbers" in result
        assert "computation_time_ms" in result
        
        # Verify structure of persistence pairs
        for pair in result["persistence_pairs"]:
            assert "dimension" in pair
            assert "birth" in pair
            assert "death" in pair
            assert "persistence" in pair
            assert pair["dimension"] in [0, 1]
            assert pair["birth"] >= 0
    
    async def test_alpha_complex_computation(self):
        """Test Alpha complex computation."""
        # Arrange
        points = [
            [0.0, 0.0], [1.0, 0.0], [0.5, 0.866]  # Triangle
        ]
        parameters = {
            "max_alpha_value": 1.0,
            "max_dimension": 2
        }
        
        # Act
        result = await self.tda_service.compute_alpha_complex(
            point_cloud=points,
            **parameters
        )
        
        # Assert
        assert "persistence_pairs" in result
        assert "alpha_shape" in result
        assert result["alpha_shape"]["vertices"] == 3
        assert result["alpha_shape"]["edges"] >= 3
    
    async def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Arrange
        np.random.seed(42)
        points = np.random.rand(1000, 3).tolist()
        parameters = {
            "max_edge_length": 0.5,
            "max_dimension": 2,
            "num_threads": 4
        }
        
        # Act
        import time
        start_time = time.time()
        result = await self.tda_service.compute_vietoris_rips(
            point_cloud=points,
            **parameters
        )
        computation_time = time.time() - start_time
        
        # Assert
        assert computation_time < 30.0  # Should complete within 30 seconds
        assert len(result["persistence_pairs"]) > 0
        assert result["computation_time_ms"] > 0
    
    async def test_invalid_input_handling(self):
        """Test handling of invalid input data."""
        # Arrange
        invalid_points = []  # Empty point cloud
        parameters = {"max_edge_length": 2.0, "max_dimension": 1}
        
        # Act & Assert
        with pytest.raises(TDAComputationError):
            await self.tda_service.compute_vietoris_rips(
                point_cloud=invalid_points,
                **parameters
            )
    
    async def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during computation."""
        # Arrange
        points = np.random.rand(500, 3).tolist()
        parameters = {
            "max_edge_length": 1.0,
            "max_dimension": 2,
            "num_threads": 2
        }
        
        # Act
        result = await self.tda_service.compute_vietoris_rips(
            point_cloud=points,
            **parameters
        )
        
        # Assert
        assert "statistics" in result
        assert "memory_peak_mb" in result["statistics"]
        assert result["statistics"]["memory_peak_mb"] > 0
        assert result["statistics"]["memory_peak_mb"] < 1000  # Reasonable upper bound
    
    async def test_multithreading_performance(self):
        """Test multithreading performance improvement."""
        # Arrange
        points = np.random.rand(200, 3).tolist()
        base_params = {
            "max_edge_length": 1.0,
            "max_dimension": 1
        }
        
        # Act - Single threaded
        single_thread_params = {**base_params, "num_threads": 1}
        start_time = time.time()
        result_single = await self.tda_service.compute_vietoris_rips(
            point_cloud=points, **single_thread_params
        )
        single_time = time.time() - start_time
        
        # Act - Multi threaded
        multi_thread_params = {**base_params, "num_threads": 4}
        start_time = time.time()
        result_multi = await self.tda_service.compute_vietoris_rips(
            point_cloud=points, **multi_thread_params
        )
        multi_time = time.time() - start_time
        
        # Assert
        # Results should be identical
        assert len(result_single["persistence_pairs"]) == len(result_multi["persistence_pairs"])
        
        # Multi-threaded should be faster (or at least not significantly slower)
        speedup = single_time / multi_time
        assert speedup >= 0.8  # Allow for some overhead
```

## 🌊 End-to-End Testing

### **Complete Pipeline Tests**

```python
# /tests/e2e/test_complete_pipeline.py
import pytest
import asyncio
import json
import time
from httpx import AsyncClient
from aiokafka import AIOKafkaConsumer

@pytest.mark.asyncio
class TestCompletePipeline:
    """End-to-end tests covering the complete TDA pipeline."""
    
    async def test_synchronous_computation_flow(
        self,
        test_client: AsyncClient,
        test_point_cloud: dict,
        test_computation_params: dict,
        test_kafka_producer,
        test_settings
    ):
        """Test complete synchronous computation flow."""
        # Arrange
        request_data = {
            "point_cloud": test_point_cloud,
            "parameters": test_computation_params,
            "job_options": {
                "async": False,
                "cache_results": True,
                "notify_completion": True
            }
        }
        
        # Setup Kafka consumer to verify events
        consumer = AIOKafkaConsumer(
            test_settings.jobs_topic,
            test_settings.results_topic,
            bootstrap_servers=test_settings.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest"
        )
        await consumer.start()
        
        try:
            # Act
            response = await test_client.post(
                "/tda/compute/vietoris-rips",
                json=request_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Assert API Response
            assert response.status_code == 200
            result = response.json()
            
            assert result["status"] == "completed"
            assert "job_id" in result
            assert "results" in result
            assert len(result["results"]["persistence_pairs"]) > 0
            
            # Assert Kafka Events
            # Should receive job.submitted and job.completed events
            events = []
            timeout = 10.0
            start_time = time.time()
            
            while len(events) < 2 and (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(consumer.getone(), timeout=2.0)
                    events.append(message.value)
                except asyncio.TimeoutError:
                    break
            
            # Verify event sequence
            assert len(events) >= 1  # At least completion event
            
            completion_event = next(
                (e for e in events if e["metadata"]["message_type"] == "job.completed"), 
                None
            )
            assert completion_event is not None
            assert completion_event["payload"]["job_id"] == result["job_id"]
            
        finally:
            await consumer.stop()
    
    async def test_asynchronous_job_flow(
        self,
        test_client: AsyncClient,
        test_point_cloud: dict,
        test_computation_params: dict,
        test_kafka_producer,
        test_settings
    ):
        """Test complete asynchronous job processing flow."""
        # Arrange
        request_data = {
            "point_cloud": test_point_cloud,
            "parameters": test_computation_params,
            "job_options": {"async": True}
        }
        
        # Setup Kafka consumer
        consumer = AIOKafkaConsumer(
            test_settings.jobs_topic,
            bootstrap_servers=test_settings.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest"
        )
        await consumer.start()
        
        try:
            # Act - Submit job
            response = await test_client.post(
                "/tda/compute/vietoris-rips",
                json=request_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Assert - Job submitted
            assert response.status_code == 202
            job_data = response.json()
            assert job_data["status"] == "submitted"
            job_id = job_data["job_id"]
            
            # Wait for job events
            events = []
            timeout = 30.0
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(consumer.getone(), timeout=5.0)
                    event = message.value
                    if event["payload"].get("job_id") == job_id:
                        events.append(event)
                        
                        # Break if we get completion event
                        if event["metadata"]["message_type"] in ["job.completed", "job.failed"]:
                            break
                except asyncio.TimeoutError:
                    break
            
            # Assert - Events received
            assert len(events) >= 1
            
            # Check job status via API
            status_response = await test_client.get(
                f"/jobs/{job_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert status_response.status_code == 200
            job_status = status_response.json()
            assert job_status["job_id"] == job_id
            assert job_status["status"] in ["running", "completed"]
            
            # If job completed, verify results
            if job_status["status"] == "completed":
                assert "results" in job_status
                assert len(job_status["results"]["persistence_pairs"]) > 0
            
        finally:
            await consumer.stop()
    
    async def test_streaming_data_processing(
        self,
        test_client: AsyncClient,
        test_kafka_producer,
        test_settings
    ):
        """Test real-time streaming data processing."""
        # Arrange
        stream_events = [
            {
                "topic": "tda_events",
                "event": {
                    "type": "point_cloud_update",
                    "data": {
                        "stream_id": "test_sensor_001",
                        "points": [[i, i, i] for i in range(10)],
                        "timestamp": "2023-12-01T12:00:00Z"
                    }
                },
                "options": {"partition_key": "test_sensor_001"}
            },
            {
                "topic": "tda_events", 
                "event": {
                    "type": "point_cloud_update",
                    "data": {
                        "stream_id": "test_sensor_001",
                        "points": [[i+10, i+10, i+10] for i in range(10)],
                        "timestamp": "2023-12-01T12:00:10Z"
                    }
                },
                "options": {"partition_key": "test_sensor_001"}
            }
        ]
        
        # Setup result consumer
        consumer = AIOKafkaConsumer(
            test_settings.results_topic,
            bootstrap_servers=test_settings.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest"
        )
        await consumer.start()
        
        try:
            # Act - Publish streaming events
            for event_data in stream_events:
                response = await test_client.post(
                    "/stream/publish",
                    json=event_data,
                    headers={"Authorization": "Bearer test-token"}
                )
                assert response.status_code == 202
                
                # Small delay between events
                await asyncio.sleep(1.0)
            
            # Wait for stream processing results
            results = []
            timeout = 60.0  # Longer timeout for stream processing
            start_time = time.time()
            
            while len(results) < 1 and (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(consumer.getone(), timeout=10.0)
                    result = message.value
                    
                    # Check if this is a TDA computation result
                    if ("stream_processing" in result.get("payload", {}) and 
                        "persistence_pairs" in result.get("payload", {})):
                        results.append(result)
                        
                except asyncio.TimeoutError:
                    break
            
            # Assert - Stream processing produced results
            assert len(results) >= 1
            
            for result in results:
                assert "persistence_pairs" in result["payload"]
                assert "window_metadata" in result["payload"]
                assert "stream_processing" in result["payload"]
            
        finally:
            await consumer.stop()
    
    async def test_file_upload_to_computation_flow(
        self,
        test_client: AsyncClient,
        test_computation_params: dict
    ):
        """Test complete flow from file upload to computation."""
        # Arrange - Create test CSV file
        csv_content = """x,y,z
0.0,0.0,0.0
1.0,0.0,0.0
0.0,1.0,0.0
0.0,0.0,1.0
1.0,1.0,0.0
1.0,0.0,1.0
0.0,1.0,1.0
1.0,1.0,1.0"""
        
        files = {"file": ("test_data.csv", csv_content, "text/csv")}
        metadata = {
            "metadata": json.dumps({
                "name": "Test Point Cloud",
                "description": "Test dataset for E2E testing",
                "dimension": 3,
                "format": "csv"
            })
        }
        
        # Act - Upload file
        upload_response = await test_client.post(
            "/data/upload",
            files=files,
            data=metadata,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert - File uploaded successfully
        assert upload_response.status_code == 201
        file_data = upload_response.json()
        assert file_data["validation"]["is_valid"] is True
        file_id = file_data["file_id"]
        
        # Act - Compute TDA using uploaded file
        computation_request = {
            "file_id": file_id,
            "parameters": test_computation_params,
            "job_options": {"async": False}
        }
        
        compute_response = await test_client.post(
            "/tda/compute/vietoris-rips",
            json=computation_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert - Computation successful
        assert compute_response.status_code == 200
        result = compute_response.json()
        
        assert result["status"] == "completed"
        assert "results" in result
        assert len(result["results"]["persistence_pairs"]) > 0
        assert result["input_data"]["file_id"] == file_id
        
        # Act - Retrieve file information
        file_response = await test_client.get(
            f"/data/{file_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Assert - File information accessible
        assert file_response.status_code == 200
        retrieved_file = file_response.json()
        assert retrieved_file["file_id"] == file_id
        assert retrieved_file["status"] == "processed"
```

## 🚀 Performance Testing

### **Load Testing Framework**

```python
# /tests/performance/test_load_performance.py
import pytest
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

@pytest.mark.performance
class TestAPIPerformance:
    """Performance testing for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_concurrent_computation_requests(self):
        """Test API performance under concurrent load."""
        # Arrange
        base_url = "http://localhost:8001/api/v1"
        headers = {"Authorization": "Bearer test-token"}
        
        test_data = {
            "point_cloud": {
                "points": [[i, i, i] for i in range(100)],
                "dimension": 3
            },
            "parameters": {
                "max_edge_length": 2.0,
                "max_dimension": 1,
                "num_threads": 2
            },
            "job_options": {"async": False}
        }
        
        concurrent_requests = 20
        response_times = []
        successful_requests = 0
        
        # Act
        async def make_request(session, request_id):
            start_time = time.time()
            try:
                async with session.post(
                    f"{base_url}/tda/compute/vietoris-rips",
                    json=test_data,
                    headers=headers
                ) as response:
                    await response.json()
                    end_time = time.time()
                    return {
                        "request_id": request_id,
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "success": response.status == 200
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "status_code": 500,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                make_request(session, i) for i in range(concurrent_requests)
            ]
            results = await asyncio.gather(*tasks)
        
        # Assert
        response_times = [r["response_time"] for r in results if r["success"]]
        successful_requests = sum(1 for r in results if r["success"])
        
        # Performance assertions
        assert successful_requests >= concurrent_requests * 0.9  # 90% success rate
        assert statistics.mean(response_times) < 10.0  # Average < 10s
        assert max(response_times) < 30.0  # Max < 30s
        assert statistics.median(response_times) < 5.0  # Median < 5s
        
        print(f"Performance Results:")
        print(f"  Successful requests: {successful_requests}/{concurrent_requests}")
        print(f"  Average response time: {statistics.mean(response_times):.2f}s")
        print(f"  Median response time: {statistics.median(response_times):.2f}s")
        print(f"  95th percentile: {sorted(response_times)[int(0.95 * len(response_times))]:.2f}s")
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self):
        """Measure API throughput over time."""
        # Arrange
        base_url = "http://localhost:8001/api/v1"
        headers = {"Authorization": "Bearer test-token"}
        
        test_data = {
            "point_cloud": {
                "points": [[i, i] for i in range(50)],  # Smaller dataset for faster processing
                "dimension": 2
            },
            "parameters": {
                "max_edge_length": 3.0,
                "max_dimension": 1,
                "num_threads": 1
            }
        }
        
        duration_seconds = 60  # Test for 1 minute
        request_interval = 0.5  # Request every 500ms
        
        # Act
        start_time = time.time()
        completed_requests = 0
        total_requests = 0
        
        async with aiohttp.ClientSession() as session:
            while (time.time() - start_time) < duration_seconds:
                total_requests += 1
                
                try:
                    async with session.post(
                        f"{base_url}/tda/compute/vietoris-rips",
                        json=test_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            completed_requests += 1
                        
                except Exception:
                    pass  # Count as failed request
                
                await asyncio.sleep(request_interval)
        
        actual_duration = time.time() - start_time
        throughput = completed_requests / actual_duration
        
        # Assert
        assert throughput >= 1.0  # At least 1 request per second
        assert (completed_requests / total_requests) >= 0.8  # 80% success rate
        
        print(f"Throughput Results:")
        print(f"  Completed requests: {completed_requests}/{total_requests}")
        print(f"  Duration: {actual_duration:.1f}s") 
        print(f"  Throughput: {throughput:.2f} requests/second")

@pytest.mark.performance
class TestKafkaPerformance:
    """Performance testing for Kafka integration."""
    
    @pytest.mark.asyncio
    async def test_kafka_producer_throughput(
        self,
        test_kafka_producer,
        test_settings
    ):
        """Test Kafka producer throughput."""
        # Arrange
        num_messages = 1000
        batch_size = 100
        
        # Act
        start_time = time.time()
        successful_sends = 0
        
        for batch in range(0, num_messages, batch_size):
            batch_tasks = []
            
            for i in range(batch, min(batch + batch_size, num_messages)):
                task = test_kafka_producer.send_job_submitted(
                    job_id=f"perf_test_{i}",
                    algorithm="vietoris_rips",
                    status="submitted",
                    user_id="perf_test_user"
                )
                batch_tasks.append(task)
            
            # Wait for batch completion
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            successful_sends += sum(1 for r in results if r is True)
        
        duration = time.time() - start_time
        throughput = successful_sends / duration
        
        # Assert
        assert successful_sends >= num_messages * 0.95  # 95% success rate
        assert throughput >= 100  # At least 100 messages/second
        
        print(f"Kafka Producer Performance:")
        print(f"  Messages sent: {successful_sends}/{num_messages}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.0f} messages/second")

@pytest.mark.performance
class TestMemoryPerformance:
    """Memory usage and leak testing."""
    
    def test_memory_usage_under_load(self):
        """Test memory usage during intensive computation."""
        import psutil
        import numpy as np
        from backend.services.tda_service import TDAService
        
        # Arrange
        tda_service = TDAService()
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        max_memory = initial_memory
        
        # Act - Perform multiple computations
        for i in range(10):
            # Generate large point cloud
            points = np.random.rand(1000, 3).tolist()
            
            # Compute TDA
            result = asyncio.run(
                tda_service.compute_vietoris_rips(
                    point_cloud=points,
                    max_edge_length=0.5,
                    max_dimension=1,
                    num_threads=2
                )
            )
            
            # Monitor memory
            current_memory = process.memory_info().rss / (1024 * 1024)
            max_memory = max(max_memory, current_memory)
            
            # Verify computation completed
            assert len(result["persistence_pairs"]) > 0
        
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - initial_memory
        
        # Assert
        assert memory_growth < 500  # Less than 500MB growth
        assert max_memory < initial_memory + 1000  # Peak < 1GB above initial
        
        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Peak: {max_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
```

## 🧪 Test Execution Framework

### **Test Runner Configuration**

```python
# /tests/pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    -q
    --strict-markers
    --disable-warnings
    --tb=short
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests  
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow running tests
    kafka: Tests requiring Kafka
    flink: Tests requiring Flink
    cpp: Tests requiring C++ core

# Async test configuration
asyncio_mode = auto

# Coverage configuration
addopts = --cov=backend --cov-report=html --cov-report=term-missing
```

### **Makefile for Test Execution**

```makefile
# /tests/Makefile
.PHONY: test test-unit test-integration test-e2e test-performance setup-test-env cleanup-test-env

# Test environment setup
setup-test-env:
	docker-compose -f docker-compose.test.yml up -d
	sleep 30  # Wait for services to be ready
	docker-compose -f docker-compose.test.yml exec test-api python -m pytest tests/health_check.py

cleanup-test-env:
	docker-compose -f docker-compose.test.yml down -v

# Test execution
test-unit:
	pytest tests/unit/ -v -m "not slow"

test-integration:
	pytest tests/integration/ -v

test-e2e: setup-test-env
	pytest tests/e2e/ -v -s
	$(MAKE) cleanup-test-env

test-performance: setup-test-env
	pytest tests/performance/ -v -s -m performance
	$(MAKE) cleanup-test-env

test-all: setup-test-env
	pytest tests/ -v --maxfail=5
	$(MAKE) cleanup-test-env

# Coverage reporting
test-coverage: setup-test-env
	pytest tests/ --cov=backend --cov-report=html --cov-report=term
	$(MAKE) cleanup-test-env

# Continuous testing
test-watch:
	pytest-watch tests/unit/ -- -v

# Test data generation
generate-test-data:
	python tests/utils/data_generator.py

# Lint tests
lint-tests:
	flake8 tests/
	mypy tests/
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Create comprehensive end-to-end test suite for TDA Platform Backend", "status": "completed"}, {"id": "2", "content": "Implement API endpoint integration tests", "status": "completed"}, {"id": "3", "content": "Create Kafka message flow testing", "status": "completed"}, {"id": "4", "content": "Implement Flink stream processing tests", "status": "completed"}, {"id": "5", "content": "Create Docker-based test environment configuration", "status": "completed"}, {"id": "6", "content": "Implement performance and load testing scenarios", "status": "completed"}]
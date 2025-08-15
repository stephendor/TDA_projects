# Kafka Architecture Design for TDA Streaming Pipeline

## Table of Contents
1. [Overview](#overview)
2. [Topic Structure Design](#topic-structure-design)
3. [Partition Strategy](#partition-strategy)
4. [Message Schema Design](#message-schema-design)
5. [Scalability Considerations](#scalability-considerations)
6. [Integration Points](#integration-points)
7. [Configuration Recommendations](#configuration-recommendations)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Security and Compliance](#security-and-compliance)
10. [Operational Procedures](#operational-procedures)

## Overview

The TDA (Topological Data Analysis) Platform employs Apache Kafka as the central streaming backbone for real-time data processing, job orchestration, and event-driven communication between distributed components. This architecture enables scalable, fault-tolerant processing of computationally intensive TDA operations.

### Core Objectives
- **High Throughput**: Handle thousands of TDA computation requests per second
- **Low Latency**: Sub-second event processing for real-time notifications
- **Fault Tolerance**: Guarantee message delivery and processing even with component failures
- **Scalability**: Horizontal scaling to handle growing computational demands
- **Event Sourcing**: Complete audit trail of all operations and state changes

### System Components
- **FastAPI Backend**: Event producers for job submissions and API operations
- **Apache Flink**: Stream processing engine for complex event processing
- **C++ TDA Core**: High-performance computation engine (event consumer)
- **PostgreSQL**: Persistent storage for results and metadata
- **Redis**: Caching layer and session management

## Topic Structure Design

### 1. tda_jobs
**Purpose**: Job submission and lifecycle management events

**Event Types**:
- `job.submitted`: New TDA computation job submitted
- `job.queued`: Job added to processing queue
- `job.started`: Job execution started
- `job.progress`: Progress updates during execution
- `job.completed`: Job completed successfully
- `job.failed`: Job execution failed
- `job.cancelled`: Job cancelled by user or system

**Key Characteristics**:
- **Retention**: 7 days (configurable via `result_retention_days`)
- **Compaction**: None (log-based retention for audit trail)
- **Cleanup Policy**: Delete based on time and size limits

### 2. tda_results
**Purpose**: Computation results and persistence diagrams

**Event Types**:
- `result.persistence_diagram`: Computed persistence diagram data
- `result.betti_numbers`: Betti number calculations
- `result.filtration_data`: Filtration sequence information
- `result.metadata`: Computation metadata and statistics
- `result.visualization`: Generated visualization artifacts

**Key Characteristics**:
- **Retention**: 30 days (results cached in PostgreSQL for longer retention)
- **Compaction**: Key-based compaction (latest result per job_id)
- **Cleanup Policy**: Compact + delete for optimal storage

### 3. tda_events
**Purpose**: System events and notifications

**Event Types**:
- `system.health_check`: Health status updates
- `system.error`: System-level errors and alerts
- `system.metric`: Performance metrics and monitoring data
- `user.notification`: User-facing notifications
- `admin.alert`: Administrative alerts and warnings

**Key Characteristics**:
- **Retention**: 14 days
- **Compaction**: None (event sequence important for debugging)
- **Cleanup Policy**: Delete based on time

### 4. tda_uploads
**Purpose**: File upload events and data ingestion

**Event Types**:
- `upload.started`: File upload initiated
- `upload.progress`: Upload progress updates
- `upload.completed`: Upload completed successfully
- `upload.failed`: Upload failed
- `upload.validated`: File format validation completed
- `upload.processed`: Data preprocessing completed

**Key Characteristics**:
- **Retention**: 3 days (uploads are temporary)
- **Compaction**: None
- **Cleanup Policy**: Delete based on time and size

### 5. tda_errors
**Purpose**: Error events and failure tracking

**Event Types**:
- `error.computation`: TDA computation errors
- `error.validation`: Data validation errors
- `error.system`: System-level errors
- `error.timeout`: Operation timeout errors
- `error.resource`: Resource exhaustion errors

**Key Characteristics**:
- **Retention**: 30 days (longer retention for error analysis)
- **Compaction**: None (all errors tracked for patterns)
- **Cleanup Policy**: Delete based on time

## Partition Strategy

### Partitioning Scheme

#### Primary Strategy: Job-Based Partitioning
```
Partition Key: job_id (for job-related topics)
Hash Function: murmur2 (Kafka default)
Partition Assignment: hash(job_id) % partition_count
```

#### Benefits:
- **Ordered Processing**: All events for a job processed in sequence
- **Consumer Affinity**: Same consumer processes all events for a job
- **State Locality**: Job state maintained on single consumer
- **Load Distribution**: Jobs distributed across partitions evenly

### Partition Configuration

#### Development Environment
```yaml
tda_jobs:
  partitions: 3
  replication_factor: 1
  
tda_results:
  partitions: 3
  replication_factor: 1
  
tda_events:
  partitions: 1  # System events don't need partitioning
  replication_factor: 1
  
tda_uploads:
  partitions: 2
  replication_factor: 1
  
tda_errors:
  partitions: 1  # Centralized error collection
  replication_factor: 1
```

#### Production Environment
```yaml
tda_jobs:
  partitions: 12  # Supports 12 concurrent job processors
  replication_factor: 3
  min_in_sync_replicas: 2
  
tda_results:
  partitions: 12
  replication_factor: 3
  min_in_sync_replicas: 2
  
tda_events:
  partitions: 3
  replication_factor: 3
  min_in_sync_replicas: 2
  
tda_uploads:
  partitions: 6
  replication_factor: 3
  min_in_sync_replicas: 2
  
tda_errors:
  partitions: 3
  replication_factor: 3
  min_in_sync_replicas: 2
```

### Replication Strategy

#### High Availability Configuration
- **Replication Factor**: 3 (production), 1 (development)
- **Min In-Sync Replicas**: 2 (production), 1 (development)
- **Unclean Leader Election**: Disabled (prevent data loss)
- **Preferred Replica Election**: Enabled (optimize leadership distribution)

#### Rack Awareness
```yaml
broker.rack: rack-1  # Broker 0,3,6
broker.rack: rack-2  # Broker 1,4,7
broker.rack: rack-3  # Broker 2,5,8
```

### Retention Policies

#### Time-Based Retention
```yaml
tda_jobs:
  retention.ms: 604800000      # 7 days
  retention.bytes: 1073741824  # 1GB per partition

tda_results:
  retention.ms: 2592000000     # 30 days
  retention.bytes: 5368709120  # 5GB per partition

tda_events:
  retention.ms: 1209600000     # 14 days
  retention.bytes: 536870912   # 512MB per partition

tda_uploads:
  retention.ms: 259200000      # 3 days
  retention.bytes: 2147483648  # 2GB per partition

tda_errors:
  retention.ms: 2592000000     # 30 days
  retention.bytes: 1073741824  # 1GB per partition
```

#### Compaction Settings
```yaml
# For tda_results topic (latest result per job)
cleanup.policy: compact
segment.ms: 86400000          # 24 hours
min.cleanable.dirty.ratio: 0.1
max.compaction.lag.ms: 3600000  # 1 hour
```

## Message Schema Design

### Base Event Schema
All events follow a consistent base schema with versioning support:

```json
{
  "schema_version": "1.0",
  "event_id": "uuid",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "event_type": "job.submitted",
  "source": "tda_backend_api",
  "correlation_id": "request_uuid",
  "headers": {
    "user_id": "user123",
    "api_version": "v1",
    "client_ip": "192.168.1.100"
  },
  "data": {
    // Event-specific payload
  }
}
```

### Job Events Schema

#### job.submitted
```json
{
  "schema_version": "1.0",
  "event_id": "job_001_submit_uuid",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "event_type": "job.submitted",
  "source": "tda_backend_api",
  "correlation_id": "api_request_uuid",
  "headers": {
    "user_id": "user123",
    "api_version": "v1",
    "client_ip": "192.168.1.100"
  },
  "data": {
    "job_id": "job_001",
    "user_id": "user123",
    "algorithm": "vietoris_rips",
    "parameters": {
      "max_dimension": 2,
      "max_edge_length": 1.0,
      "resolution": 100
    },
    "input_data": {
      "data_id": "upload_123",
      "format": "point_cloud",
      "size_bytes": 1048576,
      "point_count": 1000
    },
    "priority": "normal",
    "estimated_runtime_seconds": 30
  }
}
```

#### job.progress
```json
{
  "schema_version": "1.0",
  "event_id": "job_001_progress_uuid",
  "timestamp": "2024-01-15T10:30:15.000Z",
  "event_type": "job.progress",
  "source": "tda_cpp_engine",
  "correlation_id": "job_001",
  "data": {
    "job_id": "job_001",
    "progress_percentage": 45.5,
    "current_stage": "computing_persistence",
    "stages_completed": ["data_validation", "filtration_construction"],
    "stages_remaining": ["persistence_computation", "result_serialization"],
    "estimated_completion": "2024-01-15T10:30:45.000Z",
    "memory_usage_mb": 256,
    "cpu_usage_percentage": 85.2
  }
}
```

### Result Events Schema

#### result.persistence_diagram
```json
{
  "schema_version": "1.0",
  "event_id": "result_001_pd_uuid",
  "timestamp": "2024-01-15T10:30:45.000Z",
  "event_type": "result.persistence_diagram",
  "source": "tda_cpp_engine",
  "correlation_id": "job_001",
  "data": {
    "job_id": "job_001",
    "algorithm": "vietoris_rips",
    "dimension": 1,
    "intervals": [
      {"birth": 0.0, "death": 0.5},
      {"birth": 0.1, "death": 0.8},
      {"birth": 0.2, "death": "inf"}
    ],
    "betti_numbers": [1, 3, 0],
    "computation_time_seconds": 28.5,
    "result_size_bytes": 2048,
    "storage_location": "s3://tda-results/job_001/persistence_diagram.json"
  }
}
```

### Upload Events Schema

#### upload.completed
```json
{
  "schema_version": "1.0",
  "event_id": "upload_123_complete_uuid",
  "timestamp": "2024-01-15T10:25:00.000Z",
  "event_type": "upload.completed",
  "source": "tda_upload_service",
  "correlation_id": "upload_request_uuid",
  "data": {
    "upload_id": "upload_123",
    "user_id": "user123",
    "filename": "point_cloud_data.csv",
    "file_size_bytes": 1048576,
    "content_type": "text/csv",
    "checksum_sha256": "a1b2c3d4e5f6...",
    "storage_location": "s3://tda-uploads/user123/upload_123.csv",
    "validation_status": "passed",
    "data_format": {
      "type": "point_cloud",
      "dimensions": 3,
      "point_count": 1000,
      "bounds": {
        "min": [-1.0, -1.0, -1.0],
        "max": [1.0, 1.0, 1.0]
      }
    }
  }
}
```

### Error Events Schema

#### error.computation
```json
{
  "schema_version": "1.0",
  "event_id": "error_001_uuid",
  "timestamp": "2024-01-15T10:30:30.000Z",
  "event_type": "error.computation",
  "source": "tda_cpp_engine",
  "correlation_id": "job_001",
  "data": {
    "job_id": "job_001",
    "error_code": "MEMORY_LIMIT_EXCEEDED",
    "error_message": "Computation exceeded memory limit of 8GB",
    "error_details": {
      "memory_requested_mb": 12288,
      "memory_available_mb": 8192,
      "stage": "persistence_computation",
      "input_size": 1000000
    },
    "stack_trace": "...",
    "recovery_suggestions": [
      "Reduce input data size",
      "Increase memory limit",
      "Use sparse algorithm variant"
    ]
  }
}
```

### Schema Evolution Strategy

#### Version Management
- **Backward Compatibility**: New fields added as optional
- **Schema Registry**: Confluent Schema Registry for schema validation
- **Versioning**: Semantic versioning (major.minor format)
- **Migration**: Gradual rollout with dual schema support

#### Compatibility Matrix
```yaml
schema_versions:
  "1.0": "Initial release - all events"
  "1.1": "Added optional fields - backward compatible"
  "1.2": "Enhanced error details - backward compatible"
  "2.0": "Breaking changes - requires migration"
```

## Scalability Considerations

### Multi-Broker Setup

#### Production Cluster Configuration
```yaml
cluster_size: 9 brokers
broker_distribution:
  - rack-1: [broker-0, broker-3, broker-6]
  - rack-2: [broker-1, broker-4, broker-7] 
  - rack-3: [broker-2, broker-5, broker-8]

hardware_specs:
  cpu: 16 cores
  memory: 64GB
  storage: 2TB NVMe SSD
  network: 10Gbps
```

#### Development Cluster Configuration
```yaml
cluster_size: 3 brokers
broker_distribution:
  - single_rack: [broker-0, broker-1, broker-2]

hardware_specs:
  cpu: 4 cores
  memory: 16GB
  storage: 500GB SSD
  network: 1Gbps
```

### Consumer Group Strategies

#### Job Processing Consumer Groups
```yaml
tda_job_processors:
  group_id: "tda_job_processors"
  consumers: 12  # Match partition count
  assignment_strategy: "RoundRobinAssignor"
  auto_offset_reset: "earliest"
  enable_auto_commit: false  # Manual commit for reliability
  session_timeout_ms: 30000
  heartbeat_interval_ms: 10000
  max_poll_records: 100
  fetch_min_bytes: 1024
  fetch_max_wait_ms: 500
```

#### Real-time Notification Consumer Groups
```yaml
tda_notification_service:
  group_id: "tda_notifications"
  consumers: 3
  assignment_strategy: "RangeAssignor"
  auto_offset_reset: "latest"  # Only new events for notifications
  enable_auto_commit: true
  session_timeout_ms: 10000
  heartbeat_interval_ms: 3000
  max_poll_records: 500
  fetch_min_bytes: 1
  fetch_max_wait_ms: 100
```

#### Analytics Consumer Groups
```yaml
tda_analytics_processors:
  group_id: "tda_analytics"
  consumers: 6
  assignment_strategy: "CooperativeStickyAssignor"
  auto_offset_reset: "earliest"
  enable_auto_commit: false
  session_timeout_ms: 60000
  heartbeat_interval_ms: 20000
  max_poll_records: 1000
  fetch_min_bytes: 65536
  fetch_max_wait_ms: 5000
```

### Load Balancing Approaches

#### Producer Load Balancing
```python
# Custom partitioner for job distribution
class JobPartitioner:
    def partition(self, topic, key, all_partitions, available_partitions):
        if topic == "tda_jobs":
            # Distribute based on job priority and size
            job_hash = hash(key) if key else random.randint(0, 1000000)
            return job_hash % len(available_partitions)
        return hash(key) % len(available_partitions) if key else 0

# Sticky partitioning for better batching
producer_config = {
    'partitioner': JobPartitioner(),
    'enable.idempotence': True,
    'acks': 'all',
    'retries': 2147483647,
    'max.in.flight.requests.per.connection': 5,
    'batch.size': 16384,
    'linger.ms': 10,
    'compression.type': 'lz4'
}
```

#### Consumer Load Balancing
```python
# Dynamic consumer scaling based on lag
class DynamicConsumerScaler:
    def scale_consumers(self, lag_threshold=1000):
        for consumer_group in self.consumer_groups:
            lag = self.get_consumer_lag(consumer_group)
            if lag > lag_threshold:
                self.add_consumer_instance(consumer_group)
            elif lag < lag_threshold * 0.3:
                self.remove_consumer_instance(consumer_group)
```

### Performance Optimization

#### Broker Optimization
```yaml
# Broker configuration for high throughput
server.properties:
  # Network settings
  num.network.threads: 8
  num.io.threads: 16
  socket.send.buffer.bytes: 102400
  socket.receive.buffer.bytes: 102400
  socket.request.max.bytes: 104857600
  
  # Log settings
  num.replica.fetchers: 4
  replica.fetch.max.bytes: 1048576
  replica.fetch.response.max.bytes: 10485760
  
  # Compression
  compression.type: lz4
  
  # Memory settings
  replica.socket.receive.buffer.bytes: 65536
  controller.socket.timeout.ms: 30000
  
  # JVM settings
  KAFKA_HEAP_OPTS: "-Xmx32g -Xms32g"
  KAFKA_JVM_PERFORMANCE_OPTS: "-XX:+UseG1GC -XX:MaxGCPauseMillis=20"
```

#### Producer Optimization
```python
# High-throughput producer configuration
producer_config = {
    'bootstrap.servers': 'kafka-cluster:9092',
    'acks': 'all',
    'enable.idempotence': True,
    'max.in.flight.requests.per.connection': 5,
    'retries': 2147483647,
    'delivery.timeout.ms': 300000,
    
    # Batching for throughput
    'batch.size': 65536,  # 64KB batches
    'linger.ms': 10,      # Wait 10ms for batching
    'compression.type': 'lz4',
    
    # Memory allocation
    'buffer.memory': 67108864,  # 64MB buffer
    'max.request.size': 1048576,  # 1MB max message
    
    # Reliability
    'request.timeout.ms': 30000,
    'metadata.max.age.ms': 300000,
}
```

#### Consumer Optimization
```python
# High-throughput consumer configuration
consumer_config = {
    'bootstrap.servers': 'kafka-cluster:9092',
    'group.id': 'tda_processors',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,
    
    # Batching for efficiency
    'max.poll.records': 1000,
    'fetch.min.bytes': 65536,  # 64KB minimum fetch
    'fetch.max.wait.ms': 500,
    
    # Memory allocation
    'receive.buffer.bytes': 262144,  # 256KB
    'max.partition.fetch.bytes': 1048576,  # 1MB
    
    # Session management
    'session.timeout.ms': 30000,
    'heartbeat.interval.ms': 10000,
    'max.poll.interval.ms': 300000,
}
```

## Integration Points

### FastAPI Producer Integration

#### Event Publishing Service
```python
# kafka/producer.py
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from tda_backend.config import settings

class TDAEventProducer:
    def __init__(self):
        self.producer_config = {
            'bootstrap.servers': ','.join(settings.kafka_bootstrap_servers),
            'acks': 'all',
            'enable.idempotence': True,
            'max.in.flight.requests.per.connection': 5,
            'retries': 2147483647,
            'batch.size': 16384,
            'linger.ms': 10,
            'compression.type': 'lz4',
            'client.id': 'tda_backend_producer'
        }
        self.producer = Producer(self.producer_config)
        self.schema_version = "1.0"
    
    async def publish_event(
        self,
        topic: str,
        event_type: str,
        data: Dict[str, Any],
        key: Optional[str] = None,
        correlation_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """Publish an event to Kafka topic."""
        event_id = str(uuid.uuid4())
        
        event = {
            "schema_version": self.schema_version,
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "source": "tda_backend_api",
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "headers": headers or {},
            "data": data
        }
        
        topic_name = settings.get_kafka_topic(topic)
        
        # Async delivery callback
        def delivery_callback(err, msg):
            if err:
                logger.error(f"Message delivery failed: {err}")
            else:
                logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
        
        # Produce message
        self.producer.produce(
            topic=topic_name,
            key=key,
            value=json.dumps(event).encode('utf-8'),
            callback=delivery_callback
        )
        
        # Flush to ensure delivery
        self.producer.flush(timeout=10)
        
        return event_id
    
    async def publish_job_event(self, job_id: str, event_type: str, data: Dict[str, Any]) -> str:
        """Publish job-related event."""
        return await self.publish_event(
            topic="jobs",
            event_type=event_type,
            data=data,
            key=job_id,
            correlation_id=job_id
        )
    
    async def publish_result_event(self, job_id: str, result_data: Dict[str, Any]) -> str:
        """Publish computation result."""
        return await self.publish_event(
            topic="results",
            event_type="result.persistence_diagram",
            data=result_data,
            key=job_id,
            correlation_id=job_id
        )
    
    async def publish_upload_event(self, upload_id: str, event_type: str, data: Dict[str, Any]) -> str:
        """Publish upload-related event."""
        return await self.publish_event(
            topic="uploads",
            event_type=event_type,
            data=data,
            key=upload_id,
            correlation_id=upload_id
        )
    
    def close(self):
        """Close producer connection."""
        self.producer.flush()

# Dependency injection for FastAPI
event_producer = TDAEventProducer()

async def get_event_producer() -> TDAEventProducer:
    return event_producer
```

#### API Endpoint Integration
```python
# api/v1/tda.py
from fastapi import APIRouter, Depends, HTTPException
from tda_backend.kafka.producer import TDAEventProducer, get_event_producer

router = APIRouter()

@router.post("/compute/vietoris-rips")
async def compute_vietoris_rips(
    request: VietorisRipsRequest,
    producer: TDAEventProducer = Depends(get_event_producer)
):
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Publish job submission event
    await producer.publish_job_event(
        job_id=job_id,
        event_type="job.submitted",
        data={
            "job_id": job_id,
            "user_id": request.user_id,
            "algorithm": "vietoris_rips",
            "parameters": request.parameters.dict(),
            "input_data": request.input_data.dict(),
            "priority": "normal",
            "estimated_runtime_seconds": estimate_runtime(request)
        }
    )
    
    return {"job_id": job_id, "status": "submitted"}
```

### Flink Consumer Integration

#### Stream Processing Job
```python
# flink/tda_stream_processor.py
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types

def create_tda_stream_processor():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(settings.flink_parallelism)
    
    # Kafka consumer configuration
    kafka_consumer_props = {
        'bootstrap.servers': ','.join(settings.kafka_bootstrap_servers),
        'group.id': 'flink_tda_processor',
        'auto.offset.reset': 'earliest'
    }
    
    # Job events stream
    job_consumer = FlinkKafkaConsumer(
        settings.get_kafka_topic('jobs'),
        SimpleStringSchema(),
        kafka_consumer_props
    )
    
    job_stream = env.add_source(job_consumer)
    
    # Process job events
    processed_jobs = job_stream.map(
        lambda event: process_job_event(json.loads(event)),
        output_type=Types.STRING()
    )
    
    # Results producer
    result_producer = FlinkKafkaProducer(
        settings.get_kafka_topic('results'),
        SimpleStringSchema(),
        {'bootstrap.servers': ','.join(settings.kafka_bootstrap_servers)}
    )
    
    processed_jobs.add_sink(result_producer)
    
    # Execute the job
    env.execute("TDA Stream Processor")

def process_job_event(event: dict) -> str:
    """Process individual job events."""
    if event['event_type'] == 'job.submitted':
        # Trigger C++ computation
        result = trigger_cpp_computation(event['data'])
        
        # Create result event
        result_event = {
            "schema_version": "1.0",
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "result.persistence_diagram",
            "source": "flink_processor",
            "correlation_id": event['correlation_id'],
            "data": result
        }
        
        return json.dumps(result_event)
    
    return None
```

### C++ TDA Core Integration

#### Event Consumer Service
```cpp
// cpp/kafka_consumer.hpp
#include <librdkafka/rdkafkacpp.h>
#include <nlohmann/json.hpp>
#include "tda/core/computation_engine.hpp"

class TDAKafkaConsumer {
public:
    TDAKafkaConsumer(const std::string& brokers, const std::string& group_id);
    
    void subscribe(const std::vector<std::string>& topics);
    void start_processing();
    void stop_processing();
    
private:
    std::unique_ptr<RdKafka::KafkaConsumer> consumer_;
    std::unique_ptr<TDA::ComputationEngine> engine_;
    std::atomic<bool> running_;
    
    void process_message(RdKafka::Message* message);
    void handle_job_submitted(const nlohmann::json& event);
    void publish_progress_update(const std::string& job_id, double progress);
    void publish_result(const std::string& job_id, const TDA::PersistenceDiagram& result);
};

// Implementation
void TDAKafkaConsumer::handle_job_submitted(const nlohmann::json& event) {
    const auto& data = event["data"];
    std::string job_id = data["job_id"];
    std::string algorithm = data["algorithm"];
    
    // Progress callback
    auto progress_callback = [this, job_id](double progress) {
        publish_progress_update(job_id, progress);
    };
    
    try {
        // Perform computation
        auto result = engine_->compute(algorithm, data["parameters"], progress_callback);
        
        // Publish result
        publish_result(job_id, result);
        
    } catch (const std::exception& e) {
        // Publish error event
        publish_error(job_id, e.what());
    }
}
```

### Monitoring and Metrics

#### Kafka Metrics Collection
```python
# monitoring/kafka_metrics.py
from prometheus_client import Counter, Histogram, Gauge
from confluent_kafka.admin import AdminClient

# Metrics definitions
messages_produced = Counter('kafka_messages_produced_total', 'Total messages produced', ['topic'])
messages_consumed = Counter('kafka_messages_consumed_total', 'Total messages consumed', ['topic', 'group'])
message_processing_time = Histogram('kafka_message_processing_seconds', 'Message processing time', ['topic'])
consumer_lag = Gauge('kafka_consumer_lag', 'Consumer lag', ['topic', 'partition', 'group'])

class KafkaMetricsCollector:
    def __init__(self, admin_client: AdminClient):
        self.admin_client = admin_client
    
    async def collect_metrics(self):
        """Collect Kafka metrics periodically."""
        # Consumer lag metrics
        for group in self.get_consumer_groups():
            lag_info = self.get_consumer_lag(group)
            for topic, partitions in lag_info.items():
                for partition, lag in partitions.items():
                    consumer_lag.labels(
                        topic=topic,
                        partition=partition,
                        group=group
                    ).set(lag)
    
    def get_consumer_lag(self, group_id: str) -> dict:
        """Get consumer lag for all topics and partitions."""
        # Implementation to fetch consumer lag
        pass
```

### Health Checks and Circuit Breakers

#### Kafka Health Check
```python
# health/kafka_health.py
from confluent_kafka import Producer, Consumer

class KafkaHealthChecker:
    def __init__(self):
        self.producer = Producer({'bootstrap.servers': ','.join(settings.kafka_bootstrap_servers)})
        self.consumer = Consumer({
            'bootstrap.servers': ','.join(settings.kafka_bootstrap_servers),
            'group.id': 'health_check',
            'auto.offset.reset': 'latest'
        })
    
    async def check_producer_health(self) -> bool:
        """Check if producer can publish messages."""
        try:
            test_message = json.dumps({"health_check": True})
            self.producer.produce('health_check', test_message)
            self.producer.flush(timeout=5)
            return True
        except Exception:
            return False
    
    async def check_consumer_health(self) -> bool:
        """Check if consumer can read messages."""
        try:
            self.consumer.subscribe(['health_check'])
            msg = self.consumer.poll(timeout=5.0)
            return msg is not None and not msg.error()
        except Exception:
            return False
```

## Configuration Recommendations

### Development Environment

#### Docker Compose Configuration
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
      - zookeeper_logs:/var/lib/zookeeper/log

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "9101:9101"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_JMX_PORT: 9101
      KAFKA_JMX_HOSTNAME: localhost
      # Performance settings for development
      KAFKA_NUM_PARTITIONS: 3
      KAFKA_DEFAULT_REPLICATION_FACTOR: 1
      KAFKA_MIN_INSYNC_REPLICAS: 1
      KAFKA_LOG_RETENTION_HOURS: 168  # 7 days
      KAFKA_LOG_SEGMENT_BYTES: 1073741824  # 1GB
      KAFKA_LOG_RETENTION_CHECK_INTERVAL_MS: 300000  # 5 minutes
    volumes:
      - kafka_data:/var/lib/kafka/data

  schema-registry:
    image: confluentinc/cp-schema-registry:7.4.0
    depends_on:
      - kafka
    ports:
      - "8081:8081"
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:29092
      SCHEMA_REGISTRY_LISTENERS: http://0.0.0.0:8081

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    depends_on:
      - kafka
      - schema-registry
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:29092
      KAFKA_CLUSTERS_0_SCHEMAREGISTRY: http://schema-registry:8081

volumes:
  zookeeper_data:
  zookeeper_logs:
  kafka_data:
```

#### Application Configuration
```yaml
# .env.development
# Kafka Development Settings
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=dev_tda_
KAFKA_CONSUMER_GROUP=dev_tda_backend
KAFKA_AUTO_OFFSET_RESET=earliest
KAFKA_ENABLE_AUTO_COMMIT=false
KAFKA_SESSION_TIMEOUT_MS=30000
KAFKA_HEARTBEAT_INTERVAL_MS=10000

# Topic Configuration
KAFKA_TOPIC_TDA_JOBS=jobs
KAFKA_TOPIC_TDA_RESULTS=results
KAFKA_TOPIC_TDA_EVENTS=events
KAFKA_TOPIC_TDA_UPLOADS=uploads
KAFKA_TOPIC_TDA_ERRORS=errors

# Development overrides
KAFKA_DEBUG=true
LOG_LEVEL=DEBUG
MOCK_KAFKA=false
```

#### Topic Creation Script
```bash
#!/bin/bash
# scripts/create_dev_topics.sh

KAFKA_CONTAINER="tda_backend_kafka_1"
TOPICS_PREFIX="dev_tda_"

# Create topics with development settings
docker exec $KAFKA_CONTAINER kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic ${TOPICS_PREFIX}jobs \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config segment.ms=86400000

docker exec $KAFKA_CONTAINER kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic ${TOPICS_PREFIX}results \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=2592000000 \
  --config cleanup.policy=compact \
  --config segment.ms=86400000

docker exec $KAFKA_CONTAINER kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic ${TOPICS_PREFIX}events \
  --partitions 1 \
  --replication-factor 1 \
  --config retention.ms=1209600000

docker exec $KAFKA_CONTAINER kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic ${TOPICS_PREFIX}uploads \
  --partitions 2 \
  --replication-factor 1 \
  --config retention.ms=259200000

docker exec $KAFKA_CONTAINER kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic ${TOPICS_PREFIX}errors \
  --partitions 1 \
  --replication-factor 1 \
  --config retention.ms=2592000000

echo "Development topics created successfully!"
```

### Production Environment

#### Kubernetes Deployment
```yaml
# k8s/kafka-cluster.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: tda-kafka-cluster
  namespace: tda-platform
spec:
  kafka:
    replicas: 9
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true
      - name: external
        port: 9094
        type: loadbalancer
        tls: true
    config:
      # Replication settings
      offsets.topic.replication.factor: 3
      transaction.state.log.replication.factor: 3
      transaction.state.log.min.isr: 2
      default.replication.factor: 3
      min.insync.replicas: 2
      
      # Performance settings
      num.partitions: 12
      num.network.threads: 8
      num.io.threads: 16
      socket.send.buffer.bytes: 102400
      socket.receive.buffer.bytes: 102400
      socket.request.max.bytes: 104857600
      
      # Log settings
      log.retention.hours: 168  # 7 days default
      log.segment.bytes: 1073741824  # 1GB
      log.retention.check.interval.ms: 300000
      log.cleanup.policy: delete
      
      # Memory and disk
      log.dirs: /var/lib/kafka/data-0,/var/lib/kafka/data-1
      num.replica.fetchers: 4
      replica.fetch.max.bytes: 1048576
      
      # Compression
      compression.type: lz4
      
    storage:
      type: jbod
      volumes:
        - id: 0
          type: persistent-claim
          size: 1Ti
          class: fast-ssd
        - id: 1
          type: persistent-claim
          size: 1Ti
          class: fast-ssd
    
    resources:
      requests:
        memory: 32Gi
        cpu: 8
      limits:
        memory: 64Gi
        cpu: 16
    
    jvmOptions:
      -Xms: 32g
      -Xmx: 32g
      -XX:+UseG1GC: null
      -XX:MaxGCPauseMillis: 20
      -XX:InitiatingHeapOccupancyPercent: 35
      -XX:+ExplicitGCInvokesConcurrent: null
      
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 100Gi
      class: fast-ssd
    resources:
      requests:
        memory: 2Gi
        cpu: 1
      limits:
        memory: 4Gi
        cpu: 2

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopicOperator
metadata:
  name: tda-topic-operator
  namespace: tda-platform
spec:
  watchedNamespace: tda-platform
  reconciliationIntervalSeconds: 60
  zookeeperSessionTimeoutSeconds: 20
  topicMetadataMaxAttempts: 6
```

#### Production Topic Configuration
```yaml
# k8s/kafka-topics.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: tda-jobs
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  partitions: 12
  replicas: 3
  config:
    retention.ms: 604800000  # 7 days
    retention.bytes: 1073741824  # 1GB per partition
    segment.ms: 86400000  # 24 hours
    cleanup.policy: delete
    min.insync.replicas: 2
    unclean.leader.election.enable: false

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: tda-results
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  partitions: 12
  replicas: 3
  config:
    retention.ms: 2592000000  # 30 days
    retention.bytes: 5368709120  # 5GB per partition
    segment.ms: 86400000
    cleanup.policy: compact
    min.insync.replicas: 2
    min.cleanable.dirty.ratio: 0.1
    max.compaction.lag.ms: 3600000  # 1 hour

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: tda-events
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  partitions: 3
  replicas: 3
  config:
    retention.ms: 1209600000  # 14 days
    retention.bytes: 536870912  # 512MB per partition
    segment.ms: 86400000
    cleanup.policy: delete
    min.insync.replicas: 2

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: tda-uploads
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  partitions: 6
  replicas: 3
  config:
    retention.ms: 259200000  # 3 days
    retention.bytes: 2147483648  # 2GB per partition
    segment.ms: 86400000
    cleanup.policy: delete
    min.insync.replicas: 2

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: tda-errors
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  partitions: 3
  replicas: 3
  config:
    retention.ms: 2592000000  # 30 days
    retention.bytes: 1073741824  # 1GB per partition
    segment.ms: 86400000
    cleanup.policy: delete
    min.insync.replicas: 2
```

### Security Configuration

#### SSL/TLS Configuration
```yaml
# Security settings for production
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: tda-kafka-cluster
spec:
  kafka:
    listeners:
      - name: tls
        port: 9093
        type: internal
        tls: true
        authentication:
          type: tls
      - name: external
        port: 9094
        type: loadbalancer
        tls: true
        authentication:
          type: tls
        configuration:
          bootstrap:
            annotations:
              service.beta.kubernetes.io/aws-load-balancer-type: nlb
    
    authorization:
      type: simple
      superUsers:
        - CN=kafka-admin
      
    config:
      # SSL settings
      ssl.endpoint.identification.algorithm: HTTPS
      ssl.client.auth: required
      security.inter.broker.protocol: SSL
      
      # SASL settings (if using SASL)
      sasl.enabled.mechanisms: PLAIN,SCRAM-SHA-256
      sasl.mechanism.inter.broker.protocol: SCRAM-SHA-256
      
      # ACL settings
      allow.everyone.if.no.acl.found: false
      authorizer.class.name: kafka.security.authorizer.AclAuthorizer
```

#### ACL Configuration
```bash
# scripts/setup_acls.sh

# Producer ACLs for backend API
kubectl exec -it tda-kafka-cluster-kafka-0 -- kafka-acls \
  --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:tda-backend \
  --operation Write \
  --topic tda-jobs,tda-uploads,tda-events

# Consumer ACLs for Flink processors  
kubectl exec -it tda-kafka-cluster-kafka-0 -- kafka-acls \
  --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:flink-processor \
  --operation Read \
  --topic tda-jobs,tda-uploads \
  --group flink-tda-processor

# Producer ACLs for result publishing
kubectl exec -it tda-kafka-cluster-kafka-0 -- kafka-acls \
  --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:cpp-engine \
  --operation Write \
  --topic tda-results,tda-errors

# Admin ACLs for monitoring
kubectl exec -it tda-kafka-cluster-kafka-0 -- kafka-acls \
  --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:monitoring \
  --operation Describe \
  --resource-pattern-type prefixed \
  --topic tda-
```

## Monitoring and Observability

### Prometheus Metrics

#### Kafka Exporter Configuration
```yaml
# k8s/kafka-exporter.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka-exporter
  namespace: tda-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kafka-exporter
  template:
    metadata:
      labels:
        app: kafka-exporter
    spec:
      containers:
      - name: kafka-exporter
        image: danielqsj/kafka-exporter:v1.6.0
        args:
          - --kafka.server=tda-kafka-cluster-kafka-bootstrap:9092
          - --web.listen-address=:9308
          - --log.level=info
          - --topic.filter=tda-.*
        ports:
        - containerPort: 9308
          name: metrics
        resources:
          requests:
            memory: 128Mi
            cpu: 100m
          limits:
            memory: 256Mi
            cpu: 200m

---
apiVersion: v1
kind: Service
metadata:
  name: kafka-exporter
  namespace: tda-platform
  labels:
    app: kafka-exporter
spec:
  ports:
  - port: 9308
    targetPort: 9308
    name: metrics
  selector:
    app: kafka-exporter

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: kafka-exporter
  namespace: tda-platform
spec:
  selector:
    matchLabels:
      app: kafka-exporter
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

#### Custom Application Metrics
```python
# monitoring/tda_kafka_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Message metrics
tda_messages_produced_total = Counter(
    'tda_kafka_messages_produced_total',
    'Total messages produced to Kafka',
    ['topic', 'event_type', 'status']
)

tda_messages_consumed_total = Counter(
    'tda_kafka_messages_consumed_total', 
    'Total messages consumed from Kafka',
    ['topic', 'consumer_group', 'status']
)

# Processing metrics
tda_message_processing_duration = Histogram(
    'tda_kafka_message_processing_seconds',
    'Time spent processing Kafka messages',
    ['topic', 'event_type'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
)

# Job metrics
tda_jobs_submitted_total = Counter(
    'tda_jobs_submitted_total',
    'Total TDA jobs submitted',
    ['algorithm', 'priority']
)

tda_jobs_completed_total = Counter(
    'tda_jobs_completed_total',
    'Total TDA jobs completed',
    ['algorithm', 'status']
)

tda_job_duration = Histogram(
    'tda_job_duration_seconds',
    'TDA job processing duration',
    ['algorithm'],
    buckets=[1, 5, 10, 30, 60, 180, 300, 600, 1200, 3600]
)

# System metrics
tda_active_jobs = Gauge(
    'tda_active_jobs',
    'Number of currently active TDA jobs'
)

tda_consumer_lag = Gauge(
    'tda_kafka_consumer_lag',
    'Consumer lag per topic and partition',
    ['topic', 'partition', 'consumer_group']
)

# Error metrics
tda_errors_total = Counter(
    'tda_errors_total',
    'Total errors in TDA processing',
    ['error_type', 'component', 'severity']
)

class TDAMetricsCollector:
    def __init__(self):
        self.start_time = time.time()
    
    def record_message_produced(self, topic: str, event_type: str, success: bool):
        status = 'success' if success else 'error'
        tda_messages_produced_total.labels(
            topic=topic,
            event_type=event_type,
            status=status
        ).inc()
    
    def record_message_consumed(self, topic: str, consumer_group: str, success: bool):
        status = 'success' if success else 'error'
        tda_messages_consumed_total.labels(
            topic=topic,
            consumer_group=consumer_group,
            status=status
        ).inc()
    
    @contextmanager
    def time_message_processing(self, topic: str, event_type: str):
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            tda_message_processing_duration.labels(
                topic=topic,
                event_type=event_type
            ).observe(duration)
    
    def record_job_submitted(self, algorithm: str, priority: str):
        tda_jobs_submitted_total.labels(
            algorithm=algorithm,
            priority=priority
        ).inc()
        tda_active_jobs.inc()
    
    def record_job_completed(self, algorithm: str, success: bool, duration: float):
        status = 'success' if success else 'error'
        tda_jobs_completed_total.labels(
            algorithm=algorithm,
            status=status
        ).inc()
        tda_job_duration.labels(algorithm=algorithm).observe(duration)
        tda_active_jobs.dec()
    
    def record_error(self, error_type: str, component: str, severity: str):
        tda_errors_total.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()

# Global metrics collector
metrics = TDAMetricsCollector()
```

### Grafana Dashboards

#### Kafka Overview Dashboard
```json
{
  "dashboard": {
    "title": "TDA Kafka Overview",
    "panels": [
      {
        "title": "Message Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(kafka_topic_partition_current_offset[5m])",
            "legendFormat": "{{topic}} - {{partition}}"
          }
        ]
      },
      {
        "title": "Consumer Lag",
        "type": "graph", 
        "targets": [
          {
            "expr": "kafka_consumer_lag_sum",
            "legendFormat": "{{consumergroup}} - {{topic}}"
          }
        ]
      },
      {
        "title": "TDA Job Processing",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(tda_jobs_submitted_total[5m])",
            "legendFormat": "Jobs Submitted"
          },
          {
            "expr": "rate(tda_jobs_completed_total[5m])",
            "legendFormat": "Jobs Completed"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(tda_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

#### Structured Logging for Kafka Operations
```python
# logging/kafka_logger.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class TDAKafkaLogger:
    def __init__(self, logger_name: str = "tda.kafka"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Structured JSON formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, level: str, event_type: str, details: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "component": "kafka",
            **details
        }
        
        getattr(self.logger, level)(json.dumps(log_entry))
    
    def log_message_produced(self, topic: str, key: str, event_type: str, success: bool):
        self.log_event("info", "message_produced", {
            "topic": topic,
            "key": key,
            "event_type": event_type,
            "success": success
        })
    
    def log_message_consumed(self, topic: str, partition: int, offset: int, event_type: str):
        self.log_event("info", "message_consumed", {
            "topic": topic,
            "partition": partition,
            "offset": offset,
            "event_type": event_type
        })
    
    def log_error(self, error_type: str, message: str, details: Dict[str, Any] = None):
        self.log_event("error", "kafka_error", {
            "error_type": error_type,
            "message": message,
            "details": details or {}
        })

# Global logger instance
kafka_logger = TDAKafkaLogger()
```

### Alerting Rules

#### Prometheus Alerting Configuration
```yaml
# monitoring/alerts/kafka-alerts.yaml
groups:
- name: tda-kafka-alerts
  rules:
  - alert: KafkaConsumerLagHigh
    expr: kafka_consumer_lag_sum > 1000
    for: 5m
    labels:
      severity: warning
      component: kafka
    annotations:
      summary: "High Kafka consumer lag detected"
      description: "Consumer lag for {{ $labels.consumergroup }} on topic {{ $labels.topic }} is {{ $value }}"

  - alert: TDAJobProcessingStalled
    expr: increase(tda_jobs_submitted_total[10m]) > 0 and increase(tda_jobs_completed_total[10m]) == 0
    for: 5m
    labels:
      severity: critical
      component: tda-processing
    annotations:
      summary: "TDA job processing appears stalled"
      description: "Jobs are being submitted but none completed in the last 10 minutes"

  - alert: KafkaErrorRateHigh
    expr: rate(tda_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
      component: kafka
    annotations:
      summary: "High error rate in TDA Kafka processing"
      description: "Error rate is {{ $value }} errors per second"

  - alert: KafkaBrokerDown
    expr: kafka_brokers < 3
    for: 1m
    labels:
      severity: critical
      component: kafka-infrastructure
    annotations:
      summary: "Kafka broker(s) down"
      description: "Only {{ $value }} Kafka brokers are available (expected: 3+)"

  - alert: TDAJobTimeout
    expr: histogram_quantile(0.95, rate(tda_job_duration_seconds_bucket[5m])) > 600
    for: 5m
    labels:
      severity: warning
      component: tda-processing
    annotations:
      summary: "TDA job processing time high"
      description: "95th percentile job processing time is {{ $value }} seconds"
```

## Security and Compliance

### Authentication and Authorization

#### SASL/SCRAM Authentication
```yaml
# security/kafka-users.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaUser
metadata:
  name: tda-backend-producer
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  authentication:
    type: scram-sha-256
  authorization:
    type: simple
    acls:
    - resource:
        type: topic
        name: tda-jobs
      operation: Write
    - resource:
        type: topic
        name: tda-uploads
      operation: Write
    - resource:
        type: topic
        name: tda-events
      operation: Write

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaUser
metadata:
  name: flink-processor
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  authentication:
    type: scram-sha-256
  authorization:
    type: simple
    acls:
    - resource:
        type: topic
        name: tda-jobs
      operation: Read
    - resource:
        type: topic
        name: tda-uploads
      operation: Read
    - resource:
        type: topic
        name: tda-results
      operation: Write
    - resource:
        type: topic
        name: tda-errors
      operation: Write
    - resource:
        type: group
        name: flink-tda-processor
      operation: Read

---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaUser
metadata:
  name: cpp-engine
  namespace: tda-platform
  labels:
    strimzi.io/cluster: tda-kafka-cluster
spec:
  authentication:
    type: scram-sha-256
  authorization:
    type: simple
    acls:
    - resource:
        type: topic
        name: tda-jobs
      operation: Read
    - resource:
        type: topic
        name: tda-results
      operation: Write
    - resource:
        type: topic
        name: tda-errors
      operation: Write
    - resource:
        type: group
        name: cpp-tda-processors
      operation: Read
```

### Data Encryption

#### Encryption at Rest
```yaml
# security/encryption-config.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: tda-kafka-cluster
spec:
  kafka:
    storage:
      type: jbod
      volumes:
      - id: 0
        type: persistent-claim
        size: 1Ti
        class: encrypted-ssd  # StorageClass with encryption
      - id: 1
        type: persistent-claim
        size: 1Ti
        class: encrypted-ssd
    
    config:
      # Enable encryption for internal replication
      security.inter.broker.protocol: SSL
      ssl.keystore.type: PKCS12
      ssl.truststore.type: PKCS12
      ssl.enabled.protocols: TLSv1.2,TLSv1.3
      ssl.endpoint.identification.algorithm: HTTPS
```

#### Encryption in Transit
```yaml
# All client connections use TLS
listeners:
  - name: tls
    port: 9093
    type: internal
    tls: true
    authentication:
      type: scram-sha-256
  - name: external
    port: 9094
    type: loadbalancer
    tls: true
    authentication:
      type: scram-sha-256
```

### Audit Logging

#### Kafka Audit Configuration
```yaml
# Enable audit logging in Kafka configuration
spec:
  kafka:
    config:
      # Audit settings
      authorizer.class.name: kafka.security.authorizer.AclAuthorizer
      allow.everyone.if.no.acl.found: false
      
      # Log all authorization attempts
      log4j.logger.kafka.authorizer.logger: INFO, authorizerAppender
      log4j.additivity.kafka.authorizer.logger: false
      
      # Audit appender configuration
      log4j.appender.authorizerAppender: org.apache.log4j.DailyRollingFileAppender
      log4j.appender.authorizerAppender.DatePattern: "'.'yyyy-MM-dd-HH"
      log4j.appender.authorizerAppender.File: /var/log/kafka/kafka-authorizer.log
      log4j.appender.authorizerAppender.layout: org.apache.log4j.PatternLayout
      log4j.appender.authorizerAppender.layout.ConversionPattern: "[%d] %p %m (%c)%n"
```

### Compliance Features

#### Data Retention Policies
```python
# compliance/data_retention.py
class DataRetentionManager:
    def __init__(self):
        self.retention_policies = {
            'tda_jobs': timedelta(days=7),      # GDPR: Job data retention
            'tda_results': timedelta(days=30),   # Business: Result retention
            'tda_uploads': timedelta(days=3),    # Security: Temp data cleanup
            'tda_errors': timedelta(days=30),    # Compliance: Error tracking
            'tda_events': timedelta(days=14)     # Operations: Event history
        }
    
    async def enforce_retention_policies(self):
        """Enforce data retention policies across all topics."""
        for topic, retention_period in self.retention_policies.items():
            await self.cleanup_old_data(topic, retention_period)
    
    async def handle_data_deletion_request(self, user_id: str):
        """Handle GDPR data deletion requests."""
        # Delete all user-related events
        await self.delete_user_events(user_id)
        
        # Log deletion for audit
        await self.log_deletion_event(user_id)
```

## Operational Procedures

### Deployment Procedures

#### Rolling Updates
```bash
#!/bin/bash
# scripts/rolling_update_kafka.sh

NAMESPACE="tda-platform"
KAFKA_CLUSTER="tda-kafka-cluster"

echo "Starting rolling update of Kafka cluster..."

# Update Kafka configuration
kubectl apply -f k8s/kafka-cluster.yaml

# Wait for rolling update to complete
kubectl wait --for=condition=Ready kafka/$KAFKA_CLUSTER \
  --namespace=$NAMESPACE \
  --timeout=600s

# Verify cluster health
kubectl exec -it $KAFKA_CLUSTER-kafka-0 --namespace=$NAMESPACE -- \
  kafka-topics --bootstrap-server localhost:9092 --list

echo "Rolling update completed successfully"
```

#### Blue-Green Deployment
```bash
#!/bin/bash
# scripts/blue_green_kafka.sh

# Deploy green cluster
kubectl apply -f k8s/kafka-cluster-green.yaml

# Wait for green cluster to be ready
kubectl wait --for=condition=Ready kafka/tda-kafka-cluster-green \
  --namespace=tda-platform \
  --timeout=900s

# Copy data from blue to green cluster
kafka-mirror-maker.sh \
  --consumer.config blue-consumer.properties \
  --producer.config green-producer.properties \
  --whitelist "tda-.*"

# Switch traffic to green cluster
kubectl patch service kafka-bootstrap \
  --patch '{"spec":{"selector":{"app":"tda-kafka-cluster-green"}}}'

# Monitor for issues, rollback if needed
# kubectl patch service kafka-bootstrap \
#   --patch '{"spec":{"selector":{"app":"tda-kafka-cluster"}}}'
```

### Backup and Recovery

#### Topic Backup Script
```bash
#!/bin/bash
# scripts/backup_kafka_topics.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/kafka/$BACKUP_DATE"
KAFKA_BROKERS="kafka-0:9092,kafka-1:9092,kafka-2:9092"

mkdir -p $BACKUP_DIR

# Export topic configurations
for topic in tda-jobs tda-results tda-events tda-uploads tda-errors; do
    echo "Backing up topic: $topic"
    
    # Export topic configuration
    kafka-configs.sh --bootstrap-server $KAFKA_BROKERS \
      --entity-type topics \
      --entity-name $topic \
      --describe > $BACKUP_DIR/${topic}_config.txt
    
    # Export messages (last 7 days)
    kafka-console-consumer.sh \
      --bootstrap-server $KAFKA_BROKERS \
      --topic $topic \
      --from-beginning \
      --timeout-ms 300000 > $BACKUP_DIR/${topic}_messages.json
done

# Compress backup
tar -czf /backup/kafka_backup_$BACKUP_DATE.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: kafka_backup_$BACKUP_DATE.tar.gz"
```

#### Disaster Recovery Plan
```yaml
# disaster_recovery/kafka_dr_plan.yaml
recovery_procedures:
  data_center_failure:
    priority: critical
    rto: 30_minutes
    rpo: 5_minutes
    steps:
      - activate_standby_cluster
      - redirect_dns_traffic
      - verify_data_consistency
      - resume_processing
  
  broker_failure:
    priority: high
    rto: 10_minutes
    rpo: 0_minutes
    steps:
      - automatic_leader_election
      - redistribute_partitions
      - monitor_replication_lag
  
  topic_corruption:
    priority: medium
    rto: 60_minutes
    rpo: 24_hours
    steps:
      - stop_producers_consumers
      - restore_from_backup
      - verify_data_integrity
      - restart_processing

backup_strategy:
  frequency: daily
  retention: 30_days
  location: s3://tda-kafka-backups/
  verification: automated_restore_test
```

### Capacity Planning

#### Capacity Monitoring
```python
# operations/capacity_planning.py
class KafkaCapacityPlanner:
    def __init__(self):
        self.metrics_client = PrometheusClient()
        self.thresholds = {
            'disk_usage_percent': 80,
            'memory_usage_percent': 85,
            'cpu_usage_percent': 75,
            'network_utilization_percent': 70
        }
    
    async def check_capacity_metrics(self):
        """Check current capacity utilization."""
        metrics = await self.metrics_client.query_range([
            'kafka_log_size_bytes',
            'kafka_broker_memory_usage',
            'kafka_broker_cpu_usage',
            'kafka_network_bytes_per_sec'
        ])
        
        recommendations = []
        
        # Analyze disk usage
        disk_usage = self.calculate_disk_usage(metrics['kafka_log_size_bytes'])
        if disk_usage > self.thresholds['disk_usage_percent']:
            recommendations.append({
                'type': 'scale_storage',
                'urgency': 'high',
                'current_usage': disk_usage,
                'recommended_action': 'Add storage or reduce retention'
            })
        
        # Analyze memory usage
        memory_usage = self.calculate_memory_usage(metrics['kafka_broker_memory_usage'])
        if memory_usage > self.thresholds['memory_usage_percent']:
            recommendations.append({
                'type': 'scale_memory',
                'urgency': 'medium',
                'current_usage': memory_usage,
                'recommended_action': 'Increase broker memory allocation'
            })
        
        return recommendations
    
    async def forecast_capacity_needs(self, days_ahead: int = 30):
        """Forecast capacity needs based on growth trends."""
        historical_data = await self.get_historical_metrics(days=90)
        growth_rate = self.calculate_growth_rate(historical_data)
        
        forecast = {}
        for metric, rate in growth_rate.items():
            current_value = historical_data[metric][-1]
            projected_value = current_value * (1 + rate) ** days_ahead
            forecast[metric] = projected_value
        
        return forecast

# Automated capacity alerting
capacity_planner = KafkaCapacityPlanner()

@scheduled('0 6 * * *')  # Daily at 6 AM
async def daily_capacity_check():
    recommendations = await capacity_planner.check_capacity_metrics()
    if recommendations:
        await send_capacity_alert(recommendations)
```

### Performance Tuning

#### Broker Performance Optimization
```bash
#!/bin/bash
# scripts/optimize_kafka_performance.sh

# JVM tuning for Kafka brokers
export KAFKA_HEAP_OPTS="-Xmx32g -Xms32g"
export KAFKA_JVM_PERFORMANCE_OPTS="-XX:+UseG1GC \
  -XX:MaxGCPauseMillis=20 \
  -XX:InitiatingHeapOccupancyPercent=35 \
  -XX:+ExplicitGCInvokesConcurrent \
  -XX:MaxInlineLevel=15 \
  -Djava.awt.headless=true \
  -XX:+UseStringDeduplication"

# OS-level optimizations
echo 'vm.swappiness=1' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=60' >> /etc/sysctl.conf
echo 'vm.dirty_expire_centisecs=12000' >> /etc/sysctl.conf
echo 'net.core.rmem_default=262144' >> /etc/sysctl.conf
echo 'net.core.rmem_max=16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_default=262144' >> /etc/sysctl.conf
echo 'net.core.wmem_max=16777216' >> /etc/sysctl.conf

sysctl -p

# Disk I/O optimization
echo 'deadline' > /sys/block/nvme0n1/queue/scheduler
echo '4096' > /sys/block/nvme0n1/queue/read_ahead_kb
```

This comprehensive Kafka architecture document provides a solid foundation for implementing the TDA streaming pipeline. The design emphasizes scalability, reliability, and operational excellence while maintaining compatibility with the existing TDA platform components.
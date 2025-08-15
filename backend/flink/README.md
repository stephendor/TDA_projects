# TDA Apache Flink Integration

This directory contains the Apache Flink configuration and streaming jobs for real-time topological data analysis (TDA) processing in the TDA backend platform.

## Overview

The Flink integration provides:

- **Real-time TDA computations** on streaming point cloud data
- **Scalable processing** with configurable parallelism and windowing
- **Kafka integration** for data ingestion and result publishing  
- **Production-ready configuration** with checkpointing and monitoring
- **PyFlink support** for Python-based TDA algorithms

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kafka Topics  │    │  Flink Cluster  │    │   Result Store  │
│                 │    │                 │    │                 │
│ • tda_jobs      │───▶│ • JobManager    │───▶│ • tda_results   │
│ • tda_uploads   │    │ • TaskManagers  │    │ • Persistence   │
│ • tda_events    │    │ • Checkpoints   │    │ • Visualizations│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────▶│   Monitoring    │◀─────────────┘
                        │                 │
                        │ • Prometheus    │
                        │ • Grafana       │
                        │ • Flink Web UI  │
                        └─────────────────┘
```

## Directory Structure

```
flink/
├── config/
│   └── flink-conf.yaml         # Flink cluster configuration
├── jobs/
│   ├── tda-streaming-job.py    # Main TDA streaming job
│   └── sample-config.json      # Job configuration examples
├── scripts/
│   ├── setup-flink.sh          # Cluster setup script
│   └── submit-job.sh           # Job submission script
└── README.md                   # This file
```

## Quick Start

### 1. Setup Flink Cluster

```bash
# Navigate to scripts directory
cd scripts/

# Setup for development environment
./setup-flink.sh --environment dev

# Setup for production environment  
./setup-flink.sh --environment prod --verbose
```

### 2. Submit a TDA Streaming Job

```bash
# Submit with configuration file
./submit-job.sh --job tda-streaming-job.py --config sample-config.json

# Submit with command line parameters
./submit-job.sh --job tda-streaming-job.py \
    --parallelism 4 \
    --window-size 100 \
    --slide-interval 10
```

### 3. Monitor Jobs

```bash
# List all jobs
./submit-job.sh --list

# Check job status
./submit-job.sh --status JOB_ID

# Cancel a job
./submit-job.sh --cancel JOB_ID
```

## Configuration

### Flink Cluster Configuration

The main configuration is in `config/flink-conf.yaml`. Key optimizations for TDA workloads:

```yaml
# Memory configuration optimized for TDA computations
taskmanager.memory.process.size: 4096m
taskmanager.memory.task.heap.size: 2560m
taskmanager.memory.managed.size: 1024m

# State backend with RocksDB for large state
state.backend: rocksdb
state.backend.rocksdb.memory.managed: true

# Checkpointing for fault tolerance
execution.checkpointing.interval: 30s
execution.checkpointing.unaligned: true

# Kafka integration
connector.kafka.bootstrap.servers: kafka1:9092,kafka2:9093,kafka3:9094
connector.kafka.schema-registry.url: http://schema-registry:8081
```

### Job Configuration

Jobs can be configured via JSON files or command line parameters:

```json
{
    "job_name": "tda-streaming-production",
    "parallelism": 8,
    "checkpoint_interval": 30000,
    "window_size": 100,
    "slide_interval": 10,
    "window_timeout": 60,
    "max_dimension": 2,
    "kafka_bootstrap_servers": "kafka1:9092,kafka2:9093,kafka3:9094",
    "input_topic": "tda_jobs",
    "output_topic": "tda_results"
}
```

## TDA Streaming Jobs

### Main Processing Pipeline

The TDA streaming job (`jobs/tda-streaming-job.py`) implements:

1. **Data Ingestion**: Consumes point cloud data from Kafka topics
2. **Point Cloud Parsing**: Extracts and validates point coordinates
3. **Windowing**: Groups points into processing windows (tumbling/sliding)
4. **TDA Computation**: Computes persistence diagrams and homology
5. **Result Publishing**: Publishes results to output topics

### Windowing Strategies

#### Tumbling Windows
```python
TumblingEventTimeWindows.of(Time.seconds(60))  # 60-second windows
```

#### Sliding Windows  
```python
SlidingEventTimeWindows.of(
    Time.seconds(60),    # window size
    Time.seconds(10)     # slide interval
)
```

#### Session Windows
```python
SessionWindows.withGap(Time.minutes(5))  # 5-minute inactivity gap
```

### TDA Algorithms Supported

- **Vietoris-Rips Complex**: Standard persistence computation
- **Alpha Complex**: Geometric persistence analysis
- **Witness Complex**: Sparse landmark-based computation
- **Incremental TDA**: Real-time persistence updates

## Environment-Specific Deployment

### Development Environment

```bash
./setup-flink.sh --environment dev
```

- Single TaskManager with 2 slots
- Local filesystem for checkpoints
- Reduced memory allocation
- 1-minute checkpoint intervals

### Staging Environment

```bash
./setup-flink.sh --environment staging
```

- 2 TaskManagers with 4 slots each
- S3-compatible storage for checkpoints
- Medium memory allocation
- 30-second checkpoint intervals

### Production Environment

```bash
./setup-flink.sh --environment prod
```

- 3+ TaskManagers with 8 slots each
- S3 storage with high availability
- Maximum memory allocation
- 30-second checkpoint intervals
- Monitoring and alerting enabled

## Monitoring and Observability

### Web Interfaces

- **Flink Web UI**: http://localhost:8082 (JobManager)
- **Prometheus**: http://localhost:9091 (Metrics)
- **Grafana**: http://localhost:3001 (Dashboards, admin/admin)

### Key Metrics

#### Job Metrics
- `flink_jobmanager_job_uptime`: Job runtime
- `flink_taskmanager_job_task_numRecordsIn`: Input record rate
- `flink_taskmanager_job_task_numRecordsOut`: Output record rate

#### TDA-Specific Metrics
- `tda_computation_time`: TDA algorithm execution time
- `tda_points_processed`: Number of points processed
- `tda_persistence_pairs`: Number of persistence pairs computed
- `tda_memory_usage`: Memory usage during computation

#### System Metrics
- `flink_taskmanager_Status_JVM_Memory_Heap_Used`: JVM heap usage
- `flink_jobmanager_numRunningJobs`: Number of running jobs
- `flink_taskmanager_numAvailableTaskSlots`: Available task slots

### Alerting Rules

```yaml
# High computation time
- alert: TDAComputationTimeHigh
  expr: tda_computation_time > 60
  labels:
    severity: warning
  annotations:
    summary: "TDA computation taking too long"

# High memory usage
- alert: FlinkMemoryUsageHigh  
  expr: flink_taskmanager_Status_JVM_Memory_Heap_Used / flink_taskmanager_Status_JVM_Memory_Heap_Max > 0.9
  labels:
    severity: critical
  annotations:
    summary: "Flink TaskManager memory usage critical"
```

## Kafka Integration

### Input Topics

| Topic | Description | Schema |
|-------|-------------|--------|
| `tda_jobs` | TDA computation requests | `tda_job_event_v1` |
| `tda_uploads` | File upload events | `tda_upload_event_v1` |
| `tda_stream_input` | Direct streaming input | Custom JSON |

### Output Topics

| Topic | Description | Schema |
|-------|-------------|--------|
| `tda_results` | Computation results | `tda_result_event_v1` |
| `tda_stream_output` | Streaming results | Custom JSON |
| `tda_errors` | Error events | `tda_error_event_v1` |

### Message Formats

#### Input Message (tda_jobs)
```json
{
  "job_id": "uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "algorithm": "vietoris_rips",
  "point_cloud": {
    "points": [[0.1, 0.2], [0.3, 0.4]],
    "dimension": 2
  },
  "parameters": {
    "max_dimension": 2,
    "max_persistence": 1.0
  }
}
```

#### Output Message (tda_results)
```json
{
  "job_id": "uuid", 
  "timestamp": "2024-01-01T12:00:05Z",
  "computation_info": {
    "num_points": 100,
    "computation_time": 2.5,
    "success": true
  },
  "tda_results": {
    "betti_numbers": {"0": [1], "1": [0]},
    "persistence_pairs": [
      {"dimension": 0, "birth": 0.0, "death": "inf"}
    ],
    "homology_groups": [
      {"dimension": 0, "num_features": 1}
    ]
  }
}
```

## Performance Tuning

### Memory Configuration

For TDA workloads with large point clouds:

```yaml
# Increase task heap for large matrices
taskmanager.memory.task.heap.size: 4096m

# Use managed memory for RocksDB state
taskmanager.memory.managed.size: 2048m

# Optimize network buffers for streaming
taskmanager.memory.network.fraction: 0.15
```

### Parallelism Guidelines

| Point Cloud Size | Recommended Parallelism | Memory per Slot |
|-------------------|------------------------|-----------------|
| < 1,000 points   | 2-4                    | 1GB            |
| 1,000-10,000     | 4-8                    | 2GB            |
| 10,000-50,000    | 8-16                   | 4GB            |
| > 50,000         | 16+                    | 8GB+           |

### Checkpointing Optimization

```yaml
# For high-throughput scenarios
execution.checkpointing.interval: 60s
execution.checkpointing.unaligned: true
execution.checkpointing.alignment-timeout: 30s

# For low-latency scenarios  
execution.checkpointing.interval: 10s
execution.checkpointing.unaligned: false
```

## Troubleshooting

### Common Issues

#### Job Submission Failures
```bash
# Check Flink cluster status
curl http://localhost:8082/overview

# Check JobManager logs
docker-compose logs flink-jobmanager

# Verify JAR dependencies
ls -la ../docker/flink-jars/
```

#### Memory Issues
```bash
# Monitor memory usage
curl http://localhost:8082/taskmanagers

# Check garbage collection
docker-compose exec flink-taskmanager-1 jstat -gc 1 5s
```

#### Kafka Connectivity
```bash
# Test Kafka connection
kafka-console-producer.sh --bootstrap-server kafka1:9092 --topic tda_jobs

# Check topic configuration  
kafka-topics.sh --bootstrap-server kafka1:9092 --describe --topic tda_jobs
```

### Performance Debugging

#### Slow TDA Computations
1. Check point cloud size and complexity
2. Verify parallelism configuration
3. Monitor CPU and memory usage
4. Consider algorithm optimization

#### High Checkpoint Times
1. Reduce checkpoint frequency
2. Enable unaligned checkpoints
3. Optimize state size
4. Check storage performance

#### Kafka Lag Issues
1. Increase consumer parallelism
2. Optimize batch processing
3. Check producer configuration
4. Monitor partition distribution

## Development Guide

### Adding New TDA Algorithms

1. **Create Algorithm Class**:
```python
class NewTDAAlgorithm(MapFunction):
    def map(self, point_cloud):
        # Implement algorithm
        return results
```

2. **Register in Job Pipeline**:
```python
processed_stream = input_stream.map(NewTDAAlgorithm())
```

3. **Update Configuration**:
```json
{
  "algorithm": "new_tda_algorithm",
  "algorithm_params": {}
}
```

### Testing Jobs Locally

```bash
# Start local Flink cluster
./setup-flink.sh --environment dev

# Submit test job
./submit-job.sh --job tda-streaming-job.py --dry-run --verbose

# Send test data
python3 ../../test_kafka_integration.py
```

### Creating Custom Configurations

```bash
# Copy sample configuration
cp jobs/sample-config.json jobs/my-config.json

# Edit configuration
vim jobs/my-config.json  

# Submit with custom config
./submit-job.sh --job tda-streaming-job.py --config my-config.json
```

## API Reference

### Setup Script Options

```bash
./setup-flink.sh [OPTIONS]

OPTIONS:
  --environment dev|staging|prod    Target environment
  --skip-kafka                      Skip Kafka topics setup  
  --skip-jars                       Skip JAR downloads
  --verbose                         Enable verbose output
```

### Job Submission Options

```bash
./submit-job.sh [ACTION] [OPTIONS]

ACTIONS:
  --job FILE                        Submit job file
  --list                           List running jobs
  --cancel ID                      Cancel job by ID
  --status [ID]                    Show job status

OPTIONS:
  --config FILE                    Configuration file
  --parallelism N                  Job parallelism
  --window-size N                  Window size
  --slide-interval N               Slide interval
```

## Contributing

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comprehensive docstrings
- Include type hints where possible

### Testing
- Add unit tests for new algorithms
- Test with various point cloud sizes
- Verify memory usage and performance
- Test fault tolerance scenarios

### Documentation
- Update README for new features
- Add inline code comments
- Create examples for new algorithms
- Update monitoring dashboards

## License

This Flink integration is part of the TDA Platform and follows the same license terms as the main project.
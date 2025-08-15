# TDA Kafka Cluster Docker Setup

This directory contains a comprehensive Docker Compose configuration for running a production-ready Kafka cluster for the TDA (Topological Data Analysis) platform.

## 🎯 Overview

The setup provides:
- **3-node ZooKeeper ensemble** for high availability
- **3-broker Kafka cluster** with proper replication
- **Schema Registry** for message schema management
- **Kafka Connect** for external system integration
- **Kafka UI** for monitoring and management
- **KSQLDB** for stream processing (optional)
- **Redis** for caching and session management

## 📁 Directory Structure

```
docker/
├── docker-compose.kafka.yml    # Main Kafka cluster configuration
├── kafka.env                   # Environment variables and settings
├── scripts/                    # Management scripts
│   ├── start-kafka.sh         # Cluster startup script
│   ├── stop-kafka.sh          # Cluster shutdown script
│   └── kafka-topics.sh        # Topic management script
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Start the Cluster

```bash
# Basic startup
./scripts/start-kafka.sh

# Verbose startup with latest images
./scripts/start-kafka.sh --verbose --pull

# Production mode startup
./scripts/start-kafka.sh --environment production
```

### 2. Verify Installation

The startup script automatically performs health checks, but you can verify manually:

```bash
# Check running containers
docker ps --filter "name=tda_"

# List topics
./scripts/kafka-topics.sh list

# Check cluster health
./scripts/start-kafka.sh --health-only
```

### 3. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **Kafka UI** | http://localhost:8080 | Web interface for cluster management |
| **Schema Registry** | http://localhost:8081 | REST API for schema management |
| **Kafka Connect** | http://localhost:8083 | REST API for connector management |
| **KSQLDB Server** | http://localhost:8088 | Stream processing queries |

### 4. Connect to Kafka

| Connection Type | Bootstrap Servers |
|----------------|-------------------|
| **External (from host)** | `localhost:19092,localhost:19093,localhost:19094` |
| **Internal (containers)** | `kafka-1:9092,kafka-2:9092,kafka-3:9092` |

## 📋 TDA Topic Architecture

The cluster automatically creates topics according to the TDA platform architecture:

| Topic | Purpose | Partitions | Retention | Cleanup Policy |
|-------|---------|------------|-----------|----------------|
| `tda_jobs` | Job submission and lifecycle | 3 | 7 days | delete |
| `tda_results` | Computation results | 3 | 30 days | compact |
| `tda_events` | System events and notifications | 1 | 14 days | delete |
| `tda_uploads` | File upload events | 2 | 3 days | delete |
| `tda_errors` | Error tracking | 1 | 30 days | delete |

## 🛠️ Management Scripts

### Start Kafka Cluster

```bash
./scripts/start-kafka.sh [OPTIONS]

Options:
  --verbose              Enable detailed output
  --environment ENV      Set environment (development|staging|production)
  --pull                 Force pull latest images
  --clean               Clean restart (remove containers first)
  --no-topics           Skip topic creation
  --health-only         Only perform health checks
```

### Stop Kafka Cluster

```bash
./scripts/stop-kafka.sh [OPTIONS]

Options:
  --graceful            Graceful shutdown (default)
  --force              Force immediate shutdown
  --remove-containers   Remove containers after stopping
  --remove-volumes      Remove data volumes (⚠️ DATA LOSS)
  --clean-all          Complete cleanup
  --backup             Backup configurations before shutdown
```

### Topic Management

```bash
./scripts/kafka-topics.sh COMMAND [OPTIONS]

Commands:
  list                     List all topics
  describe TOPIC           Show topic details
  create TOPIC [CONFIG]    Create topic
  delete TOPIC             Delete topic
  create-all              Create all TDA topics
  metrics [TOPIC]         Show topic metrics
  produce TOPIC [MSG]     Send test message
  consume TOPIC [COUNT]   Read messages
```

## ⚙️ Configuration

### Environment Variables

Edit `kafka.env` to customize the cluster:

```bash
# Basic Configuration
ENVIRONMENT=development
KAFKA_BROKER_COUNT=3
KAFKA_DEFAULT_PARTITIONS=3

# Performance Tuning
KAFKA_HEAP_SIZE=2G
KAFKA_NUM_NETWORK_THREADS=8
KAFKA_NUM_IO_THREADS=8

# TDA-specific Topics
TDA_JOBS_PARTITIONS=3
TDA_RESULTS_PARTITIONS=3
TDA_EVENTS_PARTITIONS=1
```

### Topic Configuration Examples

```bash
# Create custom topic
./scripts/kafka-topics.sh create my_topic "partitions=5,replication-factor=3"

# Update topic retention
./scripts/kafka-topics.sh update tda_jobs retention.ms=1209600000

# Create topic with compaction
./scripts/kafka-topics.sh create compact_topic "cleanup.policy=compact"
```

## 🔍 Monitoring and Debugging

### Kafka UI Features

- **Cluster Overview**: Broker status, topic metrics
- **Topic Management**: Create, delete, configure topics
- **Message Browser**: View and search messages
- **Consumer Groups**: Monitor lag and offsets
- **Schema Registry**: Manage Avro schemas

### Health Checks

All services include comprehensive health checks:

```bash
# Manual health verification
docker exec tda_kafka_1 kafka-broker-api-versions --bootstrap-server localhost:9092
docker exec tda_zookeeper_1 bash -c 'echo ruok | nc localhost 2181'
curl -f http://localhost:8080  # Kafka UI
curl -f http://localhost:8081/subjects  # Schema Registry
```

### Log Analysis

```bash
# View all logs
docker-compose -f docker-compose.kafka.yml logs -f

# Specific service logs
docker-compose -f docker-compose.kafka.yml logs -f kafka-1
docker-compose -f docker-compose.kafka.yml logs -f zookeeper-1

# Search for errors
docker-compose -f docker-compose.kafka.yml logs | grep -i error
```

## 🔧 Development vs Production

### Development Settings (Default)

```bash
# Reduced resource requirements
KAFKA_HEAP_SIZE=2G
KAFKA_DEFAULT_REPLICATION_FACTOR=3
KAFKA_MIN_INSYNC_REPLICAS=2

# Faster startup
KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=3000
KAFKA_AUTO_CREATE_TOPICS_ENABLE=true
```

### Production Overrides

```bash
# Uncomment in kafka.env for production
KAFKA_HEAP_SIZE=32G
KAFKA_NUM_PARTITIONS=12
KAFKA_AUTO_CREATE_TOPICS_ENABLE=false
KAFKA_DELETE_TOPIC_ENABLE=false

# Enhanced performance
KAFKA_NUM_NETWORK_THREADS=16
KAFKA_NUM_IO_THREADS=16
```

## 🛡️ Security Features

### Authentication & Authorization

For production deployments, uncomment security settings in `kafka.env`:

```bash
# SSL/TLS Configuration
KAFKA_SECURITY_PROTOCOL=SSL
KAFKA_SSL_KEYSTORE_LOCATION=/etc/kafka/secrets/kafka.keystore.jks

# SASL Authentication
KAFKA_SASL_ENABLED_MECHANISMS=SCRAM-SHA-256
KAFKA_SASL_MECHANISM_INTER_BROKER_PROTOCOL=SCRAM-SHA-256
```

### Data Encryption

- **At Rest**: Use encrypted storage classes for persistent volumes
- **In Transit**: TLS encryption for all client connections
- **Inter-Broker**: SSL encryption between brokers

## 💾 Backup and Recovery

### Automated Backups

```bash
# Backup with configuration preservation
./scripts/stop-kafka.sh --backup

# Restore from backup
RECOVERY_MODE=true ./scripts/start-kafka.sh
```

### Manual Backup

```bash
# Export topic configurations
docker exec tda_kafka_1 kafka-configs --bootstrap-server localhost:9092 \
  --entity-type topics --describe > topics_backup.txt

# Export consumer group offsets
docker exec tda_kafka_1 kafka-consumer-groups --bootstrap-server localhost:9092 \
  --list > consumer_groups_backup.txt
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Containers Won't Start

```bash
# Check Docker resources
docker system df
docker system prune  # Clean up if needed

# Check port conflicts
netstat -tuln | grep -E "(2181|9092|8080|8081)"

# Check logs for errors
docker-compose -f docker-compose.kafka.yml logs
```

#### 2. ZooKeeper Connection Issues

```bash
# Verify ZooKeeper health
for i in {1..3}; do
  docker exec tda_zookeeper_$i bash -c 'echo ruok | nc localhost 2181'
done

# Check ZooKeeper logs
docker-compose -f docker-compose.kafka.yml logs zookeeper-1
```

#### 3. Topic Creation Failures

```bash
# Check broker status
./scripts/kafka-topics.sh list

# Verify replication settings
docker exec tda_kafka_1 kafka-topics --describe --bootstrap-server localhost:9092 --topic __consumer_offsets
```

#### 4. Performance Issues

```bash
# Check resource usage
docker stats

# Analyze consumer lag
./scripts/kafka-topics.sh metrics

# Monitor JVM metrics (JMX enabled on ports 9101-9103)
```

### Recovery Procedures

#### Complete Cluster Recovery

```bash
# 1. Stop cluster gracefully
./scripts/stop-kafka.sh --backup

# 2. Clean restart
./scripts/start-kafka.sh --clean --pull

# 3. Recreate topics
./scripts/kafka-topics.sh create-all

# 4. Verify health
./scripts/start-kafka.sh --health-only
```

#### Data Recovery

```bash
# 1. Stop cluster preserving volumes
./scripts/stop-kafka.sh

# 2. Restart with data recovery
RECOVERY_MODE=true ./scripts/start-kafka.sh

# 3. Verify data integrity
./scripts/kafka-topics.sh metrics
```

## 📈 Performance Tuning

### Hardware Recommendations

#### Development
- **CPU**: 4+ cores
- **Memory**: 16GB+ RAM
- **Storage**: 500GB+ SSD
- **Network**: 1Gbps

#### Production
- **CPU**: 16+ cores per broker
- **Memory**: 64GB+ RAM per broker
- **Storage**: 2TB+ NVMe SSD per broker
- **Network**: 10Gbps+

### JVM Tuning

```bash
# Heap sizing (in kafka.env)
KAFKA_HEAP_OPTS="-Xmx32g -Xms32g"

# GC optimization
KAFKA_JVM_PERFORMANCE_OPTS="-XX:+UseG1GC -XX:MaxGCPauseMillis=20"

# OS-level optimization
echo 'vm.swappiness=1' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=80' >> /etc/sysctl.conf
```

## 🔗 Integration

### TDA Backend Integration

```python
# Producer configuration
KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:19092,localhost:19093,localhost:19094',
    'acks': 'all',
    'enable.idempotence': True,
    'compression.type': 'lz4'
}

# Consumer configuration
CONSUMER_CONFIG = {
    'bootstrap.servers': 'localhost:19092,localhost:19093,localhost:19094',
    'group.id': 'tda-backend-processors',
    'auto.offset.reset': 'earliest'
}
```

### Docker Compose Integration

```yaml
# In your main docker-compose.yml
networks:
  default:
    external:
      name: tda-kafka-network

services:
  tda-backend:
    depends_on:
      - kafka-1
      - kafka-2
      - kafka-3
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka-1:9092,kafka-2:9092,kafka-3:9092
```

## 📝 Additional Resources

- **Kafka Documentation**: https://kafka.apache.org/documentation/
- **TDA Architecture**: `../kafka/KAFKA_ARCHITECTURE.md`
- **Schema Registry**: https://docs.confluent.io/platform/current/schema-registry/
- **KSQLDB Guide**: https://ksqldb.io/

## 🤝 Support

For issues and questions:

1. **Check logs**: `docker-compose logs`
2. **Health check**: `./scripts/start-kafka.sh --health-only`
3. **Topic verification**: `./scripts/kafka-topics.sh list`
4. **Cluster metrics**: `./scripts/kafka-topics.sh metrics`

## 🔄 Version Information

- **Kafka Version**: 7.5.0 (Confluent Platform)
- **ZooKeeper Version**: 7.5.0
- **Docker Compose Version**: 3.8
- **Schema Registry Version**: 7.5.0
- **KSQLDB Version**: 0.29.0
# Docker Deployment Guide

Complete Docker-based deployment configuration for the TDA Platform Backend with multi-service orchestration.

## 🐳 Architecture Overview

### **Multi-Service Deployment**

```mermaid
graph TB
    subgraph "Load Balancer Layer"
        LB[Nginx Load Balancer]
    end
    
    subgraph "API Layer"
        API1[FastAPI Instance 1]
        API2[FastAPI Instance 2]
        API3[FastAPI Instance 3]
    end
    
    subgraph "Message Streaming"
        K1[Kafka Broker 1]
        K2[Kafka Broker 2] 
        K3[Kafka Broker 3]
        ZK[ZooKeeper Ensemble]
        SR[Schema Registry]
    end
    
    subgraph "Stream Processing"
        JM[Flink Job Manager]
        TM1[Flink Task Manager 1]
        TM2[Flink Task Manager 2]
        TM3[Flink Task Manager 3]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL)]
        REDIS[(Redis)]
        MINIO[MinIO Object Storage]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        ALERT[AlertManager]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> K1
    API2 --> K2
    API3 --> K3
    
    K1 --> TM1
    K2 --> TM2
    K3 --> TM3
    
    API1 --> PG
    API1 --> REDIS
    API1 --> MINIO
    
    PROM --> API1
    PROM --> K1
    PROM --> JM
    GRAF --> PROM
```

## 📋 Docker Compose Configuration

### **Main Deployment File**

```yaml
# /backend/docker-compose.yml
version: '3.8'

services:
  # API Gateway and Load Balancer
  nginx:
    image: nginx:1.25-alpine
    container_name: tda-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - api-1
      - api-2
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3

  # FastAPI Application Instances
  api-1:
    build:
      context: .
      dockerfile: Dockerfile.api
      args:
        BUILD_ENV: production
    container_name: tda-api-1
    environment:
      - API_INSTANCE_ID=api-1
      - DATABASE_URL=postgresql://tda_user:${POSTGRES_PASSWORD}@postgres:5432/tda_db
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092,kafka-2:9093,kafka-3:9094
      - SCHEMA_REGISTRY_URL=http://schema-registry:8081
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENVIRONMENT=production
    volumes:
      - api_logs:/app/logs
      - uploaded_files:/app/uploads
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      kafka-1:
        condition: service_healthy
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  api-2:
    build:
      context: .
      dockerfile: Dockerfile.api
      args:
        BUILD_ENV: production
    container_name: tda-api-2
    environment:
      - API_INSTANCE_ID=api-2
      - DATABASE_URL=postgresql://tda_user:${POSTGRES_PASSWORD}@postgres:5432/tda_db
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092,kafka-2:9093,kafka-3:9094
      - SCHEMA_REGISTRY_URL=http://schema-registry:8081
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENVIRONMENT=production
    volumes:
      - api_logs:/app/logs
      - uploaded_files:/app/uploads
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      kafka-1:
        condition: service_healthy
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Database Services
  postgres:
    image: postgres:16-alpine
    container_name: tda-postgres
    environment:
      - POSTGRES_DB=tda_db
      - POSTGRES_USER=tda_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
      - postgres_backups:/backups
    ports:
      - "5432:5432"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tda_user -d tda_db"]
      interval: 30s
      timeout: 10s
      retries: 5
    command: >
      postgres
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100

  redis:
    image: redis:7-alpine
    container_name: tda-redis
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
      - ./redis/redis.conf:/etc/redis/redis.conf:ro
    ports:
      - "6379:6379"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Object Storage
  minio:
    image: minio/minio:latest
    container_name: tda-minio
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ROOT_USER:-admin}
      - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Apache Kafka Cluster
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: tda-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
      ZOOKEEPER_SYNC_LIMIT: 2
      ZOOKEEPER_INIT_LIMIT: 5
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
      - zookeeper_logs:/var/lib/zookeeper/log
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "bash", "-c", "echo ruok | nc localhost 2181"]
      interval: 30s
      timeout: 10s
      retries: 3

  kafka-1:
    image: confluentinc/cp-kafka:7.4.0
    container_name: tda-kafka-1
    depends_on:
      zookeeper:
        condition: service_healthy
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-1:9092,PLAINTEXT_HOST://localhost:29092
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_MIN_IN_SYNC_REPLICAS: 2
      KAFKA_NUM_PARTITIONS: 6
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: false
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_RETENTION_BYTES: 1073741824
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_COMPRESSION_TYPE: lz4
    volumes:
      - kafka_1_data:/var/lib/kafka/data
    ports:
      - "29092:29092"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  kafka-2:
    image: confluentinc/cp-kafka:7.4.0
    container_name: tda-kafka-2
    depends_on:
      zookeeper:
        condition: service_healthy
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-2:9093,PLAINTEXT_HOST://localhost:29093
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_MIN_IN_SYNC_REPLICAS: 2
      KAFKA_NUM_PARTITIONS: 6
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: false
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_RETENTION_BYTES: 1073741824
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_COMPRESSION_TYPE: lz4
    volumes:
      - kafka_2_data:/var/lib/kafka/data
    ports:
      - "29093:29093"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9093"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  kafka-3:
    image: confluentinc/cp-kafka:7.4.0
    container_name: tda-kafka-3
    depends_on:
      zookeeper:
        condition: service_healthy
    environment:
      KAFKA_BROKER_ID: 3
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-3:9094,PLAINTEXT_HOST://localhost:29094
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_MIN_IN_SYNC_REPLICAS: 2
      KAFKA_NUM_PARTITIONS: 6
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: false
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_RETENTION_BYTES: 1073741824
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_COMPRESSION_TYPE: lz4
    volumes:
      - kafka_3_data:/var/lib/kafka/data
    ports:
      - "29094:29094"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9094"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Schema Registry
  schema-registry:
    image: confluentinc/cp-schema-registry:7.4.0
    container_name: tda-schema-registry
    depends_on:
      kafka-1:
        condition: service_healthy
      kafka-2:
        condition: service_healthy
      kafka-3:
        condition: service_healthy
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka-1:9092,kafka-2:9093,kafka-3:9094
      SCHEMA_REGISTRY_LISTENERS: http://0.0.0.0:8081
      SCHEMA_REGISTRY_SCHEMA_COMPATIBILITY_LEVEL: backward
    ports:
      - "8081:8081"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/subjects"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Apache Flink Cluster
  flink-jobmanager:
    image: flink:1.18.0-scala_2.12-java11
    container_name: tda-flink-jobmanager
    command: jobmanager
    environment:
      - FLINK_PROPERTIES=jobmanager.rpc.address: flink-jobmanager
      - JOB_MANAGER_RPC_ADDRESS=flink-jobmanager
    volumes:
      - flink_data:/tmp
      - ./flink/conf:/opt/flink/conf
      - ./flink/jobs:/opt/flink/jobs
    ports:
      - "8082:8081"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081"]
      interval: 30s
      timeout: 10s
      retries: 3

  flink-taskmanager-1:
    image: flink:1.18.0-scala_2.12-java11
    container_name: tda-flink-taskmanager-1
    command: taskmanager
    environment:
      - FLINK_PROPERTIES=jobmanager.rpc.address: flink-jobmanager
      - JOB_MANAGER_RPC_ADDRESS=flink-jobmanager
      - TASK_MANAGER_NUMBER_OF_TASK_SLOTS=4
    volumes:
      - flink_data:/tmp
      - ./flink/conf:/opt/flink/conf
    depends_on:
      flink-jobmanager:
        condition: service_healthy
    networks:
      - tda_network
    restart: unless-stopped

  flink-taskmanager-2:
    image: flink:1.18.0-scala_2.12-java11
    container_name: tda-flink-taskmanager-2
    command: taskmanager
    environment:
      - FLINK_PROPERTIES=jobmanager.rpc.address: flink-jobmanager
      - JOB_MANAGER_RPC_ADDRESS=flink-jobmanager
      - TASK_MANAGER_NUMBER_OF_TASK_SLOTS=4
    volumes:
      - flink_data:/tmp
      - ./flink/conf:/opt/flink/conf
    depends_on:
      flink-jobmanager:
        condition: service_healthy
    networks:
      - tda_network
    restart: unless-stopped

  # Monitoring Services
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: tda-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    volumes:
      - ./monitoring/prometheus:/etc/prometheus:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  grafana:
    image: grafana/grafana:10.0.0
    container_name: tda-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - tda_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

# Persistent Volumes
volumes:
  postgres_data:
    driver: local
  postgres_backups:
    driver: local
  redis_data:
    driver: local
  minio_data:
    driver: local
  kafka_1_data:
    driver: local
  kafka_2_data:
    driver: local
  kafka_3_data:
    driver: local
  zookeeper_data:
    driver: local
  zookeeper_logs:
    driver: local
  flink_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  api_logs:
    driver: local
  nginx_logs:
    driver: local
  uploaded_files:
    driver: local

# Networks
networks:
  tda_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 🔧 Application Dockerfiles

### **FastAPI Application**

```dockerfile
# /backend/Dockerfile.api
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libeigen3-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
CMD ["uvicorn", "tda_backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r tda && useradd -r -g tda tda

# Copy application code
COPY tda_backend/ ./tda_backend/
COPY tda_core_py/ ./tda_core_py/
COPY setup.py ./

# Install TDA core bindings
RUN cd tda_core_py && pip install .

# Install application
RUN pip install -e .

# Create directories
RUN mkdir -p /app/logs /app/uploads /app/cache && \
    chown -R tda:tda /app

# Switch to non-root user
USER tda

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "tda_backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### **Flink Job Container**

```dockerfile
# /backend/Dockerfile.flink
FROM flink:1.18.0-scala_2.12-java11

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyFlink and TDA dependencies
COPY requirements-flink.txt /opt/
RUN pip3 install -r /opt/requirements-flink.txt

# Copy TDA core bindings
COPY tda_core_py/ /opt/tda_core_py/
RUN cd /opt/tda_core_py && pip3 install .

# Copy Flink job files
COPY flink/ /opt/flink/jobs/
COPY flink/config/ /opt/flink/conf/

# Set permissions
RUN chown -R flink:flink /opt/flink/

# Switch to flink user
USER flink

# Copy job submission script
COPY scripts/submit-flink-job.sh /opt/
RUN chmod +x /opt/submit-flink-job.sh

CMD ["/opt/submit-flink-job.sh"]
```

## ⚙️ Configuration Files

### **Nginx Load Balancer**

```nginx
# /backend/nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream tda_api {
        least_conn;
        server api-1:8000 max_fails=3 fail_timeout=30s;
        server api-2:8000 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;

    # Logging format
    log_format detailed '$remote_addr - $remote_user [$time_local] '
                       '"$request" $status $body_bytes_sent '
                       '"$http_referer" "$http_user_agent" '
                       'rt=$request_time uct="$upstream_connect_time" '
                       'uht="$upstream_header_time" urt="$upstream_response_time"';

    # Main API proxy
    server {
        listen 80;
        server_name api.tda-platform.local;

        access_log /var/log/nginx/tda_access.log detailed;
        error_log /var/log/nginx/tda_error.log warn;

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://tda_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://tda_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }

        # File upload endpoints with special handling
        location /api/v1/data/upload {
            limit_req zone=upload_limit burst=5 nodelay;
            
            client_max_body_size 100M;
            proxy_request_buffering off;
            
            proxy_pass http://tda_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Extended timeouts for uploads
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # WebSocket support for real-time updates
        location /ws/ {
            proxy_pass http://tda_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }

    # Monitoring endpoints
    server {
        listen 80;
        server_name monitoring.tda-platform.local;

        location /prometheus/ {
            proxy_pass http://prometheus:9090/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /grafana/ {
            proxy_pass http://grafana:3000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /flink/ {
            proxy_pass http://flink-jobmanager:8081/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

### **Environment Configuration**

```env
# /backend/.env.production
# Database Configuration
POSTGRES_PASSWORD=secure_postgres_password_here
DATABASE_URL=postgresql://tda_user:${POSTGRES_PASSWORD}@postgres:5432/tda_db

# Redis Configuration  
REDIS_URL=redis://redis:6379/0

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092,kafka-2:9093,kafka-3:9094
SCHEMA_REGISTRY_URL=http://schema-registry:8081

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=RS256
JWT_EXPIRE_MINUTES=1440

# Object Storage
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=secure_minio_password_here
MINIO_ENDPOINT=minio:9000
MINIO_BUCKET=tda-storage

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_ADMIN_PASSWORD=secure_grafana_password_here

# Feature Flags
ENABLE_CACHING=true
ENABLE_METRICS=true
ENABLE_TRACING=false
ENABLE_AUTHENTICATION=true
MOCK_CPP_ENGINE=false

# Performance Tuning
API_WORKERS=4
API_WORKER_CONNECTIONS=1000
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
REDIS_MAX_CONNECTIONS=100

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_ACCESS_LOGS=true
```

## 🚀 Deployment Scripts

### **Production Deployment Script**

```bash
#!/bin/bash
# /backend/scripts/deploy-production.sh

set -e

echo "🚀 Starting TDA Platform Backend Deployment..."

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"

# Functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error_exit() {
    echo "[ERROR] $1" >&2
    exit 1
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || error_exit "Docker is required but not installed"
    command -v docker-compose >/dev/null 2>&1 || error_exit "Docker Compose is required but not installed"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        error_exit "Environment file $ENV_FILE not found"
    fi
    
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error_exit "Docker Compose file $COMPOSE_FILE not found"
    fi
    
    log "Prerequisites check completed ✅"
}

create_backup() {
    log "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if docker-compose ps postgres | grep -q "Up"; then
        log "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dump -U tda_user tda_db > "$BACKUP_DIR/postgres_backup.sql"
    fi
    
    # Backup Redis data
    if docker-compose ps redis | grep -q "Up"; then
        log "Backing up Redis data..."
        docker-compose exec -T redis redis-cli BGSAVE
        docker cp tda-redis:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"
    fi
    
    # Backup uploaded files
    if [[ -d "./volumes/uploaded_files" ]]; then
        log "Backing up uploaded files..."
        tar -czf "$BACKUP_DIR/uploaded_files.tar.gz" ./volumes/uploaded_files/
    fi
    
    log "Backup completed: $BACKUP_DIR ✅"
}

deploy_services() {
    log "Deploying services..."
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose --env-file "$ENV_FILE" pull
    
    # Build custom images
    log "Building custom images..."
    docker-compose --env-file "$ENV_FILE" build --no-cache
    
    # Deploy with rolling update
    log "Starting deployment..."
    docker-compose --env-file "$ENV_FILE" up -d --remove-orphans
    
    log "Services deployed ✅"
}

initialize_kafka() {
    log "Initializing Kafka topics..."
    
    # Wait for Kafka to be ready
    log "Waiting for Kafka cluster..."
    sleep 30
    
    # Create topics
    docker-compose exec kafka-1 kafka-topics --create \
        --bootstrap-server kafka-1:9092 \
        --topic tda_jobs \
        --partitions 6 \
        --replication-factor 3 \
        --config retention.ms=604800000 \
        --if-not-exists
    
    docker-compose exec kafka-1 kafka-topics --create \
        --bootstrap-server kafka-1:9092 \
        --topic tda_results \
        --partitions 3 \
        --replication-factor 3 \
        --config retention.ms=2592000000 \
        --if-not-exists
    
    docker-compose exec kafka-1 kafka-topics --create \
        --bootstrap-server kafka-1:9092 \
        --topic tda_events \
        --partitions 12 \
        --replication-factor 3 \
        --config retention.ms=86400000 \
        --if-not-exists
    
    docker-compose exec kafka-1 kafka-topics --create \
        --bootstrap-server kafka-1:9092 \
        --topic tda_monitoring \
        --partitions 3 \
        --replication-factor 3 \
        --config retention.ms=604800000 \
        --if-not-exists
    
    log "Kafka topics initialized ✅"
}

run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    log "Waiting for database..."
    while ! docker-compose exec postgres pg_isready -U tda_user -d tda_db; do
        sleep 5
    done
    
    # Run migrations
    docker-compose exec api-1 alembic upgrade head
    
    log "Database migrations completed ✅"
}

submit_flink_jobs() {
    log "Submitting Flink jobs..."
    
    # Wait for Flink to be ready
    log "Waiting for Flink cluster..."
    sleep 60
    
    # Submit TDA streaming job
    docker-compose exec flink-jobmanager flink run \
        --class org.apache.flink.client.python.PythonDriver \
        --python /opt/flink/jobs/tda-streaming-job.py \
        --parallelism 4 \
        --detached
    
    log "Flink jobs submitted ✅"
}

health_check() {
    log "Performing health checks..."
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost/health >/dev/null 2>&1; then
            log "API health check passed ✅"
            break
        fi
        log "Waiting for API to be ready... ($i/30)"
        sleep 10
    done
    
    # Check Kafka health
    if docker-compose exec kafka-1 kafka-broker-api-versions --bootstrap-server kafka-1:9092 >/dev/null 2>&1; then
        log "Kafka health check passed ✅"
    else
        error_exit "Kafka health check failed"
    fi
    
    # Check Flink health
    if curl -f http://localhost:8082 >/dev/null 2>&1; then
        log "Flink health check passed ✅"
    else
        error_exit "Flink health check failed"
    fi
    
    log "All health checks passed ✅"
}

# Main deployment flow
main() {
    log "Starting deployment process..."
    
    check_prerequisites
    create_backup
    deploy_services
    
    log "Waiting for services to start..."
    sleep 30
    
    initialize_kafka
    run_migrations
    submit_flink_jobs
    health_check
    
    log "🎉 Deployment completed successfully!"
    log "API available at: http://localhost"
    log "Grafana dashboard: http://localhost:3000"
    log "Flink dashboard: http://localhost:8082"
    log "Prometheus: http://localhost:9090"
}

# Handle signals
trap 'error_exit "Deployment interrupted"' INT TERM

# Run main function
main "$@"
```

### **Development Environment Script**

```bash
#!/bin/bash
# /backend/scripts/dev-setup.sh

set -e

echo "🔧 Setting up TDA Platform Development Environment..."

# Configuration
ENV_FILE=".env.development"
COMPOSE_FILE="docker-compose.dev.yml"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

setup_environment() {
    log "Setting up development environment..."
    
    # Create development environment file
    cat > "$ENV_FILE" << EOF
# Development Environment Configuration
POSTGRES_PASSWORD=dev_password
DATABASE_URL=postgresql://tda_user:dev_password@postgres:5432/tda_db
REDIS_URL=redis://redis:6379/0
KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092
SCHEMA_REGISTRY_URL=http://schema-registry:8081
JWT_SECRET_KEY=dev_jwt_secret_key
MINIO_ROOT_PASSWORD=dev_minio_password
GRAFANA_ADMIN_PASSWORD=dev_grafana_password

# Feature Flags for Development
ENABLE_CACHING=false
ENABLE_METRICS=true
ENABLE_TRACING=true
ENABLE_AUTHENTICATION=false
MOCK_CPP_ENGINE=true

# Development Performance Settings
API_WORKERS=1
LOG_LEVEL=DEBUG
EOF
    
    log "Environment configuration created ✅"
}

start_core_services() {
    log "Starting core development services..."
    
    # Start essential services for development
    docker-compose --env-file "$ENV_FILE" up -d \
        postgres \
        redis \
        kafka-1 \
        schema-registry \
        prometheus \
        grafana
    
    log "Core services started ✅"
}

setup_test_data() {
    log "Setting up test data..."
    
    # Wait for services
    sleep 20
    
    # Create test topics
    docker-compose exec kafka-1 kafka-topics --create \
        --bootstrap-server kafka-1:9092 \
        --topic tda_events_dev \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists
    
    # Insert test data
    docker-compose exec postgres psql -U tda_user -d tda_db -c "
        INSERT INTO test_datasets (name, points) VALUES 
        ('circle', '[[0,0],[1,0],[0,1],[-1,0],[0,-1]]'),
        ('triangle', '[[0,0],[1,0],[0.5,0.866]]');
    "
    
    log "Test data setup completed ✅"
}

main() {
    setup_environment
    start_core_services
    setup_test_data
    
    log "🎉 Development environment ready!"
    log "Database: postgresql://tda_user:dev_password@localhost:5432/tda_db"
    log "Redis: redis://localhost:6379/0"
    log "Kafka: localhost:29092"
    log "Grafana: http://localhost:3000 (admin/dev_grafana_password)"
    log ""
    log "To start API development server:"
    log "  cd backend && uvicorn tda_backend.main:app --reload"
}

main "$@"
```

## 📊 Monitoring Integration

### **Prometheus Configuration**

```yaml
# /backend/monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alert_rules.yml"

scrape_configs:
  # TDA API instances
  - job_name: 'tda-api'
    static_configs:
      - targets: ['api-1:8000', 'api-2:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Kafka brokers
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka-1:9092', 'kafka-2:9093', 'kafka-3:9094']
    scrape_interval: 30s

  # Flink cluster
  - job_name: 'flink'
    static_configs:
      - targets: ['flink-jobmanager:8081']
    metrics_path: '/metrics'

  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  # System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

---

*Next: [Monitoring Setup](../monitoring/prometheus.md) | [Development Guide](../development/getting-started.md)*
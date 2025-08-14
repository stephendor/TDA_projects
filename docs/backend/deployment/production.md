# Production Deployment Configuration

Comprehensive guide for deploying the TDA Platform Backend in production environments with enterprise-grade reliability, security, and performance.

## 🏭 Production Architecture Overview

### **Multi-Environment Strategy**

```mermaid
graph TB
    subgraph "Development"
        DEV_API[FastAPI Dev]
        DEV_DB[(PostgreSQL Dev)]
        DEV_KAFKA[Kafka Dev]
    end
    
    subgraph "Staging"
        STAGE_LB[Load Balancer]
        STAGE_API1[FastAPI Replica 1]
        STAGE_API2[FastAPI Replica 2]
        STAGE_DB[(PostgreSQL Cluster)]
        STAGE_KAFKA[Kafka Cluster]
        STAGE_REDIS[Redis Cluster]
    end
    
    subgraph "Production"
        PROD_LB[Load Balancer + WAF]
        PROD_API1[FastAPI Replica 1]
        PROD_API2[FastAPI Replica 2]
        PROD_API3[FastAPI Replica 3]
        PROD_DB_PRIMARY[(PostgreSQL Primary)]
        PROD_DB_REPLICA[(PostgreSQL Replica)]
        PROD_KAFKA[Kafka Cluster]
        PROD_REDIS[Redis Cluster]
        PROD_FLINK[Flink Cluster]
    end
    
    DEV_API --> STAGE_LB
    STAGE_API1 --> PROD_LB
    STAGE_API2 --> PROD_LB
```

### **Environment Configuration Matrix**

| Component | Development | Staging | Production |
|-----------|-------------|---------|------------|
| **API Replicas** | 1 | 2 | 3+ |
| **Database** | Single instance | Cluster (3 nodes) | Primary + Replicas |
| **Redis** | Single instance | Cluster (3 nodes) | Cluster (6 nodes) |
| **Kafka** | Single broker | 3 brokers | 5+ brokers |
| **Flink** | Local mode | 2 Task Managers | 4+ Task Managers |
| **Load Balancer** | None | Nginx | Nginx + HAProxy |
| **SSL/TLS** | Self-signed | Let's Encrypt | Enterprise CA |
| **Monitoring** | Basic | Full stack | Full + Alerting |

## 🔐 Environment-Specific Configurations

### **Production Docker Compose**

```yaml
# /deployment/production/docker-compose.prod.yml
version: '3.8'

services:
  # Nginx Load Balancer with SSL Termination
  nginx:
    image: nginx:1.25-alpine
    container_name: tda-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
    depends_on:
      - api-1
      - api-2
      - api-3
    restart: unless-stopped
    networks:
      - tda-network

  # FastAPI Application Instances (Production Scale)
  api-1:
    image: tda-platform/backend:${VERSION:-latest}
    container_name: tda-api-1
    environment:
      - ENVIRONMENT=production
      - API_INSTANCE_ID=api-1
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - JWT_PRIVATE_KEY_PATH=/app/certs/jwt-private.pem
      - JWT_PUBLIC_KEY_PATH=/app/certs/jwt-public.pem
    volumes:
      - ./certs:/app/certs:ro
      - api_logs:/app/logs
      - uploaded_files:/app/uploads
    depends_on:
      postgres-primary:
        condition: service_healthy
      redis-cluster:
        condition: service_healthy
      kafka-1:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - tda-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"

  api-2:
    image: tda-platform/backend:${VERSION:-latest}
    container_name: tda-api-2
    environment:
      - ENVIRONMENT=production
      - API_INSTANCE_ID=api-2
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - JWT_PRIVATE_KEY_PATH=/app/certs/jwt-private.pem
      - JWT_PUBLIC_KEY_PATH=/app/certs/jwt-public.pem
    volumes:
      - ./certs:/app/certs:ro
      - api_logs:/app/logs
      - uploaded_files:/app/uploads
    depends_on:
      postgres-primary:
        condition: service_healthy
      redis-cluster:
        condition: service_healthy
      kafka-1:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - tda-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"

  api-3:
    image: tda-platform/backend:${VERSION:-latest}
    container_name: tda-api-3
    environment:
      - ENVIRONMENT=production
      - API_INSTANCE_ID=api-3
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - JWT_PRIVATE_KEY_PATH=/app/certs/jwt-private.pem
      - JWT_PUBLIC_KEY_PATH=/app/certs/jwt-public.pem
    volumes:
      - ./certs:/app/certs:ro
      - api_logs:/app/logs
      - uploaded_files:/app/uploads
    depends_on:
      postgres-primary:
        condition: service_healthy
      redis-cluster:
        condition: service_healthy
      kafka-1:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - tda-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"

  # PostgreSQL Primary-Replica Cluster
  postgres-primary:
    image: postgres:16-alpine
    container_name: tda-postgres-primary
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-tda_db}
      POSTGRES_USER: ${POSTGRES_USER:-tda_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_REPLICATION_USER: ${POSTGRES_REPLICATION_USER:-replicator}
      POSTGRES_REPLICATION_PASSWORD: ${POSTGRES_REPLICATION_PASSWORD}
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
      - ./postgres/primary/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./postgres/primary/pg_hba.conf:/etc/postgresql/pg_hba.conf
      - ./postgres/init-primary.sql:/docker-entrypoint-initdb.d/init-primary.sql
    ports:
      - "5432:5432"
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-tda_user} -d ${POSTGRES_DB:-tda_db}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - tda-network

  postgres-replica:
    image: postgres:16-alpine
    container_name: tda-postgres-replica
    environment:
      PGUSER: ${POSTGRES_REPLICATION_USER:-replicator}
      POSTGRES_MASTER_SERVICE: postgres-primary
      POSTGRES_REPLICATION_USER: ${POSTGRES_REPLICATION_USER:-replicator}
      POSTGRES_REPLICATION_PASSWORD: ${POSTGRES_REPLICATION_PASSWORD}
    volumes:
      - postgres_replica_data:/var/lib/postgresql/data
      - ./postgres/replica/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./postgres/setup-replica.sh:/docker-entrypoint-initdb.d/setup-replica.sh
    depends_on:
      - postgres-primary
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    restart: unless-stopped
    networks:
      - tda-network

  # Redis Cluster (High Availability)
  redis-cluster:
    image: redis:7-alpine
    container_name: tda-redis-cluster
    command: redis-server /etc/redis/redis.conf
    volumes:
      - ./redis/redis-cluster.conf:/etc/redis/redis.conf
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - tda-network

  # Kafka Cluster (Production Scale)
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: tda-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
      ZOOKEEPER_SYNC_LIMIT: 2
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
      - zookeeper_logs:/var/lib/zookeeper/log
    restart: unless-stopped
    networks:
      - tda-network

  kafka-1:
    image: confluentinc/cp-kafka:7.4.0
    container_name: tda-kafka-1
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-1:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_NUM_PARTITIONS: 6
    volumes:
      - kafka_1_data:/var/lib/kafka/data
    ports:
      - "9092:9092"
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: unless-stopped
    networks:
      - tda-network

  kafka-2:
    image: confluentinc/cp-kafka:7.4.0
    container_name: tda-kafka-2
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-2:29093,PLAINTEXT_HOST://localhost:9093
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_NUM_PARTITIONS: 6
    volumes:
      - kafka_2_data:/var/lib/kafka/data
    ports:
      - "9093:9093"
    restart: unless-stopped
    networks:
      - tda-network

  kafka-3:
    image: confluentinc/cp-kafka:7.4.0
    container_name: tda-kafka-3
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 3
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-3:29094,PLAINTEXT_HOST://localhost:9094
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_NUM_PARTITIONS: 6
    volumes:
      - kafka_3_data:/var/lib/kafka/data
    ports:
      - "9094:9094"
    restart: unless-stopped
    networks:
      - tda-network

  # Schema Registry
  schema-registry:
    image: confluentinc/cp-schema-registry:7.4.0
    container_name: tda-schema-registry
    depends_on:
      - kafka-1
      - kafka-2
      - kafka-3
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka-1:29092,kafka-2:29093,kafka-3:29094
      SCHEMA_REGISTRY_LISTENERS: http://0.0.0.0:8081
    ports:
      - "8081:8081"
    restart: unless-stopped
    networks:
      - tda-network

  # Flink Cluster
  flink-jobmanager:
    image: flink:1.18.0
    container_name: tda-flink-jobmanager
    ports:
      - "8081:8081"
    command: jobmanager
    environment:
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: flink-jobmanager
        taskmanager.numberOfTaskSlots: 4
        parallelism.default: 8
        jobmanager.memory.process.size: 2gb
        taskmanager.memory.process.size: 4gb
    volumes:
      - flink_jobmanager_data:/opt/flink/data
    restart: unless-stopped
    networks:
      - tda-network

  flink-taskmanager-1:
    image: flink:1.18.0
    container_name: tda-flink-taskmanager-1
    depends_on:
      - flink-jobmanager
    command: taskmanager
    environment:
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: flink-jobmanager
        taskmanager.numberOfTaskSlots: 4
        taskmanager.memory.process.size: 4gb
    volumes:
      - flink_taskmanager_1_data:/opt/flink/data
    restart: unless-stopped
    networks:
      - tda-network

  flink-taskmanager-2:
    image: flink:1.18.0
    container_name: tda-flink-taskmanager-2
    depends_on:
      - flink-jobmanager
    command: taskmanager
    environment:
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: flink-jobmanager
        taskmanager.numberOfTaskSlots: 4
        taskmanager.memory.process.size: 4gb
    volumes:
      - flink_taskmanager_2_data:/opt/flink/data
    restart: unless-stopped
    networks:
      - tda-network

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: tda-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert-rules.yml:/etc/prometheus/alert-rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - tda-network

  grafana:
    image: grafana/grafana:10.2.0
    container_name: tda-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    networks:
      - tda-network

volumes:
  postgres_primary_data:
  postgres_replica_data:
  redis_data:
  kafka_1_data:
  kafka_2_data:
  kafka_3_data:
  zookeeper_data:
  zookeeper_logs:
  flink_jobmanager_data:
  flink_taskmanager_1_data:
  flink_taskmanager_2_data:
  prometheus_data:
  grafana_data:
  api_logs:
  uploaded_files:

networks:
  tda-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 🔒 Secret Management

### **Environment Variables Configuration**

```bash
# /deployment/production/.env.prod
# =================================
# Production Environment Variables
# =================================

# Application Configuration
ENVIRONMENT=production
VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://tda_user:${POSTGRES_PASSWORD}@postgres-primary:5432/tda_db
POSTGRES_DB=tda_db
POSTGRES_USER=tda_user
POSTGRES_PASSWORD=${VAULT_POSTGRES_PASSWORD}
POSTGRES_REPLICATION_USER=replicator
POSTGRES_REPLICATION_PASSWORD=${VAULT_POSTGRES_REPLICATION_PASSWORD}

# Redis Configuration
REDIS_URL=redis://tda-redis-cluster:6379/0
REDIS_PASSWORD=${VAULT_REDIS_PASSWORD}

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka-1:29092,kafka-2:29093,kafka-3:29094
SCHEMA_REGISTRY_URL=http://schema-registry:8081

# JWT Configuration
JWT_SECRET_KEY=${VAULT_JWT_SECRET_KEY}
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MAX_CONNECTIONS=100
RATE_LIMIT_REQUESTS_PER_MINUTE=1000

# File Upload Configuration
MAX_UPLOAD_SIZE_MB=100
UPLOAD_STORAGE_PATH=/app/uploads
ALLOWED_FILE_TYPES=csv,json,txt

# TDA Engine Configuration
TDA_MAX_POINTS=100000
TDA_MAX_DIMENSION=10
TDA_DEFAULT_THREADS=4
TDA_MEMORY_LIMIT_MB=2048

# Monitoring Configuration
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=${VAULT_GRAFANA_ADMIN_PASSWORD}

# SSL/TLS Configuration
SSL_CERT_PATH=/app/certs/tda-platform.crt
SSL_KEY_PATH=/app/certs/tda-platform.key
SSL_CA_PATH=/app/certs/ca.crt

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_S3_BUCKET=${VAULT_BACKUP_S3_BUCKET}
BACKUP_S3_ACCESS_KEY=${VAULT_BACKUP_S3_ACCESS_KEY}
BACKUP_S3_SECRET_KEY=${VAULT_BACKUP_S3_SECRET_KEY}

# External Services
EXTERNAL_API_BASE_URL=https://api.external-service.com
EXTERNAL_API_KEY=${VAULT_EXTERNAL_API_KEY}
EXTERNAL_API_TIMEOUT=30

# Feature Flags
FEATURE_CACHING_ENABLED=true
FEATURE_ASYNC_PROCESSING=true
FEATURE_STREAMING_ENABLED=true
FEATURE_ADVANCED_METRICS=true
```

### **HashiCorp Vault Integration**

```bash
#!/bin/bash
# /deployment/scripts/setup-secrets.sh

set -e

VAULT_ADDR=${VAULT_ADDR:-"https://vault.company.com"}
VAULT_TOKEN=${VAULT_TOKEN}

# Authenticate with Vault
vault auth -method=userpass username=$VAULT_USERNAME password=$VAULT_PASSWORD

# Store database secrets
vault kv put secret/tda-platform/postgres \
  password="$(openssl rand -base64 32)" \
  replication_password="$(openssl rand -base64 32)"

# Store Redis secrets
vault kv put secret/tda-platform/redis \
  password="$(openssl rand -base64 32)"

# Generate and store JWT keys
openssl genrsa -out jwt-private.pem 2048
openssl rsa -in jwt-private.pem -pubout -out jwt-public.pem

vault kv put secret/tda-platform/jwt \
  secret_key="$(openssl rand -base64 64)" \
  private_key="$(cat jwt-private.pem)" \
  public_key="$(cat jwt-public.pem)"

# Store external API keys
vault kv put secret/tda-platform/external \
  api_key="${EXTERNAL_API_KEY}" \
  backup_s3_access_key="${BACKUP_S3_ACCESS_KEY}" \
  backup_s3_secret_key="${BACKUP_S3_SECRET_KEY}" \
  backup_s3_bucket="${BACKUP_S3_BUCKET}"

# Store Grafana admin password
vault kv put secret/tda-platform/monitoring \
  grafana_admin_password="$(openssl rand -base64 16)"

echo "Secrets stored successfully in Vault"
```

### **Secret Retrieval Script**

```bash
#!/bin/bash
# /deployment/scripts/load-secrets.sh

set -e

# Load secrets from Vault and export as environment variables
export VAULT_POSTGRES_PASSWORD=$(vault kv get -field=password secret/tda-platform/postgres)
export VAULT_POSTGRES_REPLICATION_PASSWORD=$(vault kv get -field=replication_password secret/tda-platform/postgres)
export VAULT_REDIS_PASSWORD=$(vault kv get -field=password secret/tda-platform/redis)
export VAULT_JWT_SECRET_KEY=$(vault kv get -field=secret_key secret/tda-platform/jwt)
export VAULT_GRAFANA_ADMIN_PASSWORD=$(vault kv get -field=grafana_admin_password secret/tda-platform/monitoring)
export VAULT_EXTERNAL_API_KEY=$(vault kv get -field=api_key secret/tda-platform/external)
export VAULT_BACKUP_S3_ACCESS_KEY=$(vault kv get -field=backup_s3_access_key secret/tda-platform/external)
export VAULT_BACKUP_S3_SECRET_KEY=$(vault kv get -field=backup_s3_secret_key secret/tda-platform/external)
export VAULT_BACKUP_S3_BUCKET=$(vault kv get -field=backup_s3_bucket secret/tda-platform/external)

# Save JWT keys to files
vault kv get -field=private_key secret/tda-platform/jwt > /deployment/certs/jwt-private.pem
vault kv get -field=public_key secret/tda-platform/jwt > /deployment/certs/jwt-public.pem

chmod 600 /deployment/certs/jwt-*.pem

echo "Secrets loaded successfully"
```

## 🌐 SSL/TLS Configuration

### **Nginx SSL Configuration**

```nginx
# /deployment/nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log notice;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # Performance settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss
               application/atom+xml image/svg+xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=10r/m;

    # Upstream backend servers
    upstream tda_backend {
        least_conn;
        server api-1:8000 max_fails=3 fail_timeout=30s;
        server api-2:8000 max_fails=3 fail_timeout=30s;
        server api-3:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name api.tda-platform.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name api.tda-platform.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/tda-platform.crt;
        ssl_certificate_key /etc/nginx/ssl/tda-platform.key;
        ssl_session_timeout 1d;
        ssl_session_cache shared:SSL:50m;
        ssl_session_tickets off;

        # Modern SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
        ssl_prefer_server_ciphers off;

        # OCSP stapling
        ssl_stapling on;
        ssl_stapling_verify on;
        ssl_trusted_certificate /etc/nginx/ssl/ca.crt;

        # API routes
        location /api/v1/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://tda_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 60s;
        }

        # File upload endpoint with higher rate limit
        location /api/v1/data/upload {
            limit_req zone=upload burst=5 nodelay;
            
            proxy_pass http://tda_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 600s;
            proxy_connect_timeout 60s;
            client_max_body_size 100M;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://tda_backend;
            access_log off;
        }

        # Static file serving (if needed)
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### **SSL Certificate Management**

```bash
#!/bin/bash
# /deployment/scripts/setup-ssl.sh

set -e

DOMAIN="api.tda-platform.com"
CERT_DIR="/deployment/nginx/ssl"

# Create certificate directory
mkdir -p $CERT_DIR

# Option 1: Let's Encrypt (for staging/development)
if [ "$ENVIRONMENT" != "production" ]; then
    echo "Setting up Let's Encrypt certificate..."
    certbot certonly --standalone \
        --email admin@company.com \
        --agree-tos \
        --no-eff-email \
        -d $DOMAIN
    
    # Copy certificates to nginx directory
    cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $CERT_DIR/tda-platform.crt
    cp /etc/letsencrypt/live/$DOMAIN/privkey.pem $CERT_DIR/tda-platform.key
    cp /etc/letsencrypt/live/$DOMAIN/chain.pem $CERT_DIR/ca.crt

# Option 2: Enterprise CA (for production)
else
    echo "Setting up enterprise CA certificate..."
    
    # Generate private key
    openssl genrsa -out $CERT_DIR/tda-platform.key 2048
    
    # Generate certificate signing request
    openssl req -new -key $CERT_DIR/tda-platform.key \
        -out $CERT_DIR/tda-platform.csr \
        -subj "/C=US/ST=CA/L=San Francisco/O=Company/CN=$DOMAIN"
    
    # Submit CSR to enterprise CA (manual step)
    echo "Certificate signing request generated: $CERT_DIR/tda-platform.csr"
    echo "Submit this CSR to your enterprise CA and place the signed certificate in $CERT_DIR/tda-platform.crt"
    echo "Also place the CA certificate in $CERT_DIR/ca.crt"
fi

# Set appropriate permissions
chmod 600 $CERT_DIR/tda-platform.key
chmod 644 $CERT_DIR/tda-platform.crt
chmod 644 $CERT_DIR/ca.crt

echo "SSL certificates configured successfully"
```

## 🗄️ Database Configuration

### **PostgreSQL Primary Configuration**

```ini
# /deployment/postgres/primary/postgresql.conf
# PostgreSQL Primary Server Configuration

# Connection settings
listen_addresses = '*'
port = 5432
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 3GB

# Write-ahead logging (WAL) for replication
wal_level = replica
max_wal_senders = 10
max_replication_slots = 10
wal_keep_size = 1GB
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/data/archive/%f'

# Checkpoint settings
checkpoint_completion_target = 0.9
checkpoint_timeout = 15min
max_wal_size = 2GB
min_wal_size = 80MB

# Query tuning
work_mem = 64MB
maintenance_work_mem = 256MB
random_page_cost = 1.1

# Logging
logging_collector = on
log_destination = 'stderr'
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'mod'
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

# Autovacuum
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 20s
```

### **PostgreSQL Replica Setup**

```bash
#!/bin/bash
# /deployment/postgres/setup-replica.sh

set -e

# Wait for primary to be ready
until pg_isready -h postgres-primary -p 5432 -U $POSTGRES_REPLICATION_USER; do
    echo "Waiting for primary database..."
    sleep 2
done

# Create base backup from primary
pg_basebackup -h postgres-primary -D /var/lib/postgresql/data \
    -U $POSTGRES_REPLICATION_USER -W -v -P -R

# Configure replica
cat >> /var/lib/postgresql/data/postgresql.conf << EOF
# Replica-specific settings
hot_standby = on
hot_standby_feedback = on
max_standby_streaming_delay = 30s
wal_receiver_status_interval = 10s
EOF

echo "Replica setup completed"
```

### **Database Backup Strategy**

```bash
#!/bin/bash
# /deployment/scripts/backup-database.sh

set -e

BACKUP_DIR="/backups/postgresql"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="tda_db_backup_$BACKUP_DATE.sql"
S3_BUCKET=${BACKUP_S3_BUCKET}

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump -h postgres-primary -U $POSTGRES_USER -d $POSTGRES_DB \
    --no-password --clean --create --verbose \
    > $BACKUP_DIR/$BACKUP_FILE

# Compress backup
gzip $BACKUP_DIR/$BACKUP_FILE

# Upload to S3 (if configured)
if [ ! -z "$S3_BUCKET" ]; then
    aws s3 cp $BACKUP_DIR/$BACKUP_FILE.gz s3://$S3_BUCKET/database-backups/
    echo "Backup uploaded to S3: s3://$S3_BUCKET/database-backups/$BACKUP_FILE.gz"
fi

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Database backup completed: $BACKUP_FILE.gz"
```

## 🔄 Redis Clustering

### **Redis Cluster Configuration**

```ini
# /deployment/redis/redis-cluster.conf
# Redis Cluster Configuration

# Network
bind 0.0.0.0
port 6379
protected-mode yes
requirepass ${REDIS_PASSWORD}

# Persistence
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# Append-only file
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Memory management
maxmemory 1gb
maxmemory-policy allkeys-lru

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
rename-command CONFIG "CONFIG_b835f0e1f2"

# Logging
loglevel notice
logfile /data/redis.log

# Client settings
timeout 300
tcp-keepalive 300
tcp-backlog 511

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128
```

## 🚀 Production Deployment Scripts

### **Deployment Orchestration**

```bash
#!/bin/bash
# /deployment/scripts/deploy-production.sh

set -e

echo "🚀 Starting TDA Platform Production Deployment"

# Load configuration
source /deployment/production/.env.prod
source /deployment/scripts/load-secrets.sh

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
./scripts/health-checks.sh
./scripts/verify-secrets.sh
./scripts/check-resources.sh

# Backup current state
echo "💾 Creating backup..."
./scripts/backup-database.sh

# Pull latest images
echo "📦 Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Deploy with zero-downtime strategy
echo "🔄 Starting blue-green deployment..."

# Start new instances (green)
docker-compose -f docker-compose.prod.yml up -d \
    --scale api-old=0 \
    --scale api-1=1 \
    --scale api-2=1 \
    --scale api-3=1

# Wait for health checks
echo "⏳ Waiting for services to be healthy..."
./scripts/wait-for-health.sh

# Run database migrations
echo "🗄️ Running database migrations..."
docker-compose -f docker-compose.prod.yml exec api-1 \
    python -m alembic upgrade head

# Switch traffic (nginx reload)
echo "🔀 Switching traffic to new instances..."
docker-compose -f docker-compose.prod.yml exec nginx \
    nginx -s reload

# Verify deployment
echo "✅ Verifying deployment..."
./scripts/post-deployment-tests.sh

# Cleanup old instances
echo "🧹 Cleaning up old instances..."
docker system prune -f

echo "🎉 Production deployment completed successfully!"
```

### **Health Check Script**

```bash
#!/bin/bash
# /deployment/scripts/wait-for-health.sh

set -e

TIMEOUT=300
INTERVAL=10
HEALTH_ENDPOINT="http://localhost:80/health"

echo "⏳ Waiting for services to be healthy (timeout: ${TIMEOUT}s)..."

for i in $(seq 1 $((TIMEOUT / INTERVAL))); do
    echo "  Check $i: Testing health endpoint..."
    
    if curl -f -s $HEALTH_ENDPOINT > /dev/null; then
        echo "✅ Services are healthy!"
        exit 0
    fi
    
    echo "  Services not ready, waiting ${INTERVAL}s..."
    sleep $INTERVAL
done

echo "❌ Health check timeout after ${TIMEOUT}s"
exit 1
```

### **Rollback Script**

```bash
#!/bin/bash
# /deployment/scripts/rollback.sh

set -e

echo "🔄 Starting rollback procedure..."

# Load previous version
PREVIOUS_VERSION=$(cat /deployment/.previous-version 2>/dev/null || echo "")

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "❌ No previous version found for rollback"
    exit 1
fi

echo "📦 Rolling back to version: $PREVIOUS_VERSION"

# Set rollback version
export VERSION=$PREVIOUS_VERSION

# Deploy previous version
docker-compose -f docker-compose.prod.yml up -d

# Wait for health checks
./scripts/wait-for-health.sh

# Verify rollback
./scripts/post-deployment-tests.sh

echo "✅ Rollback completed successfully!"
```

---

*Next: [Monitoring Setup](../monitoring/prometheus.md) | [CI/CD Pipeline](./cicd.md)*
# TDA Kafka Monitoring System

A comprehensive monitoring and health check system for the TDA (Topological Data Analysis) platform's Kafka infrastructure. This system provides deep visibility into cluster health, topic performance, consumer lag, TDA-specific workflows, and business metrics.

## Overview

The TDA Kafka Monitoring System consists of several integrated components:

1. **Prometheus Configuration** - Metrics collection from Kafka brokers, TDA services, and custom exporters
2. **Grafana Dashboard** - Rich visualizations for cluster health, TDA workflows, and performance metrics
3. **Alert Rules** - Comprehensive alerting for critical conditions and performance degradation
4. **Health Check Service** - Deep health monitoring with workflow validation
5. **Custom Metrics Exporter** - TDA-specific business metrics and job processing statistics

## Components

### 1. Prometheus Configuration (`prometheus-config.yml`)

Comprehensive metrics collection setup including:

- **Kafka Broker Metrics**: JMX-based broker health, partition status, replication metrics
- **Topic-Level Metrics**: Message rates, byte rates, error rates for TDA topics
- **Consumer Group Monitoring**: Lag tracking, member status, stability metrics
- **TDA Custom Metrics**: Job processing, algorithm performance, workflow health
- **System Metrics**: Node exporter, cAdvisor for container metrics
- **External Services**: Schema Registry, TDA backend API health

**Key Features:**
- Service discovery for dynamic broker registration
- TDA-specific metric filtering and labeling
- JMX exporter rules for detailed Kafka metrics
- Custom metric collection for business logic

### 2. Grafana Dashboard (`grafana-dashboard.json`)

Rich visualization dashboard with multiple panels:

#### Cluster Health Overview
- Broker status and connectivity
- Active controller tracking
- Under-replicated partition monitoring
- Overall TDA system health score

#### TDA Topic Performance
- Message throughput per TDA topic
- Byte rate monitoring (in/out)
- Topic-specific error rates
- Partition distribution analysis

#### Consumer Lag Monitoring
- Real-time lag tracking per consumer group
- Critical lag alerting (>10,000 messages)
- Consumer group stability metrics
- Topic-specific lag breakdown

#### TDA Job Processing
- Job submission/completion rates by type
- Processing time percentiles by algorithm
- Job failure tracking and analysis
- Queue depth monitoring

#### Persistence Diagram Analytics
- H0, H1, H2 homology group metrics
- Point count distributions
- Persistence value statistics
- Topological feature extraction rates

#### System Resources
- Broker CPU, memory, disk usage
- Network throughput and connectivity
- Container resource utilization
- Capacity planning metrics

#### Workflow Health
- Data ingestion pipeline status
- Computation workflow monitoring
- Result delivery tracking
- Error handling efficiency

### 3. Alert Rules (`alerts.yml`)

Multi-tiered alerting system with escalation procedures:

#### Critical Alerts (Immediate Response)
- **Broker Down**: Kafka broker unavailable
- **No Active Controller**: Controller election issues
- **High Under-Replicated Partitions**: Data integrity risks
- **TDA Backend Down**: Core service unavailability
- **Schema Registry Down**: Schema evolution blocked

#### High Priority Alerts (Urgent Response)
- **High Consumer Lag**: Processing delays (>10,000 messages)
- **TDA Job Processing Failures**: High failure rates
- **High Error Rate**: Topic-level error conditions
- **Disk Space Critical**: Storage capacity issues
- **Result Delivery Failures**: Output pipeline problems

#### Warning Alerts (Attention Required)
- **Moderate Consumer Lag**: Performance degradation
- **High CPU/Memory Usage**: Resource pressure
- **TDA Processing Time High**: Algorithm performance issues
- **Low Throughput**: Capacity underutilization

#### Performance Monitoring
- **Workflow Health Degraded**: TDA pipeline issues
- **Replication Lag High**: Consistency concerns
- **Dead Letter Queue Growth**: Error accumulation

#### Capacity Planning
- **Topic Partition Imbalance**: Load distribution issues
- **Network Throughput High**: Bandwidth saturation
- **Job Queue Depth Growing**: Scaling requirements

### 4. Health Check Service (`scripts/health-check.py`)

Comprehensive health monitoring service with multiple check types:

#### Broker Connectivity Tests
- Individual broker response time monitoring
- Leadership and replica distribution analysis
- Broker resource utilization tracking
- Controller election status verification

#### Topic Health Analysis
- Partition health and replication status
- Under-replicated partition detection
- Offline partition monitoring
- Topic configuration drift detection

#### Consumer Lag Monitoring
- Real-time lag calculation per consumer group
- High water mark comparison
- Consumer group stability analysis
- Partition assignment tracking

#### TDA Workflow Validation
- Data ingestion pipeline health
- Computation workflow status
- Result delivery verification
- Error handling effectiveness

#### External Service Integration
- TDA backend API health checks
- Schema Registry connectivity
- Flink job manager status
- Database connectivity verification

**Features:**
- Real-time dashboard with rich terminal UI
- Prometheus metrics export
- JSON output for API integration
- Continuous monitoring mode
- Configurable health thresholds
- Alert integration capabilities

### 5. Custom Metrics Exporter (`monitoring/metrics-exporter.py`)

TDA-specific business metrics collection and export:

#### Job Processing Metrics
- **Job Lifecycle Tracking**: Submitted, queued, processing, completed, failed
- **Algorithm Performance**: Processing time distributions by algorithm type
- **Resource Utilization**: Memory usage, CPU consumption per job
- **Queue Depth Monitoring**: Current backlog by job type
- **Success Rate Tracking**: Algorithm reliability metrics

#### Persistence Diagram Analytics
- **Point Count Metrics**: H0, H1, H2 homology groups
- **Persistence Statistics**: Max, average, total persistence values
- **Birth/Death Analysis**: Filtration parameter distributions
- **Topological Features**: Feature extraction rates by dimension

#### Algorithm Benchmarking
- **Performance Comparison**: Vietoris-Rips vs Alpha Complex vs Čech Complex
- **Computational Complexity**: Time/space complexity tracking
- **Memory Efficiency**: Output size vs memory usage ratios
- **Success Rate Analysis**: Algorithm reliability by input characteristics

#### Workflow Health Metrics
- **Throughput Monitoring**: Jobs per minute by workflow
- **Error Rate Tracking**: Failure rates by workflow component
- **End-to-End Latency**: Pipeline processing times
- **Data Quality Scores**: Input validation and quality metrics

#### Business Intelligence
- **Feature Extraction Rates**: Topological features per second
- **Data Quality Assessment**: Input data validation scores
- **Computational Resource Efficiency**: Cost per computation metrics
- **System Utilization**: Overall platform efficiency metrics

## Installation and Setup

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Dependencies include:
# - confluent-kafka>=2.3.0
# - prometheus-client>=0.19.0
# - pydantic>=2.5.0
# - rich>=13.7.0
# - click>=8.1.0
# - aiohttp>=3.9.0
# - numpy>=1.24.0
# - pyyaml>=6.0
```

### Configuration

1. **Update Kafka Connection Settings**:
   ```yaml
   # config/health-check.yml and config/metrics-exporter.yml
   kafka:
     bootstrap_servers: "your-kafka-brokers:9092"
     security_protocol: "SASL_SSL"  # if using authentication
   ```

2. **Configure TDA Backend Integration**:
   ```yaml
   tda_backend:
     health_endpoint: "http://your-tda-backend:8000/health"
     metrics_endpoint: "http://your-tda-backend:8000/metrics"
   ```

3. **Set Alert Notification Channels**:
   ```yaml
   # alerts.yml
   alerting:
     contact_points:
       - name: "team-notifications"
         type: "slack"
         settings:
           webhook_url: "${SLACK_WEBHOOK_URL}"
   ```

### Deployment

#### Option 1: Docker Deployment (Recommended)

```bash
# Build Docker images
docker build -t tda-health-checker ./scripts/
docker build -t tda-metrics-exporter ./monitoring/

# Run with Docker Compose
docker-compose -f docker-compose.monitoring.yml up -d
```

#### Option 2: Direct Python Execution

```bash
# Start Health Check Service
python scripts/health-check.py --config config/health-check.yml --continuous --dashboard

# Start Metrics Exporter
python monitoring/metrics-exporter.py --config config/metrics-exporter.yml

# Run one-time health check
python scripts/health-check.py --json-output
```

#### Option 3: Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/monitoring/
kubectl apply -f k8s/health-check/
```

### Prometheus Integration

1. **Add Scrape Configurations**:
   ```yaml
   # Add to your prometheus.yml
   scrape_configs:
     - job_name: 'tda-health-check'
       static_configs:
         - targets: ['tda-health-service:8091']
     
     - job_name: 'tda-metrics-exporter'
       static_configs:
         - targets: ['tda-metrics-exporter:8090']
   ```

2. **Import Alert Rules**:
   ```bash
   # Copy alert rules to Prometheus
   cp monitoring/alerts.yml /etc/prometheus/rules/
   ```

### Grafana Dashboard Import

1. **Import Dashboard**:
   - Navigate to Grafana → Dashboards → Import
   - Upload `grafana-dashboard.json`
   - Configure data source connections

2. **Set Up Variables**:
   - Environment: Select target environment
   - Cluster: Choose Kafka cluster
   - Topics: Filter TDA topics

## Usage Examples

### Health Check Service

```bash
# Interactive dashboard mode
python scripts/health-check.py --dashboard

# JSON output for automation
python scripts/health-check.py --json-output

# Continuous monitoring
python scripts/health-check.py --continuous --interval 60

# Custom configuration
python scripts/health-check.py --config /path/to/config.yml
```

### Metrics Exporter

```bash
# Start metrics exporter
python monitoring/metrics-exporter.py --port 8090

# Custom configuration
python monitoring/metrics-exporter.py --config /path/to/config.yml

# Check metrics endpoint
curl http://localhost:8090/metrics

# View statistics
curl http://localhost:8090/stats | jq
```

### API Endpoints

#### Health Check Service (Port 8091)
- `GET /health/metrics` - Prometheus metrics
- `GET /health` - Health status JSON
- `GET /health/dashboard` - Terminal dashboard

#### Metrics Exporter (Port 8090)
- `GET /metrics` - Prometheus metrics
- `GET /health` - Service health
- `GET /stats` - Processing statistics

## Monitoring Best Practices

### 1. Alert Tuning

**Critical Alerts**: Should fire within 1-2 minutes and page on-call immediately
**High Priority**: Should fire within 5 minutes and notify team leads
**Warnings**: Should fire within 10-15 minutes and notify the team

### 2. Threshold Configuration

Adjust thresholds based on your environment:

```yaml
thresholds:
  consumer_lag_warning: 5000      # Adjust based on normal lag patterns
  consumer_lag_critical: 10000    # Set based on SLA requirements
  error_rate_warning: 0.01        # 1% error rate warning
  error_rate_critical: 0.05       # 5% error rate critical
```

### 3. Dashboard Customization

- **Time Windows**: Adjust based on your monitoring needs (1h, 6h, 24h)
- **Refresh Rates**: Balance between real-time updates and system load
- **Panel Layout**: Organize by priority (critical metrics at top)

### 4. Metric Retention

Configure appropriate retention policies:

```yaml
metrics:
  retention_hours: 24           # Local cache retention
  aggregation_window: 300       # 5-minute aggregation windows
```

### 5. Performance Optimization

- **Scrape Intervals**: Balance between accuracy and overhead
- **Metric Cardinality**: Monitor label combinations to avoid high cardinality
- **Resource Allocation**: Ensure adequate CPU/memory for monitoring services

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

```bash
# Check metrics exporter memory
ps aux | grep metrics-exporter
curl http://localhost:8090/stats | jq '.job_history_size'

# Reduce retention if needed
# Edit config/metrics-exporter.yml
metrics:
  retention_hours: 12  # Reduce from 24
```

#### 2. Missing Metrics

```bash
# Verify Kafka connectivity
python scripts/health-check.py --json-output | jq '.components.broker'

# Check consumer group status
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group tda-metrics-exporter
```

#### 3. Alert Fatigue

```bash
# Review alert frequency
grep -c "FIRING" /var/log/alertmanager.log

# Adjust thresholds in alerts.yml
# Add inhibition rules for related alerts
```

#### 4. Dashboard Performance

- Reduce time ranges for heavy queries
- Optimize Prometheus query efficiency
- Use recording rules for complex calculations

### Debugging Commands

```bash
# Check service status
systemctl status tda-health-checker
systemctl status tda-metrics-exporter

# View logs
tail -f /var/log/tda-kafka-health.log
tail -f /var/log/tda-metrics-exporter.log

# Test Kafka connectivity
kafka-topics.sh --bootstrap-server localhost:9092 --list

# Verify Prometheus scraping
curl http://prometheus:9090/api/v1/targets

# Test alert rules
promtool query instant 'up{job="tda-health-check"}'
```

## Integration with Existing Systems

### TDA Backend Integration

The monitoring system integrates seamlessly with the existing TDA backend:

1. **Health Endpoint Integration**: Monitors `/health` endpoint for overall system status
2. **Metrics Forwarding**: Collects and forwards TDA-specific metrics
3. **Workflow Validation**: Validates end-to-end TDA processing workflows
4. **Error Correlation**: Links Kafka errors with TDA processing failures

### External Monitoring Systems

- **PagerDuty**: Critical alert escalation
- **Slack**: Team notifications and status updates
- **Email**: Digest reports and summary notifications
- **Webhook**: Custom integrations with existing tools

## Security Considerations

### 1. Access Control

```yaml
# Secure endpoints with authentication
security:
  metrics_auth: true
  health_auth: false  # Usually kept open for health checks
  admin_endpoints_auth: true
```

### 2. Network Security

- Use TLS for Kafka connections
- Implement firewall rules for monitoring ports
- Use VPN or private networks for inter-service communication

### 3. Secrets Management

```bash
# Use environment variables for sensitive data
export KAFKA_SASL_PASSWORD="your-password"
export SLACK_WEBHOOK_URL="your-webhook-url"
export PAGERDUTY_INTEGRATION_KEY="your-key"
```

## Performance Metrics

The monitoring system itself is monitored for:

- **Collection Latency**: Time to collect and process metrics
- **Memory Usage**: Monitoring service resource consumption
- **Error Rates**: Failed metric collections or health checks
- **Availability**: Uptime of monitoring services

## Contributing

### Adding New Metrics

1. **Define Metric in Exporter**:
   ```python
   new_metric = Gauge('tda_new_metric', 'Description', ['label1', 'label2'])
   ```

2. **Add Collection Logic**:
   ```python
   async def _collect_new_metric(self):
       # Collection logic here
       new_metric.labels(label1='value1').set(computed_value)
   ```

3. **Update Dashboard**: Add new panel to Grafana dashboard

4. **Add Alerts**: Define alert rules if needed

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Load testing
python tests/load_test.py --duration 300
```

## License

This TDA Kafka Monitoring System is part of the TDA Platform and is subject to the platform's licensing terms.
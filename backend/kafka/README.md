# TDA Kafka Management Suite

A comprehensive collection of advanced topic management and administrative utilities for the TDA platform Kafka deployment.

## Overview

This suite provides production-ready tools for managing Kafka topics, monitoring cluster health, and performing administrative operations in the TDA platform environment.

### Components

1. **Topic Management** (`topic-manager.py`) - Advanced topic lifecycle management
2. **Topic Monitoring** (`topic-monitor.py`) - Real-time monitoring and analytics
3. **Administrative Utilities** (`kafka-admin.py`) - Cluster administration and operations
4. **Configuration Management** - YAML-based topic and ACL configurations

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install confluent-kafka pydantic rich click prometheus_client pyyaml

# Set environment variables
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export SCHEMA_REGISTRY_URL="http://localhost:8081"
```

### Basic Usage

```bash
# Create topics for development environment
python scripts/topic-manager.py create --env development

# Start real-time monitoring with dashboard
python scripts/topic-monitor.py monitor --dashboard

# Check cluster health
python scripts/kafka-admin.py cluster-health --detailed

# List consumer groups
python scripts/kafka-admin.py consumer-groups --detailed
```

## Configuration Files

### `config/topics.yml`

Defines all TDA platform topics with environment-specific configurations:

- **Topic Specifications**: Partitions, replication, retention settings
- **Environment Overrides**: Development, staging, production configurations  
- **Schema Requirements**: Avro schema definitions and compatibility settings
- **Monitoring Configuration**: Alert thresholds and metrics collection
- **Security Settings**: Access control and compliance requirements

Key topics:
- `tda_jobs` - Job lifecycle events
- `tda_results` - Computation results and persistence diagrams
- `tda_events` - System events and notifications
- `tda_uploads` - File upload and ingestion events
- `tda_errors` - Error and failure tracking
- `tda_audit` - Audit logs for compliance
- `tda_dlq` - Dead letter queue for failed messages

### `config/acl-config.yml`

Comprehensive access control configuration:

- **Service Accounts**: Authentication and credential management
- **User Groups**: Role-based access control
- **Topic Permissions**: Fine-grained access control per topic
- **Consumer Group Management**: Access restrictions and monitoring
- **Security Policies**: Encryption, compliance, and audit requirements
- **Network Security**: IP restrictions and SSL/TLS configuration

## Tools Documentation

### Topic Manager (`topic-manager.py`)

Advanced topic management with comprehensive validation and safety features.

#### Key Features

- **Environment-aware Configuration**: Different settings for dev/staging/prod
- **Schema Registry Integration**: Automatic schema registration and validation
- **Batch Operations**: Create, update, or delete multiple topics efficiently
- **Configuration Validation**: Comprehensive checks before deployment
- **Health Monitoring**: Real-time topic health assessment
- **Safety Mechanisms**: Protected topic handling and confirmation requirements

#### Usage Examples

```bash
# Create all topics for production environment
python scripts/topic-manager.py create --env production

# Update specific topic configuration
python scripts/topic-manager.py update --topic tda_jobs --env production

# Validate configuration before deployment
python scripts/topic-manager.py validate --env production

# Perform health check on all topics
python scripts/topic-manager.py health-check

# Show configuration tree
python scripts/topic-manager.py show-config --env production

# Dry run to see what would be created
python scripts/topic-manager.py create --env development --dry-run
```

#### Topic Configuration Structure

```yaml
topics:
  tda_jobs:
    description: "TDA computation job lifecycle events"
    partitions: 12
    config:
      retention_ms: 604800000  # 7 days
      cleanup_policy: "delete"
    schema:
      value_schema: "tda_job_event_v1"
      compatibility: "BACKWARD"
    environment_overrides:
      development:
        partitions: 3
        config:
          retention_ms: 259200000  # 3 days
```

### Topic Monitor (`topic-monitor.py`)

Real-time monitoring and analytics with rich visualizations and alerting.

#### Key Features

- **Real-time Monitoring**: Live dashboard with auto-refresh
- **Performance Metrics**: Message rates, throughput, and lag tracking
- **Consumer Group Analysis**: Detailed lag and member monitoring
- **Health Scoring**: Automated health assessment with recommendations
- **Alerting System**: Configurable thresholds with multiple severity levels
- **Prometheus Integration**: Metrics export for external monitoring systems
- **Historical Data**: Trend analysis and performance history
- **Report Generation**: Comprehensive monitoring reports

#### Usage Examples

```bash
# Start real-time dashboard monitoring
python scripts/topic-monitor.py monitor --dashboard --topics tda_jobs,tda_results

# Generate monitoring report
python scripts/topic-monitor.py report --format json --output tda-kafka-report.json

# Start alerting service
python scripts/topic-monitor.py alerts --check-interval 60

# Monitor specific topics without dashboard
python scripts/topic-monitor.py monitor --topics tda_jobs --interval 30
```

#### Dashboard Features

The monitoring dashboard provides:
- **Cluster Overview**: Broker count, topics, partitions, health metrics
- **Topic Health Matrix**: Real-time health scores with color-coded status
- **Consumer Group Status**: Active groups, lag, and member counts
- **Alert Stream**: Recent alerts with severity indicators
- **Performance Metrics**: Message rates and throughput visualization

#### Prometheus Metrics

Exported metrics include:
- `kafka_topic_messages_total` - Total messages per topic
- `kafka_topic_consumer_lag` - Consumer lag per topic and group
- `kafka_topic_health_score` - Calculated health score (0-100)
- `kafka_cluster_under_replicated_partitions` - Under-replicated partition count
- `kafka_consumer_group_lag_total` - Total lag per consumer group

### Administrative Utilities (`kafka-admin.py`)

Comprehensive cluster administration and operational tools.

#### Key Features

- **Cluster Health Diagnostics**: Deep health analysis with recommendations
- **Consumer Group Management**: Detailed analysis and manipulation
- **Offset Management**: Safe offset reset with multiple targeting options
- **Data Migration**: Topic-to-topic data migration with progress tracking
- **Performance Analysis**: Detailed performance profiling and optimization
- **Security Auditing**: Access control and permission analysis
- **Disaster Recovery**: Backup and restore capabilities
- **Batch Operations**: Efficient bulk administrative operations

#### Usage Examples

```bash
# Comprehensive cluster health check
python scripts/kafka-admin.py cluster-health --detailed

# List all consumer groups with details
python scripts/kafka-admin.py consumer-groups --detailed

# Reset consumer group to earliest offset
python scripts/kafka-admin.py reset-offsets --group tda_processors --topic tda_jobs --to-earliest

# Reset to specific datetime
python scripts/kafka-admin.py reset-offsets --group analytics_group --to-datetime "2024-01-15T10:00:00"

# Migrate data between topics
python scripts/kafka-admin.py migrate-data --source-topic old_results --target-topic tda_results

# Check migration status
python scripts/kafka-admin.py migration-status --task-id migration_1642234567

# Analyze topic performance
python scripts/kafka-admin.py analyze-performance --topic tda_jobs --duration 60

# Export cluster configuration
python scripts/kafka-admin.py export-cluster --format json --output cluster-backup.json
```

#### Offset Reset Safety

The offset reset functionality includes multiple safety mechanisms:
- **Consumer Group State Check**: Ensures group is inactive before reset
- **Dry Run Mode**: Preview changes before execution
- **Multiple Reset Targets**: Earliest, latest, datetime, or specific offset
- **Partition-specific Reset**: Target specific partitions if needed
- **Confirmation Requirements**: Explicit confirmation for dangerous operations

#### Data Migration Features

- **Progress Tracking**: Real-time migration progress with ETA
- **Resumable Operations**: Handle interruptions and resume migration
- **Data Transformation**: Optional message transformation during migration
- **Preservation Options**: Maintain original timestamps and partition assignment
- **Error Handling**: Robust error handling with detailed logging

## Environment Configuration

### Development Environment

```yaml
# Relaxed settings for development
defaults:
  replication_factor: 1
  min_in_sync_replicas: 1
  retention_ms: 604800000  # 7 days

# Security relaxed for development
security:
  mfa_required_for_admin: false
  ip_allowlists:
    relaxed_access: ["0.0.0.0/0"]
```

### Production Environment

```yaml
# High availability settings
defaults:
  replication_factor: 3
  min_in_sync_replicas: 2
  retention_ms: 2592000000  # 30 days

# Strict security for production
security:
  mfa_required_for_admin: true
  encryption_required: true
  audit_logging: true
  strict_mode: true
```

## Security and Access Control

### Service Account Management

The ACL configuration defines service accounts for different platform components:

```yaml
service_accounts:
  tda_backend_api:
    authentication: "scram-sha-256"
    description: "TDA Backend API service"
    password_policy: "strong"
    rotation_days: 90
```

### Topic-Level Permissions

Fine-grained access control per topic:

```yaml
topic_access:
  tda_jobs:
    producers:
      - principal: "User:tda_backend_api"
        operations: ["WRITE", "DESCRIBE"]
    consumers:
      - principal: "User:flink_processor"
        operations: ["READ", "DESCRIBE"]
        consumer_groups: ["flink_tda_processor"]
```

### Compliance Features

- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Retention**: Configurable retention policies per topic type
- **Encryption**: At-rest and in-transit encryption requirements
- **Access Monitoring**: Real-time access pattern analysis
- **GDPR Compliance**: Right to be forgotten and data protection

## Monitoring and Alerting

### Health Scoring System

Each topic receives a health score (0-100) based on:
- **Partition Health**: Under-replicated and offline partitions
- **Consumer Lag**: Critical and warning thresholds
- **Replication Factor**: Fault tolerance assessment
- **Error Rates**: Message processing failure rates

### Alert Conditions

Configurable alert thresholds:
- **Critical Consumer Lag**: > 10,000 messages
- **High Error Rate**: > 5% of messages
- **Disk Usage**: > 85% capacity
- **Under-replicated Partitions**: > 10% of total

### Integration Options

- **Prometheus**: Metrics export for Grafana dashboards
- **Webhook Notifications**: Custom alert delivery
- **Email Alerts**: SMTP-based notification system
- **Slack Integration**: Real-time team notifications

## Performance Optimization

### Recommended Settings

#### High Throughput Topics
```yaml
config:
  batch_size: 65536
  linger_ms: 10
  compression_type: "lz4"
  acks: "all"
```

#### Low Latency Topics
```yaml
config:
  batch_size: 16384
  linger_ms: 0
  compression_type: "snappy"
  acks: "1"
```

### Partition Strategy

- **Job Events**: 12 partitions (keyed by job_id for ordering)
- **Results**: 12 partitions (keyed by job_id for results correlation)
- **System Events**: 3 partitions (lower volume, broader distribution)
- **Uploads**: 6 partitions (balanced for upload throughput)

### Consumer Group Design

- **Processing Groups**: Dedicated groups per service type
- **Monitoring Groups**: Separate groups for metrics collection
- **Analytics Groups**: Specialized groups for data science workflows
- **Backup Groups**: Dedicated groups for data archival

## Troubleshooting

### Common Issues

#### Under-replicated Partitions
```bash
# Check cluster health
python scripts/kafka-admin.py cluster-health --detailed

# Verify broker connectivity
python scripts/topic-monitor.py monitor --dashboard
```

#### Consumer Lag Issues
```bash
# Analyze consumer groups
python scripts/kafka-admin.py consumer-groups --group-pattern "tda_*" --detailed

# Check topic performance
python scripts/kafka-admin.py analyze-performance --topic tda_jobs --duration 30
```

#### Topic Configuration Drift
```bash
# Validate current configuration
python scripts/topic-manager.py validate --env production

# Update to correct configuration
python scripts/topic-manager.py update --env production
```

### Performance Debugging

#### Slow Consumer Processing
1. Check consumer lag trends
2. Analyze message rates and batch sizes
3. Verify partition distribution
4. Review consumer group stability

#### High Message Loss
1. Examine dead letter queue activity
2. Check error topic for failure patterns
3. Verify producer acknowledgment settings
4. Review network connectivity and timeouts

### Emergency Procedures

#### Critical System Failure
1. **Immediate Assessment**:
   ```bash
   python scripts/kafka-admin.py cluster-health --detailed
   ```

2. **Service Impact Analysis**:
   ```bash
   python scripts/kafka-admin.py consumer-groups --detailed
   ```

3. **Data Integrity Check**:
   ```bash
   python scripts/topic-monitor.py report --format json
   ```

#### Data Recovery
1. **Backup Verification**:
   ```bash
   python scripts/kafka-admin.py export-cluster --format json --output emergency-backup.json
   ```

2. **Offset Recovery**:
   ```bash
   python scripts/kafka-admin.py reset-offsets --group emergency_recovery --to-datetime "2024-01-15T12:00:00"
   ```

## Development and Testing

### Local Development Setup

1. **Start Kafka Cluster**:
   ```bash
   cd ../docker
   docker-compose -f docker-compose.kafka.yml up -d
   ```

2. **Create Development Topics**:
   ```bash
   python scripts/topic-manager.py create --env development
   ```

3. **Start Monitoring**:
   ```bash
   python scripts/topic-monitor.py monitor --dashboard
   ```

### Testing Procedures

#### Configuration Testing
```bash
# Validate all configurations
python scripts/topic-manager.py validate --env development
python scripts/topic-manager.py validate --env staging
python scripts/topic-manager.py validate --env production

# Test topic creation in dry-run mode
python scripts/topic-manager.py create --env production --dry-run
```

#### Performance Testing
```bash
# Analyze performance under load
python scripts/kafka-admin.py analyze-performance --topic tda_jobs --duration 120

# Monitor during load testing
python scripts/topic-monitor.py monitor --topics tda_jobs,tda_results --interval 10
```

## Best Practices

### Topic Design
- **Naming Convention**: Use `tda_` prefix for all platform topics
- **Partition Count**: Start with 3x expected peak throughput
- **Replication Factor**: Minimum 3 for production topics
- **Retention Policy**: Based on business requirements and compliance

### Security
- **Principle of Least Privilege**: Grant minimum necessary permissions
- **Regular Rotation**: Rotate credentials according to policy
- **Audit Trail**: Enable comprehensive audit logging
- **Network Isolation**: Use VPCs and security groups

### Monitoring
- **Proactive Alerting**: Set thresholds before issues impact users
- **Regular Health Checks**: Automated daily health assessments
- **Performance Baselines**: Establish and monitor performance trends
- **Capacity Planning**: Monitor growth trends for scaling decisions

### Operations
- **Change Management**: Use configuration management for all changes
- **Backup Strategy**: Regular cluster configuration backups
- **Disaster Recovery**: Tested recovery procedures and runbooks
- **Documentation**: Keep operational procedures current

## Support and Maintenance

### Regular Maintenance Tasks

#### Daily
- Health check review
- Alert triage and resolution
- Consumer lag monitoring
- Performance trend analysis

#### Weekly
- Configuration drift detection
- Consumer group optimization
- Performance baseline updates
- Security audit review

#### Monthly
- Capacity planning review
- Configuration optimization
- Disaster recovery testing
- Documentation updates

### Getting Help

For issues or questions:

1. **Check Logs**: Review application and tool logs for error details
2. **Run Diagnostics**: Use built-in health check and validation tools
3. **Monitor Dashboard**: Real-time cluster status and metrics
4. **Configuration Review**: Validate current vs. desired configuration

### Contributing

When contributing improvements:

1. **Test Thoroughly**: Validate in development environment first
2. **Document Changes**: Update this README and inline documentation
3. **Configuration Validation**: Ensure all configurations pass validation
4. **Security Review**: Consider security implications of changes

## Version History

- **v1.0.0**: Initial release with core topic management and monitoring
- **v1.1.0**: Added administrative utilities and data migration
- **v1.2.0**: Enhanced security and ACL management
- **v1.3.0**: Performance analysis and optimization tools

## License

This TDA Kafka Management Suite is part of the TDA Platform and is subject to the platform's licensing terms.
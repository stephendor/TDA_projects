#!/bin/bash
set -e

# =============================================================================
# TDA Flink Cluster Setup Script
# =============================================================================
# 
# This script initializes the Apache Flink cluster for TDA processing,
# sets up Kafka topics, downloads required JARs, and configures monitoring.
#
# Usage:
#   ./setup-flink.sh [--environment dev|staging|prod] [--skip-kafka] [--skip-jars]
#
# Environment Variables:
#   TDA_ENVIRONMENT: Target environment (dev, staging, prod)
#   KAFKA_BOOTSTRAP_SERVERS: Kafka cluster endpoints
#   SCHEMA_REGISTRY_URL: Schema registry URL
#   FLINK_HOME: Flink installation directory
#
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FLINK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
ENVIRONMENT="${TDA_ENVIRONMENT:-dev}"
SKIP_KAFKA=false
SKIP_JARS=false
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-kafka)
            SKIP_KAFKA=true
            shift
            ;;
        --skip-jars)
            SKIP_JARS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--environment dev|staging|prod] [--skip-kafka] [--skip-jars] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --environment   Target environment (default: dev)"
            echo "  --skip-kafka    Skip Kafka topics setup"
            echo "  --skip-jars     Skip JAR downloads"
            echo "  --verbose       Enable verbose output"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Environment-specific configuration
case $ENVIRONMENT in
    dev)
        KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
        SCHEMA_REGISTRY_URL="${SCHEMA_REGISTRY_URL:-http://localhost:8081}"
        FLINK_REST_URL="${FLINK_REST_URL:-http://localhost:8082}"
        PARALLELISM=2
        MEMORY_CONFIG="small"
        ;;
    staging)
        KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-kafka1:9092,kafka2:9093}"
        SCHEMA_REGISTRY_URL="${SCHEMA_REGISTRY_URL:-http://schema-registry:8081}"
        FLINK_REST_URL="${FLINK_REST_URL:-http://flink-jobmanager:8081}"
        PARALLELISM=4
        MEMORY_CONFIG="medium"
        ;;
    prod)
        KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-kafka1:9092,kafka2:9093,kafka3:9094}"
        SCHEMA_REGISTRY_URL="${SCHEMA_REGISTRY_URL:-http://schema-registry:8081}"
        FLINK_REST_URL="${FLINK_REST_URL:-http://flink-jobmanager:8081}"
        PARALLELISM=8
        MEMORY_CONFIG="large"
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT. Use dev, staging, or prod."
        exit 1
        ;;
esac

log_info "Setting up TDA Flink cluster for environment: $ENVIRONMENT"

# =============================================================================
# Functions
# =============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check if running in Docker
    if [ -f /.dockerenv ]; then
        log_info "Running inside Docker container"
        FLINK_HOME="${FLINK_HOME:-/opt/flink}"
    else
        log_info "Running on host system"
        FLINK_HOME="${FLINK_HOME:-$PROJECT_ROOT/flink}"
    fi
    
    # Create required directories
    mkdir -p "$PROJECT_ROOT/backend/docker/flink-jars"
    mkdir -p "$PROJECT_ROOT/backend/docker/flink-checkpoints"
    mkdir -p "$PROJECT_ROOT/backend/docker/flink-savepoints"
    mkdir -p "$PROJECT_ROOT/backend/docker/flink-data"
    
    log_success "Prerequisites check completed"
}

download_required_jars() {
    if [ "$SKIP_JARS" = true ]; then
        log_info "Skipping JAR downloads"
        return 0
    fi
    
    log_info "Downloading required JAR files..."
    
    local jar_dir="$PROJECT_ROOT/backend/docker/flink-jars"
    local flink_version="1.18.0"
    local kafka_version="3.0.1"
    local hadoop_version="3.3.4"
    
    # Download URLs
    local base_url="https://repo1.maven.org/maven2"
    
    # Kafka connector
    local kafka_jar="flink-sql-connector-kafka-${flink_version}.jar"
    if [ ! -f "$jar_dir/$kafka_jar" ]; then
        log_info "Downloading Kafka connector..."
        curl -L "$base_url/org/apache/flink/flink-sql-connector-kafka/${flink_version}/$kafka_jar" \
            -o "$jar_dir/$kafka_jar"
    fi
    
    # Avro format support
    local avro_jar="flink-avro-${flink_version}.jar"
    if [ ! -f "$jar_dir/$avro_jar" ]; then
        log_info "Downloading Avro format support..."
        curl -L "$base_url/org/apache/flink/flink-avro/${flink_version}/$avro_jar" \
            -o "$jar_dir/$avro_jar"
    fi
    
    # JSON format support
    local json_jar="flink-json-${flink_version}.jar"
    if [ ! -f "$jar_dir/$json_jar" ]; then
        log_info "Downloading JSON format support..."
        curl -L "$base_url/org/apache/flink/flink-json/${flink_version}/$json_jar" \
            -o "$jar_dir/$json_jar"
    fi
    
    # Hadoop filesystem support
    local hadoop_jar="flink-shaded-hadoop-3-uber-${hadoop_version}-${flink_version}.jar"
    if [ ! -f "$jar_dir/$hadoop_jar" ]; then
        log_info "Downloading Hadoop filesystem support..."
        curl -L "$base_url/org/apache/flink/flink-shaded-hadoop-3-uber/${hadoop_version}-${flink_version}/$hadoop_jar" \
            -o "$jar_dir/$hadoop_jar"
    fi
    
    # Metrics reporters
    local prometheus_jar="flink-metrics-prometheus-${flink_version}.jar"
    if [ ! -f "$jar_dir/$prometheus_jar" ]; then
        log_info "Downloading Prometheus metrics reporter..."
        curl -L "$base_url/org/apache/flink/flink-metrics-prometheus/${flink_version}/$prometheus_jar" \
            -o "$jar_dir/$prometheus_jar"
    fi
    
    # PyFlink dependencies (if needed)
    local pyflink_jar="flink-python-${flink_version}.jar"
    if [ ! -f "$jar_dir/$pyflink_jar" ]; then
        log_info "Downloading PyFlink support..."
        curl -L "$base_url/org/apache/flink/flink-python/${flink_version}/$pyflink_jar" \
            -o "$jar_dir/$pyflink_jar"
    fi
    
    log_success "JAR files downloaded successfully"
}

setup_kafka_topics() {
    if [ "$SKIP_KAFKA" = true ]; then
        log_info "Skipping Kafka topics setup"
        return 0
    fi
    
    log_info "Setting up Kafka topics for TDA processing..."
    
    # Wait for Kafka to be ready
    log_info "Waiting for Kafka to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$KAFKA_BOOTSTRAP_SERVERS" &> /dev/null; then
            break
        fi
        log_info "Waiting for Kafka... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Kafka is not available after $max_attempts attempts"
        return 1
    fi
    
    # Use the existing topic management script
    local topic_script="$PROJECT_ROOT/backend/kafka/scripts/topic-manager.py"
    if [ -f "$topic_script" ]; then
        log_info "Using existing topic manager script..."
        cd "$PROJECT_ROOT/backend/kafka"
        python3 "$topic_script" --environment "$ENVIRONMENT" --create-all
    else
        log_warning "Topic manager script not found, using basic topic creation..."
        
        # Basic topic creation using docker-compose exec
        local topics=(
            "tda_jobs:12:3"
            "tda_results:12:3" 
            "tda_events:3:3"
            "tda_uploads:6:3"
            "tda_errors:3:3"
            "tda_dlq:3:3"
            "tda_audit:6:3"
            "tda_stream_input:12:3"
            "tda_stream_output:12:3"
        )
        
        for topic_config in "${topics[@]}"; do
            IFS=':' read -r topic partitions replication <<< "$topic_config"
            
            # Adjust for environment
            case $ENVIRONMENT in
                dev)
                    partitions=$((partitions / 2))
                    replication=1
                    ;;
                staging)
                    replication=2
                    ;;
            esac
            
            log_info "Creating topic: $topic (partitions: $partitions, replication: $replication)"
            
            docker-compose -f "$PROJECT_ROOT/backend/docker/docker-compose.kafka.yml" exec -T kafka1 \
                kafka-topics.sh --bootstrap-server localhost:9092 \
                --create --if-not-exists \
                --topic "$topic" \
                --partitions "$partitions" \
                --replication-factor "$replication" \
                --config cleanup.policy=delete \
                --config retention.ms=604800000 || true
        done
    fi
    
    log_success "Kafka topics setup completed"
}

start_flink_cluster() {
    log_info "Starting Flink cluster..."
    
    cd "$PROJECT_ROOT/backend/docker"
    
    # Copy configuration file
    if [ -f "$FLINK_DIR/config/flink-conf.yaml" ]; then
        log_info "Using custom Flink configuration..."
        cp "$FLINK_DIR/config/flink-conf.yaml" "./flink-conf.yaml"
    fi
    
    # Start Flink services
    log_info "Starting Flink services..."
    docker-compose -f docker-compose.flink.yml up -d
    
    # Wait for Flink to be ready
    log_info "Waiting for Flink cluster to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$FLINK_REST_URL/overview" &> /dev/null; then
            break
        fi
        log_info "Waiting for Flink... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Flink cluster is not ready after $max_attempts attempts"
        return 1
    fi
    
    log_success "Flink cluster started successfully"
    
    # Show cluster information
    log_info "Flink cluster information:"
    curl -s "$FLINK_REST_URL/overview" | jq '.' || echo "Could not retrieve cluster info"
}

setup_monitoring() {
    log_info "Setting up monitoring and metrics..."
    
    # Start monitoring stack
    cd "$PROJECT_ROOT/backend/docker"
    
    if [ -f "docker-compose.flink.yml" ]; then
        log_info "Starting Prometheus and Grafana..."
        docker-compose -f docker-compose.flink.yml up -d flink-prometheus flink-grafana
        
        # Wait for services to be ready
        sleep 10
        
        log_info "Monitoring services:"
        echo "  - Prometheus: http://localhost:9091"
        echo "  - Grafana: http://localhost:3001 (admin/admin)"
        echo "  - Flink Web UI: $FLINK_REST_URL"
    fi
    
    log_success "Monitoring setup completed"
}

create_sample_job_config() {
    log_info "Creating sample job configuration..."
    
    local config_file="$FLINK_DIR/jobs/sample-config.json"
    
    cat > "$config_file" << EOF
{
    "job_name": "tda-streaming-${ENVIRONMENT}",
    "parallelism": ${PARALLELISM},
    "checkpoint_interval": 30000,
    "window_size": 100,
    "slide_interval": 10,
    "window_timeout": 60,
    "max_dimension": 2,
    "max_persistence": 1.0,
    "kafka_bootstrap_servers": "${KAFKA_BOOTSTRAP_SERVERS}",
    "schema_registry_url": "${SCHEMA_REGISTRY_URL}",
    "input_topic": "tda_jobs",
    "output_topic": "tda_results",
    "consumer_group": "flink-tda-consumer-${ENVIRONMENT}",
    "buffer_timeout": 100,
    "network_buffer_size": 32768,
    "enable_object_reuse": true
}
EOF

    log_success "Sample job configuration created at: $config_file"
}

validate_setup() {
    log_info "Validating setup..."
    
    # Check Flink cluster
    if curl -s "$FLINK_REST_URL/overview" &> /dev/null; then
        log_success "Flink cluster is accessible"
    else
        log_error "Flink cluster is not accessible"
        return 1
    fi
    
    # Check Kafka topics
    if [ "$SKIP_KAFKA" = false ]; then
        log_info "Validating Kafka topics..."
        # This would need to be implemented based on available tools
        log_success "Kafka topics validation skipped (implement if needed)"
    fi
    
    # Check JAR files
    local jar_count=$(find "$PROJECT_ROOT/backend/docker/flink-jars" -name "*.jar" | wc -l)
    if [ "$jar_count" -gt 0 ]; then
        log_success "Found $jar_count JAR files"
    else
        log_warning "No JAR files found"
    fi
    
    log_success "Setup validation completed"
}

show_next_steps() {
    echo ""
    log_success "==================================================================="
    log_success "TDA Flink Cluster Setup Complete!"
    log_success "==================================================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Submit a TDA streaming job:"
    echo "   cd $FLINK_DIR/scripts"
    echo "   ./submit-job.sh --job tda-streaming-job.py --config sample-config.json"
    echo ""
    echo "2. Monitor the cluster:"
    echo "   - Flink Web UI: $FLINK_REST_URL"
    echo "   - Prometheus: http://localhost:9091"
    echo "   - Grafana: http://localhost:3001"
    echo ""
    echo "3. Test with sample data:"
    echo "   cd $PROJECT_ROOT/backend"
    echo "   python3 test_kafka_integration.py"
    echo ""
    echo "4. Check logs:"
    echo "   docker-compose -f $PROJECT_ROOT/backend/docker/docker-compose.flink.yml logs -f"
    echo ""
    echo "Configuration files:"
    echo "   - Flink config: $FLINK_DIR/config/flink-conf.yaml"
    echo "   - Job config: $FLINK_DIR/jobs/sample-config.json"
    echo "   - Scripts: $FLINK_DIR/scripts/"
    echo ""
}

# =============================================================================
# Main execution
# =============================================================================

main() {
    log_info "Starting TDA Flink cluster setup..."
    
    check_prerequisites
    
    if [ "$SKIP_JARS" = false ]; then
        download_required_jars
    fi
    
    if [ "$SKIP_KAFKA" = false ]; then
        setup_kafka_topics
    fi
    
    start_flink_cluster
    setup_monitoring
    create_sample_job_config
    validate_setup
    show_next_steps
    
    log_success "TDA Flink cluster setup completed successfully!"
}

# Error handling
trap 'log_error "Setup failed on line $LINENO. Exit code: $?"' ERR

# Run main function
main "$@"
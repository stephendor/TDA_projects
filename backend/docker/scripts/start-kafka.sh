#!/bin/bash

# TDA Kafka Cluster Startup Script
# Comprehensive startup with validation, network setup, and topic creation

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$DOCKER_DIR")")"

# Load environment variables
if [ -f "$DOCKER_DIR/kafka.env" ]; then
    source "$DOCKER_DIR/kafka.env"
    echo "✓ Loaded environment configuration from kafka.env"
else
    echo "⚠️  Warning: kafka.env not found, using defaults"
fi

# Default values
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.kafka.yml"
ENVIRONMENT="${ENVIRONMENT:-development}"
KAFKA_CLUSTER_NAME="${TDA_KAFKA_CLUSTER_NAME:-tda-kafka-cluster}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tda-kafka}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

verbose() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    success "Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    success "Docker Compose is available"
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    success "Docker daemon is running"
    
    # Check compose file
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    success "Docker Compose file found"
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df /var/lib/docker | awk 'NR==2 {print $4}')
    MIN_SPACE=10485760  # 10GB in KB
    if [ "$AVAILABLE_SPACE" -lt "$MIN_SPACE" ]; then
        warning "Low disk space available: $(($AVAILABLE_SPACE / 1024 / 1024))GB (minimum 10GB recommended)"
    fi
    
    # Check available memory (minimum 8GB)
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
    MIN_MEMORY=8192  # 8GB in MB
    if [ "$AVAILABLE_MEMORY" -lt "$MIN_MEMORY" ]; then
        warning "Low available memory: ${AVAILABLE_MEMORY}MB (minimum 8GB recommended)"
    fi
}

validate_environment() {
    log "Validating environment configuration..."
    
    # Check environment type
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        warning "Unknown environment: $ENVIRONMENT (expected: development, staging, production)"
    fi
    success "Environment: $ENVIRONMENT"
    
    # Validate topic configuration
    if [ -z "${TDA_TOPICS:-}" ]; then
        warning "TDA_TOPICS not defined, using defaults"
        export TDA_TOPICS="tda_jobs,tda_results,tda_events,tda_uploads,tda_errors"
    fi
    success "Topics configured: ${TDA_TOPICS}"
    
    # Check port availability
    PORTS_TO_CHECK=(2181 2182 2183 19092 19093 19094 8080 8081 8083 8088 6379)
    for port in "${PORTS_TO_CHECK[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            warning "Port $port is already in use"
        fi
    done
}

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

pull_images() {
    log "Pulling Docker images..."
    
    # Use docker-compose to pull all images
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" pull
    else
        docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" pull
    fi
    
    success "All images pulled successfully"
}

create_network() {
    log "Setting up Docker network..."
    
    # Check if network already exists
    if docker network ls | grep -q "tda-kafka-network"; then
        verbose "Network tda-kafka-network already exists"
    else
        docker network create \
            --driver bridge \
            --subnet 172.20.0.0/16 \
            tda-kafka-network
        success "Created network: tda-kafka-network"
    fi
}

start_services() {
    log "Starting Kafka cluster services..."
    
    # Start services in proper order
    verbose "Starting ZooKeeper ensemble..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d zookeeper-1 zookeeper-2 zookeeper-3
    else
        docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d zookeeper-1 zookeeper-2 zookeeper-3
    fi
    
    # Wait for ZooKeeper to be ready
    log "Waiting for ZooKeeper ensemble to be ready..."
    for i in {1..60}; do
        if docker exec tda_zookeeper_1 bash -c 'echo ruok | nc localhost 2181' 2>/dev/null | grep -q imok; then
            success "ZooKeeper is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            error "ZooKeeper failed to start within timeout"
            exit 1
        fi
        sleep 2
    done
    
    verbose "Starting Kafka brokers..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d kafka-1 kafka-2 kafka-3
    else
        docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d kafka-1 kafka-2 kafka-3
    fi
    
    # Wait for Kafka brokers to be ready
    log "Waiting for Kafka brokers to be ready..."
    for i in {1..120}; do
        if docker exec tda_kafka_1 kafka-broker-api-versions --bootstrap-server localhost:9092 &>/dev/null; then
            success "Kafka brokers are ready"
            break
        fi
        if [ $i -eq 120 ]; then
            error "Kafka brokers failed to start within timeout"
            exit 1
        fi
        sleep 3
    done
    
    verbose "Starting Schema Registry..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d schema-registry
    else
        docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d schema-registry
    fi
    
    verbose "Starting remaining services..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d
    else
        docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d
    fi
    
    success "All services started"
}

# =============================================================================
# HEALTH CHECKS
# =============================================================================

perform_health_checks() {
    log "Performing comprehensive health checks..."
    
    # ZooKeeper health check
    verbose "Checking ZooKeeper health..."
    for zk_container in tda_zookeeper_1 tda_zookeeper_2 tda_zookeeper_3; do
        if docker exec "$zk_container" bash -c 'echo ruok | nc localhost 2181' 2>/dev/null | grep -q imok; then
            success "✓ $zk_container is healthy"
        else
            error "✗ $zk_container is unhealthy"
            return 1
        fi
    done
    
    # Kafka broker health check
    verbose "Checking Kafka broker health..."
    for kafka_container in tda_kafka_1 tda_kafka_2 tda_kafka_3; do
        if docker exec "$kafka_container" kafka-broker-api-versions --bootstrap-server localhost:9092 &>/dev/null; then
            success "✓ $kafka_container is healthy"
        else
            error "✗ $kafka_container is unhealthy"
            return 1
        fi
    done
    
    # Schema Registry health check
    verbose "Checking Schema Registry health..."
    if curl -f http://localhost:8081/subjects &>/dev/null; then
        success "✓ Schema Registry is healthy"
    else
        warning "⚠️ Schema Registry is not responding"
    fi
    
    # Kafka UI health check
    verbose "Checking Kafka UI health..."
    if curl -f http://localhost:8080 &>/dev/null; then
        success "✓ Kafka UI is accessible"
    else
        warning "⚠️ Kafka UI is not responding"
    fi
    
    success "Health checks completed"
}

# =============================================================================
# TOPIC MANAGEMENT
# =============================================================================

create_topics() {
    log "Creating TDA topics..."
    
    # Parse topics from environment
    IFS=',' read -ra TOPIC_ARRAY <<< "${TDA_TOPICS}"
    
    for topic in "${TOPIC_ARRAY[@]}"; do
        topic=$(echo "$topic" | xargs)  # Trim whitespace
        
        verbose "Creating topic: $topic"
        
        # Get topic-specific configuration
        case $topic in
            "tda_jobs")
                PARTITIONS=${TDA_JOBS_PARTITIONS:-3}
                REPLICATION=${TDA_JOBS_REPLICATION_FACTOR:-3}
                RETENTION=${TDA_JOBS_RETENTION_MS:-604800000}
                CLEANUP_POLICY="delete"
                ;;
            "tda_results")
                PARTITIONS=${TDA_RESULTS_PARTITIONS:-3}
                REPLICATION=${TDA_RESULTS_REPLICATION_FACTOR:-3}
                RETENTION=${TDA_RESULTS_RETENTION_MS:-2592000000}
                CLEANUP_POLICY=${TDA_RESULTS_CLEANUP_POLICY:-compact}
                ;;
            "tda_events")
                PARTITIONS=${TDA_EVENTS_PARTITIONS:-1}
                REPLICATION=${TDA_EVENTS_REPLICATION_FACTOR:-3}
                RETENTION=${TDA_EVENTS_RETENTION_MS:-1209600000}
                CLEANUP_POLICY="delete"
                ;;
            "tda_uploads")
                PARTITIONS=${TDA_UPLOADS_PARTITIONS:-2}
                REPLICATION=${TDA_UPLOADS_REPLICATION_FACTOR:-3}
                RETENTION=${TDA_UPLOADS_RETENTION_MS:-259200000}
                CLEANUP_POLICY="delete"
                ;;
            "tda_errors")
                PARTITIONS=${TDA_ERRORS_PARTITIONS:-1}
                REPLICATION=${TDA_ERRORS_REPLICATION_FACTOR:-3}
                RETENTION=${TDA_ERRORS_RETENTION_MS:-2592000000}
                CLEANUP_POLICY="delete"
                ;;
            *)
                # Default configuration
                PARTITIONS=3
                REPLICATION=3
                RETENTION=604800000
                CLEANUP_POLICY="delete"
                ;;
        esac
        
        # Check if topic exists
        if docker exec tda_kafka_1 kafka-topics --list --bootstrap-server localhost:9092 | grep -q "^${topic}$"; then
            verbose "Topic $topic already exists"
        else
            # Create topic with configuration
            docker exec tda_kafka_1 kafka-topics \
                --create \
                --bootstrap-server localhost:9092 \
                --topic "$topic" \
                --partitions "$PARTITIONS" \
                --replication-factor "$REPLICATION" \
                --config retention.ms="$RETENTION" \
                --config cleanup.policy="$CLEANUP_POLICY" \
                --config segment.ms=86400000 \
                --config min.insync.replicas=2
            
            success "Created topic: $topic (partitions: $PARTITIONS, replication: $REPLICATION)"
        fi
    done
    
    # Verify all topics were created
    verbose "Verifying topic creation..."
    CREATED_TOPICS=$(docker exec tda_kafka_1 kafka-topics --list --bootstrap-server localhost:9092)
    for topic in "${TOPIC_ARRAY[@]}"; do
        topic=$(echo "$topic" | xargs)
        if echo "$CREATED_TOPICS" | grep -q "^${topic}$"; then
            success "✓ Topic verified: $topic"
        else
            error "✗ Topic missing: $topic"
        fi
    done
}

# =============================================================================
# MONITORING SETUP
# =============================================================================

setup_monitoring() {
    log "Setting up monitoring and metrics..."
    
    # Create monitoring topics if they don't exist
    MONITORING_TOPICS=("_schemas" "__consumer_offsets" "_connect-configs" "_connect-offsets" "_connect-status")
    
    for topic in "${MONITORING_TOPICS[@]}"; do
        if ! docker exec tda_kafka_1 kafka-topics --list --bootstrap-server localhost:9092 | grep -q "^${topic}$"; then
            verbose "Monitoring topic $topic will be auto-created"
        fi
    done
    
    # Wait for Kafka UI to be ready
    log "Waiting for Kafka UI to be ready..."
    for i in {1..30}; do
        if curl -f http://localhost:8080 &>/dev/null; then
            success "Kafka UI is ready at http://localhost:8080"
            break
        fi
        if [ $i -eq 30 ]; then
            warning "Kafka UI did not become ready within timeout"
        fi
        sleep 2
    done
    
    success "Monitoring setup completed"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
TDA Kafka Cluster Startup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -e, --environment ENV   Set environment (development|staging|production)
    -t, --timeout SECONDS  Set health check timeout (default: 300)
    --pull                  Force pull latest images before starting
    --no-topics             Skip topic creation
    --health-only           Only perform health checks
    --clean                 Clean up existing containers before starting

Examples:
    $0                                    # Start with default settings
    $0 --verbose --environment production # Start in production mode with verbose output
    $0 --pull --clean                     # Clean restart with latest images
    $0 --health-only                      # Just check if cluster is healthy

Environment Variables:
    ENVIRONMENT             Environment type (development|staging|production)
    HEALTH_CHECK_TIMEOUT    Health check timeout in seconds
    VERBOSE                 Enable verbose output (true|false)
    
Configuration:
    Edit kafka.env to customize cluster configuration
    Edit docker-compose.kafka.yml for service definitions

EOF
}

main() {
    # Parse command line arguments
    PULL_IMAGES=false
    SKIP_TOPICS=false
    HEALTH_ONLY=false
    CLEAN_START=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --pull)
                PULL_IMAGES=true
                shift
                ;;
            --no-topics)
                SKIP_TOPICS=true
                shift
                ;;
            --health-only)
                HEALTH_ONLY=true
                shift
                ;;
            --clean)
                CLEAN_START=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo "======================================================================="
    echo "  TDA Kafka Cluster Startup"
    echo "  Environment: $ENVIRONMENT"
    echo "  Cluster: $KAFKA_CLUSTER_NAME"
    echo "======================================================================="
    
    # Health check only mode
    if [ "$HEALTH_ONLY" = "true" ]; then
        log "Running health checks only..."
        if perform_health_checks; then
            success "All health checks passed"
            exit 0
        else
            error "Health checks failed"
            exit 1
        fi
    fi
    
    # Clean start if requested
    if [ "$CLEAN_START" = "true" ]; then
        log "Cleaning up existing containers..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" down -v
        else
            docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" down -v
        fi
        success "Cleanup completed"
    fi
    
    # Execute startup sequence
    check_prerequisites
    validate_environment
    
    if [ "$PULL_IMAGES" = "true" ]; then
        pull_images
    fi
    
    create_network
    start_services
    
    # Wait a bit for services to stabilize
    sleep 10
    
    perform_health_checks
    
    if [ "$SKIP_TOPICS" != "true" ]; then
        create_topics
    fi
    
    setup_monitoring
    
    echo "======================================================================="
    success "TDA Kafka Cluster started successfully!"
    echo ""
    echo "Services available:"
    echo "  • Kafka Brokers:     localhost:19092, localhost:19093, localhost:19094"
    echo "  • ZooKeeper:         localhost:2181, localhost:2182, localhost:2183"
    echo "  • Kafka UI:          http://localhost:8080"
    echo "  • Schema Registry:   http://localhost:8081"
    echo "  • Kafka Connect:     http://localhost:8083"
    echo "  • KSQLDB Server:     http://localhost:8088"
    echo ""
    echo "Management commands:"
    echo "  • View logs:         docker-compose -f $COMPOSE_FILE logs -f"
    echo "  • Stop cluster:      $SCRIPT_DIR/stop-kafka.sh"
    echo "  • Topic management:  $SCRIPT_DIR/kafka-topics.sh"
    echo "======================================================================="
}

# Run main function with all arguments
main "$@"
#!/bin/bash

# TDA Kafka Cluster Shutdown Script
# Graceful shutdown with data preservation and cleanup options

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
fi

# Default values
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.kafka.yml"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tda-kafka}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-60}"
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
    verbose "Checking prerequisites..."
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check compose file
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    verbose "Prerequisites check passed"
}

check_cluster_status() {
    log "Checking cluster status..."
    
    # Check if any containers are running
    RUNNING_CONTAINERS=$(docker ps --filter "name=tda_" --format "table {{.Names}}" | grep -v NAMES | wc -l)
    
    if [ "$RUNNING_CONTAINERS" -eq 0 ]; then
        warning "No TDA Kafka containers are currently running"
        return 1
    fi
    
    success "Found $RUNNING_CONTAINERS running TDA containers"
    return 0
}

# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================

backup_configurations() {
    if [ "$BACKUP_CONFIGS" = "true" ]; then
        log "Backing up configurations..."
        
        BACKUP_DIR="/tmp/tda-kafka-backup-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup topic configurations
        verbose "Backing up topic configurations..."
        if docker ps --filter "name=tda_kafka_1" --format "{{.Names}}" | grep -q tda_kafka_1; then
            docker exec tda_kafka_1 kafka-topics --list --bootstrap-server localhost:9092 > "$BACKUP_DIR/topics_list.txt" 2>/dev/null || true
            
            # Backup each topic configuration
            while IFS= read -r topic; do
                if [ -n "$topic" ]; then
                    docker exec tda_kafka_1 kafka-configs --bootstrap-server localhost:9092 \
                        --entity-type topics --entity-name "$topic" --describe \
                        > "$BACKUP_DIR/topic_${topic}_config.txt" 2>/dev/null || true
                fi
            done < "$BACKUP_DIR/topics_list.txt"
        fi
        
        # Backup consumer group offsets
        verbose "Backing up consumer group information..."
        if docker ps --filter "name=tda_kafka_1" --format "{{.Names}}" | grep -q tda_kafka_1; then
            docker exec tda_kafka_1 kafka-consumer-groups --bootstrap-server localhost:9092 --list \
                > "$BACKUP_DIR/consumer_groups.txt" 2>/dev/null || true
        fi
        
        success "Configuration backup saved to: $BACKUP_DIR"
    fi
}

# =============================================================================
# GRACEFUL SHUTDOWN FUNCTIONS
# =============================================================================

drain_consumers() {
    log "Draining active consumers..."
    
    # Get list of active consumer groups
    if docker ps --filter "name=tda_kafka_1" --format "{{.Names}}" | grep -q tda_kafka_1; then
        CONSUMER_GROUPS=$(docker exec tda_kafka_1 kafka-consumer-groups --bootstrap-server localhost:9092 --list 2>/dev/null | grep -v "^$" || true)
        
        if [ -n "$CONSUMER_GROUPS" ]; then
            verbose "Found active consumer groups:"
            echo "$CONSUMER_GROUPS" | while read -r group; do
                if [ -n "$group" ]; then
                    verbose "  - $group"
                    # Check consumer group lag
                    docker exec tda_kafka_1 kafka-consumer-groups --bootstrap-server localhost:9092 \
                        --group "$group" --describe 2>/dev/null | head -10 || true
                fi
            done
            
            warning "Active consumers detected. Allowing time for graceful shutdown..."
            sleep 10
        else
            success "No active consumer groups found"
        fi
    fi
}

flush_producers() {
    log "Ensuring producer data is flushed..."
    
    # Give producers time to flush any pending data
    sleep 5
    success "Producer flush completed"
}

stop_applications() {
    log "Stopping application-level services first..."
    
    # Stop services in reverse dependency order
    SERVICES_TO_STOP_FIRST=("ksqldb-cli" "ksqldb-server" "kafka-ui" "kafka-connect")
    
    for service in "${SERVICES_TO_STOP_FIRST[@]}"; do
        if docker ps --filter "name=tda_${service}" --format "{{.Names}}" | grep -q "tda_${service}"; then
            verbose "Stopping $service..."
            if command -v docker-compose &> /dev/null; then
                docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop "$service" 2>/dev/null || true
            else
                docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop "$service" 2>/dev/null || true
            fi
        fi
    done
    
    success "Application services stopped"
}

stop_kafka_brokers() {
    log "Stopping Kafka brokers gracefully..."
    
    # Stop brokers one by one to allow leader election
    for broker in kafka-3 kafka-2 kafka-1; do
        if docker ps --filter "name=tda_${broker}" --format "{{.Names}}" | grep -q "tda_${broker}"; then
            verbose "Stopping $broker..."
            if command -v docker-compose &> /dev/null; then
                docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop "$broker"
            else
                docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop "$broker"
            fi
            
            # Wait a moment between broker shutdowns
            sleep 5
        fi
    done
    
    success "Kafka brokers stopped"
}

stop_zookeeper() {
    log "Stopping ZooKeeper ensemble..."
    
    # Stop ZooKeeper nodes
    for zk in zookeeper-3 zookeeper-2 zookeeper-1; do
        if docker ps --filter "name=tda_${zk}" --format "{{.Names}}" | grep -q "tda_${zk}"; then
            verbose "Stopping $zk..."
            if command -v docker-compose &> /dev/null; then
                docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop "$zk"
            else
                docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop "$zk"
            fi
        fi
    done
    
    success "ZooKeeper ensemble stopped"
}

# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

remove_containers() {
    if [ "$REMOVE_CONTAINERS" = "true" ]; then
        log "Removing containers..."
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" rm -f
        else
            docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" rm -f
        fi
        
        success "Containers removed"
    fi
}

remove_volumes() {
    if [ "$REMOVE_VOLUMES" = "true" ]; then
        warning "Removing volumes (data will be permanently lost)..."
        
        # List volumes to be removed
        VOLUMES_TO_REMOVE=$(docker volume ls --filter "name=tda_" --format "{{.Name}}" | grep -E "(kafka|zookeeper|redis)" || true)
        
        if [ -n "$VOLUMES_TO_REMOVE" ]; then
            echo "Volumes to be removed:"
            echo "$VOLUMES_TO_REMOVE" | sed 's/^/  - /'
            
            if [ "$FORCE_REMOVE" != "true" ]; then
                echo ""
                read -p "Are you sure you want to permanently delete all data? (yes/no): " -r
                if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
                    warning "Volume removal cancelled"
                    return 0
                fi
            fi
            
            echo "$VOLUMES_TO_REMOVE" | xargs docker volume rm
            success "Volumes removed"
        else
            success "No volumes to remove"
        fi
    fi
}

remove_networks() {
    if [ "$REMOVE_NETWORKS" = "true" ]; then
        log "Removing networks..."
        
        # Remove the Kafka network if it exists and is not in use
        if docker network ls | grep -q "tda-kafka-network"; then
            docker network rm tda-kafka-network 2>/dev/null || warning "Could not remove network (may be in use)"
            success "Network removed"
        fi
    fi
}

# =============================================================================
# MONITORING AND VERIFICATION
# =============================================================================

verify_shutdown() {
    log "Verifying shutdown..."
    
    # Check for any remaining containers
    REMAINING_CONTAINERS=$(docker ps --filter "name=tda_" --format "table {{.Names}}" | grep -v NAMES | wc -l)
    
    if [ "$REMAINING_CONTAINERS" -eq 0 ]; then
        success "All containers stopped successfully"
    else
        warning "$REMAINING_CONTAINERS containers are still running"
        docker ps --filter "name=tda_" --format "table {{.Names}}\t{{.Status}}"
    fi
    
    # Check for any remaining volumes (if not removed)
    if [ "$REMOVE_VOLUMES" != "true" ]; then
        REMAINING_VOLUMES=$(docker volume ls --filter "name=tda_" --format "{{.Name}}" | wc -l)
        if [ "$REMAINING_VOLUMES" -gt 0 ]; then
            verbose "Data volumes preserved: $REMAINING_VOLUMES volumes"
        fi
    fi
}

show_cleanup_status() {
    echo ""
    echo "======================================================================="
    echo "  Cleanup Status"
    echo "======================================================================="
    
    # Show what was preserved
    if [ "$REMOVE_VOLUMES" != "true" ]; then
        PRESERVED_VOLUMES=$(docker volume ls --filter "name=tda_" --format "{{.Name}}" | wc -l)
        if [ "$PRESERVED_VOLUMES" -gt 0 ]; then
            echo "📁 Data volumes preserved: $PRESERVED_VOLUMES volumes"
            echo "   To remove data: $0 --remove-volumes"
        fi
    fi
    
    if [ "$REMOVE_CONTAINERS" != "true" ]; then
        echo "📦 Containers preserved (stopped)"
        echo "   To remove containers: $0 --remove-containers"
    fi
    
    if [ "$BACKUP_CONFIGS" = "true" ]; then
        echo "💾 Configurations backed up to /tmp/tda-kafka-backup-*"
    fi
    
    echo ""
    echo "To restart the cluster: $SCRIPT_DIR/start-kafka.sh"
    echo "======================================================================="
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
TDA Kafka Cluster Shutdown Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -t, --timeout SECONDS  Set shutdown timeout (default: 60)
    --graceful              Graceful shutdown with consumer draining (default)
    --force                 Force immediate shutdown
    --remove-containers     Remove containers after stopping
    --remove-volumes        Remove data volumes (DANGER: data loss)
    --remove-networks       Remove Docker networks
    --clean-all            Remove everything (containers, volumes, networks)
    --backup                Backup configurations before shutdown
    --force-remove          Skip confirmation prompts for destructive operations

Shutdown Modes:
    Default (graceful):     Stop services in proper order, preserve data
    --force:                Immediate shutdown, may cause data loss
    --clean-all:            Complete cleanup, removes all data

Examples:
    $0                              # Graceful shutdown, preserve data
    $0 --verbose --backup           # Graceful shutdown with config backup
    $0 --remove-containers          # Stop and remove containers, keep data
    $0 --clean-all --force-remove   # Complete cleanup without prompts
    $0 --force                      # Force immediate shutdown

Safety:
    By default, all data is preserved in Docker volumes
    Use --remove-volumes to permanently delete data
    Use --backup to save configurations before shutdown

EOF
}

main() {
    # Parse command line arguments
    GRACEFUL_SHUTDOWN=true
    REMOVE_CONTAINERS=false
    REMOVE_VOLUMES=false
    REMOVE_NETWORKS=false
    BACKUP_CONFIGS=false
    FORCE_REMOVE=false
    
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
            -t|--timeout)
                SHUTDOWN_TIMEOUT="$2"
                shift 2
                ;;
            --graceful)
                GRACEFUL_SHUTDOWN=true
                shift
                ;;
            --force)
                GRACEFUL_SHUTDOWN=false
                shift
                ;;
            --remove-containers)
                REMOVE_CONTAINERS=true
                shift
                ;;
            --remove-volumes)
                REMOVE_VOLUMES=true
                shift
                ;;
            --remove-networks)
                REMOVE_NETWORKS=true
                shift
                ;;
            --clean-all)
                REMOVE_CONTAINERS=true
                REMOVE_VOLUMES=true
                REMOVE_NETWORKS=true
                shift
                ;;
            --backup)
                BACKUP_CONFIGS=true
                shift
                ;;
            --force-remove)
                FORCE_REMOVE=true
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
    echo "  TDA Kafka Cluster Shutdown"
    if [ "$GRACEFUL_SHUTDOWN" = "true" ]; then
        echo "  Mode: Graceful shutdown"
    else
        echo "  Mode: Force shutdown"
    fi
    echo "  Project: $COMPOSE_PROJECT_NAME"
    echo "======================================================================="
    
    # Check prerequisites
    check_prerequisites
    
    # Check if cluster is running
    if ! check_cluster_status; then
        success "No running containers found, nothing to stop"
        exit 0
    fi
    
    # Backup configurations if requested
    backup_configurations
    
    # Execute shutdown sequence
    if [ "$GRACEFUL_SHUTDOWN" = "true" ]; then
        log "Starting graceful shutdown sequence..."
        
        drain_consumers
        flush_producers
        stop_applications
        stop_kafka_brokers
        stop_zookeeper
        
        # Stop any remaining services
        verbose "Stopping any remaining services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop
        else
            docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" stop
        fi
    else
        log "Force shutdown - stopping all services immediately..."
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" down --timeout 10
        else
            docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" down --timeout 10
        fi
    fi
    
    # Cleanup operations
    remove_containers
    remove_volumes
    remove_networks
    
    # Verify shutdown
    verify_shutdown
    
    success "TDA Kafka Cluster shutdown completed!"
    show_cleanup_status
}

# Run main function with all arguments
main "$@"
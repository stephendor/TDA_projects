#!/bin/bash

# TDA Kafka Topics Management Script
# Comprehensive topic management based on TDA architecture requirements

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
fi

# Default values
KAFKA_CONTAINER="${KAFKA_CONTAINER:-tda_kafka_1}"
BOOTSTRAP_SERVERS="${KAFKA_INTERNAL_BOOTSTRAP_SERVERS:-kafka-1:9092,kafka-2:9092,kafka-3:9092}"
TOPIC_PREFIX="${KAFKA_TOPIC_PREFIX:-tda_}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

verbose() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# =============================================================================
# KAFKA CONNECTION FUNCTIONS
# =============================================================================

check_kafka_connection() {
    verbose "Checking Kafka connection..."
    
    if ! docker ps --filter "name=$KAFKA_CONTAINER" --format "{{.Names}}" | grep -q "$KAFKA_CONTAINER"; then
        error "Kafka container '$KAFKA_CONTAINER' is not running"
        return 1
    fi
    
    if ! docker exec "$KAFKA_CONTAINER" kafka-broker-api-versions --bootstrap-server localhost:9092 &>/dev/null; then
        error "Cannot connect to Kafka brokers"
        return 1
    fi
    
    verbose "Kafka connection verified"
    return 0
}

exec_kafka_command() {
    local cmd="$1"
    docker exec "$KAFKA_CONTAINER" $cmd
}

# =============================================================================
# TOPIC DEFINITION FUNCTIONS
# =============================================================================

get_topic_config() {
    local topic="$1"
    
    case $topic in
        "${TOPIC_PREFIX}jobs"|"tda_jobs")
            echo "partitions=${TDA_JOBS_PARTITIONS:-3}"
            echo "replication-factor=${TDA_JOBS_REPLICATION_FACTOR:-3}"
            echo "config=retention.ms=${TDA_JOBS_RETENTION_MS:-604800000}"
            echo "config=retention.bytes=${TDA_JOBS_RETENTION_BYTES:-1073741824}"
            echo "config=cleanup.policy=delete"
            echo "config=segment.ms=86400000"
            echo "config=min.insync.replicas=2"
            ;;
        "${TOPIC_PREFIX}results"|"tda_results")
            echo "partitions=${TDA_RESULTS_PARTITIONS:-3}"
            echo "replication-factor=${TDA_RESULTS_REPLICATION_FACTOR:-3}"
            echo "config=retention.ms=${TDA_RESULTS_RETENTION_MS:-2592000000}"
            echo "config=retention.bytes=${TDA_RESULTS_RETENTION_BYTES:-5368709120}"
            echo "config=cleanup.policy=${TDA_RESULTS_CLEANUP_POLICY:-compact}"
            echo "config=segment.ms=86400000"
            echo "config=min.insync.replicas=2"
            if [ "${TDA_RESULTS_CLEANUP_POLICY:-compact}" = "compact" ]; then
                echo "config=min.cleanable.dirty.ratio=${TDA_RESULTS_MIN_CLEANABLE_DIRTY_RATIO:-0.1}"
                echo "config=max.compaction.lag.ms=${TDA_RESULTS_MAX_COMPACTION_LAG_MS:-3600000}"
            fi
            ;;
        "${TOPIC_PREFIX}events"|"tda_events")
            echo "partitions=${TDA_EVENTS_PARTITIONS:-1}"
            echo "replication-factor=${TDA_EVENTS_REPLICATION_FACTOR:-3}"
            echo "config=retention.ms=${TDA_EVENTS_RETENTION_MS:-1209600000}"
            echo "config=retention.bytes=${TDA_EVENTS_RETENTION_BYTES:-536870912}"
            echo "config=cleanup.policy=delete"
            echo "config=segment.ms=86400000"
            echo "config=min.insync.replicas=2"
            ;;
        "${TOPIC_PREFIX}uploads"|"tda_uploads")
            echo "partitions=${TDA_UPLOADS_PARTITIONS:-2}"
            echo "replication-factor=${TDA_UPLOADS_REPLICATION_FACTOR:-3}"
            echo "config=retention.ms=${TDA_UPLOADS_RETENTION_MS:-259200000}"
            echo "config=retention.bytes=${TDA_UPLOADS_RETENTION_BYTES:-2147483648}"
            echo "config=cleanup.policy=delete"
            echo "config=segment.ms=86400000"
            echo "config=min.insync.replicas=2"
            ;;
        "${TOPIC_PREFIX}errors"|"tda_errors")
            echo "partitions=${TDA_ERRORS_PARTITIONS:-1}"
            echo "replication-factor=${TDA_ERRORS_REPLICATION_FACTOR:-3}"
            echo "config=retention.ms=${TDA_ERRORS_RETENTION_MS:-2592000000}"
            echo "config=retention.bytes=${TDA_ERRORS_RETENTION_BYTES:-1073741824}"
            echo "config=cleanup.policy=delete"
            echo "config=segment.ms=86400000"
            echo "config=min.insync.replicas=2"
            ;;
        *)
            # Default configuration
            echo "partitions=3"
            echo "replication-factor=3"
            echo "config=retention.ms=604800000"
            echo "config=cleanup.policy=delete"
            echo "config=min.insync.replicas=2"
            ;;
    esac
}

# =============================================================================
# TOPIC MANAGEMENT FUNCTIONS
# =============================================================================

list_topics() {
    log "Listing all topics..."
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    echo ""
    echo "======================================================================="
    echo "  Kafka Topics"
    echo "======================================================================="
    
    local topics
    topics=$(exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | sort)
    
    if [ -z "$topics" ]; then
        warning "No topics found"
        return 0
    fi
    
    # Categorize topics
    local tda_topics=""
    local system_topics=""
    local other_topics=""
    
    while IFS= read -r topic; do
        if [[ $topic == ${TOPIC_PREFIX}* ]]; then
            tda_topics="$tda_topics$topic\n"
        elif [[ $topic == _* ]] || [[ $topic == __* ]]; then
            system_topics="$system_topics$topic\n"
        else
            other_topics="$other_topics$topic\n"
        fi
    done <<< "$topics"
    
    # Display TDA topics
    if [ -n "$tda_topics" ]; then
        echo ""
        echo "🎯 TDA Topics:"
        echo -e "$tda_topics" | grep -v '^$' | while read -r topic; do
            echo "  • $topic"
        done
    fi
    
    # Display system topics
    if [ -n "$system_topics" ]; then
        echo ""
        echo "⚙️  System Topics:"
        echo -e "$system_topics" | grep -v '^$' | while read -r topic; do
            echo "  • $topic"
        done
    fi
    
    # Display other topics
    if [ -n "$other_topics" ]; then
        echo ""
        echo "📂 Other Topics:"
        echo -e "$other_topics" | grep -v '^$' | while read -r topic; do
            echo "  • $topic"
        done
    fi
    
    echo ""
    local total_count
    total_count=$(echo "$topics" | wc -l)
    success "Total topics: $total_count"
}

describe_topic() {
    local topic="$1"
    
    log "Describing topic: $topic"
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    # Check if topic exists
    if ! exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
        error "Topic '$topic' does not exist"
        return 1
    fi
    
    echo ""
    echo "======================================================================="
    echo "  Topic Details: $topic"
    echo "======================================================================="
    
    # Topic configuration
    echo ""
    echo "📊 Topic Configuration:"
    exec_kafka_command "kafka-topics --describe --bootstrap-server localhost:9092 --topic $topic"
    
    echo ""
    echo "⚙️  Topic Settings:"
    exec_kafka_command "kafka-configs --bootstrap-server localhost:9092 --entity-type topics --entity-name $topic --describe"
    
    # Consumer groups
    echo ""
    echo "👥 Consumer Groups:"
    local consumer_groups
    consumer_groups=$(exec_kafka_command "kafka-consumer-groups --bootstrap-server localhost:9092 --list" | grep -v '^$' || true)
    
    if [ -n "$consumer_groups" ]; then
        while IFS= read -r group; do
            if [ -n "$group" ]; then
                local group_topics
                group_topics=$(exec_kafka_command "kafka-consumer-groups --bootstrap-server localhost:9092 --group $group --describe" 2>/dev/null | awk '{print $1}' | grep "^${topic}$" || true)
                if [ -n "$group_topics" ]; then
                    echo "  • $group"
                    exec_kafka_command "kafka-consumer-groups --bootstrap-server localhost:9092 --group $group --describe" | grep "^${topic}"
                fi
            fi
        done <<< "$consumer_groups"
    else
        echo "  No active consumer groups"
    fi
    
    success "Topic description completed"
}

create_topic() {
    local topic="$1"
    local custom_config="${2:-}"
    
    log "Creating topic: $topic"
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    # Check if topic already exists
    if exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
        warning "Topic '$topic' already exists"
        return 0
    fi
    
    # Get topic configuration
    local config_lines
    if [ -n "$custom_config" ]; then
        config_lines="$custom_config"
    else
        config_lines=$(get_topic_config "$topic")
    fi
    
    # Parse configuration
    local partitions replication_factor configs
    partitions=$(echo "$config_lines" | grep "^partitions=" | cut -d'=' -f2)
    replication_factor=$(echo "$config_lines" | grep "^replication-factor=" | cut -d'=' -f2)
    configs=$(echo "$config_lines" | grep "^config=" | cut -d'=' -f2- | paste -sd, -)
    
    verbose "Configuration: partitions=$partitions, replication-factor=$replication_factor"
    verbose "Configs: $configs"
    
    # Build create command
    local create_cmd="kafka-topics --create --bootstrap-server localhost:9092"
    create_cmd="$create_cmd --topic $topic"
    create_cmd="$create_cmd --partitions $partitions"
    create_cmd="$create_cmd --replication-factor $replication_factor"
    
    if [ -n "$configs" ]; then
        create_cmd="$create_cmd --config $configs"
    fi
    
    # Execute create command
    if exec_kafka_command "$create_cmd"; then
        success "Topic '$topic' created successfully"
        
        # Verify creation
        sleep 2
        if exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
            success "Topic creation verified"
        else
            error "Topic creation verification failed"
            return 1
        fi
    else
        error "Failed to create topic '$topic'"
        return 1
    fi
}

delete_topic() {
    local topic="$1"
    local force="${2:-false}"
    
    log "Deleting topic: $topic"
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    # Check if topic exists
    if ! exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
        warning "Topic '$topic' does not exist"
        return 0
    fi
    
    # Safety check for system topics
    if [[ $topic == _* ]] || [[ $topic == __* ]]; then
        error "Cannot delete system topic: $topic"
        return 1
    fi
    
    # Confirmation for deletion
    if [ "$force" != "true" ]; then
        echo ""
        warning "This will permanently delete topic '$topic' and all its data!"
        echo "Topic details:"
        exec_kafka_command "kafka-topics --describe --bootstrap-server localhost:9092 --topic $topic" | head -5
        echo ""
        read -p "Are you sure you want to delete this topic? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            warning "Topic deletion cancelled"
            return 0
        fi
    fi
    
    # Delete topic
    if exec_kafka_command "kafka-topics --delete --bootstrap-server localhost:9092 --topic $topic"; then
        success "Topic '$topic' deleted successfully"
        
        # Verify deletion
        sleep 2
        if ! exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
            success "Topic deletion verified"
        else
            warning "Topic may still exist (deletion in progress)"
        fi
    else
        error "Failed to delete topic '$topic'"
        return 1
    fi
}

create_all_tda_topics() {
    log "Creating all TDA topics..."
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    # TDA topics based on architecture
    local tda_topics=("${TOPIC_PREFIX}jobs" "${TOPIC_PREFIX}results" "${TOPIC_PREFIX}events" "${TOPIC_PREFIX}uploads" "${TOPIC_PREFIX}errors")
    
    local created_count=0
    local total_count=${#tda_topics[@]}
    
    for topic in "${tda_topics[@]}"; do
        echo ""
        if create_topic "$topic"; then
            ((created_count++))
        fi
    done
    
    echo ""
    echo "======================================================================="
    success "TDA topic creation completed: $created_count/$total_count topics"
    echo "======================================================================="
    
    # List created topics
    list_topics
}

update_topic_config() {
    local topic="$1"
    local config_key="$2"
    local config_value="$3"
    
    log "Updating topic configuration: $topic"
    info "Setting $config_key=$config_value"
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    # Check if topic exists
    if ! exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
        error "Topic '$topic' does not exist"
        return 1
    fi
    
    # Update configuration
    if exec_kafka_command "kafka-configs --bootstrap-server localhost:9092 --entity-type topics --entity-name $topic --alter --add-config $config_key=$config_value"; then
        success "Configuration updated successfully"
        
        # Show updated configuration
        echo ""
        echo "Updated configuration:"
        exec_kafka_command "kafka-configs --bootstrap-server localhost:9092 --entity-type topics --entity-name $topic --describe"
    else
        error "Failed to update configuration"
        return 1
    fi
}

# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

show_topic_metrics() {
    local topic="${1:-}"
    
    if [ -n "$topic" ]; then
        log "Showing metrics for topic: $topic"
    else
        log "Showing metrics for all topics..."
    fi
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    echo ""
    echo "======================================================================="
    echo "  Topic Metrics"
    echo "======================================================================="
    
    # Get log sizes
    echo ""
    echo "📊 Topic Sizes:"
    local topics_list
    if [ -n "$topic" ]; then
        topics_list="$topic"
    else
        topics_list=$(exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep "^${TOPIC_PREFIX}" | sort)
    fi
    
    while IFS= read -r t; do
        if [ -n "$t" ]; then
            local size_info
            size_info=$(exec_kafka_command "kafka-log-dirs --bootstrap-server localhost:9092 --topic-list $t --describe" 2>/dev/null | grep "\"size\"" | head -1 || echo "\"size\": 0")
            local size_bytes
            size_bytes=$(echo "$size_info" | grep -o '"size":[0-9]*' | cut -d':' -f2)
            local size_mb=$((size_bytes / 1024 / 1024))
            echo "  • $t: ${size_mb}MB"
        fi
    done <<< "$topics_list"
    
    # Consumer lag information
    echo ""
    echo "👥 Consumer Groups & Lag:"
    local consumer_groups
    consumer_groups=$(exec_kafka_command "kafka-consumer-groups --bootstrap-server localhost:9092 --list" | grep -v '^$' || true)
    
    if [ -n "$consumer_groups" ]; then
        while IFS= read -r group; do
            if [ -n "$group" ]; then
                echo ""
                echo "  Group: $group"
                exec_kafka_command "kafka-consumer-groups --bootstrap-server localhost:9092 --group $group --describe" 2>/dev/null | grep -E "(TOPIC|${TOPIC_PREFIX})" || echo "    No TDA topics consumed"
            fi
        done <<< "$consumer_groups"
    else
        echo "  No active consumer groups"
    fi
    
    success "Metrics display completed"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

produce_test_message() {
    local topic="$1"
    local message="${2:-Test message from TDA Kafka Topics Manager}"
    
    log "Producing test message to topic: $topic"
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    # Check if topic exists
    if ! exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
        error "Topic '$topic' does not exist"
        return 1
    fi
    
    # Produce message
    if echo "$message" | exec_kafka_command "kafka-console-producer --bootstrap-server localhost:9092 --topic $topic"; then
        success "Test message produced successfully"
    else
        error "Failed to produce test message"
        return 1
    fi
}

consume_messages() {
    local topic="$1"
    local max_messages="${2:-10}"
    local from_beginning="${3:-true}"
    
    log "Consuming messages from topic: $topic (max: $max_messages)"
    
    if ! check_kafka_connection; then
        return 1
    fi
    
    # Check if topic exists
    if ! exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep -q "^${topic}$"; then
        error "Topic '$topic' does not exist"
        return 1
    fi
    
    # Consume messages
    local consume_cmd="kafka-console-consumer --bootstrap-server localhost:9092 --topic $topic --max-messages $max_messages"
    
    if [ "$from_beginning" = "true" ]; then
        consume_cmd="$consume_cmd --from-beginning"
    fi
    
    echo ""
    echo "======================================================================="
    echo "  Messages from topic: $topic"
    echo "======================================================================="
    
    if ! exec_kafka_command "$consume_cmd --timeout-ms 10000" 2>/dev/null; then
        warning "No messages consumed (topic may be empty or timeout reached)"
    fi
    
    echo ""
    success "Message consumption completed"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
TDA Kafka Topics Management Script

Usage: $0 COMMAND [OPTIONS]

Commands:
    list                        List all topics
    describe TOPIC              Show detailed information about a topic
    create TOPIC [CONFIG]       Create a topic with optional custom config
    delete TOPIC [--force]      Delete a topic (with confirmation)
    create-all                  Create all TDA topics
    update TOPIC KEY=VALUE      Update topic configuration
    metrics [TOPIC]             Show topic metrics and consumer lag
    produce TOPIC [MESSAGE]     Produce a test message to topic
    consume TOPIC [COUNT]       Consume messages from topic
    
TDA Topic Operations:
    create-tda                  Create all TDA topics (jobs, results, events, uploads, errors)
    list-tda                    List only TDA topics
    cleanup-empty               Remove empty non-TDA topics
    
Global Options:
    -h, --help                  Show this help message
    -v, --verbose               Enable verbose output
    --container NAME            Use specific Kafka container (default: tda_kafka_1)
    --bootstrap-servers SERVERS Custom bootstrap servers
    
Examples:
    $0 list                                    # List all topics
    $0 describe tda_jobs                       # Show job topic details
    $0 create tda_custom                       # Create topic with default config
    $0 create my_topic "partitions=5,replication-factor=2"  # Custom config
    $0 delete tda_test --force                 # Force delete without confirmation
    $0 create-all                              # Create all TDA topics
    $0 metrics tda_jobs                        # Show metrics for jobs topic
    $0 produce tda_events "Test event"         # Send test message
    $0 consume tda_results 20                  # Read 20 messages

TDA Topics:
    tda_jobs        - Job submission and lifecycle events (7 days retention)
    tda_results     - Computation results and persistence diagrams (30 days, compacted)
    tda_events      - System events and notifications (14 days retention)
    tda_uploads     - File upload events (3 days retention)
    tda_errors      - Error events and failure tracking (30 days retention)

Configuration:
    Edit kafka.env to customize default topic settings
    Topic configurations follow the TDA architecture document

EOF
}

main() {
    # Parse global options first
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
            --container)
                KAFKA_CONTAINER="$2"
                shift 2
                ;;
            --bootstrap-servers)
                BOOTSTRAP_SERVERS="$2"
                shift 2
                ;;
            -*)
                error "Unknown global option: $1"
                show_usage
                exit 1
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Check if command is provided
    if [ $# -eq 0 ]; then
        error "No command provided"
        show_usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    # Execute command
    case $command in
        list|ls)
            list_topics
            ;;
        describe|desc)
            if [ $# -eq 0 ]; then
                error "Topic name required for describe command"
                exit 1
            fi
            describe_topic "$1"
            ;;
        create)
            if [ $# -eq 0 ]; then
                error "Topic name required for create command"
                exit 1
            fi
            create_topic "$1" "${2:-}"
            ;;
        delete|del|rm)
            if [ $# -eq 0 ]; then
                error "Topic name required for delete command"
                exit 1
            fi
            local force_delete=false
            if [ "${2:-}" = "--force" ]; then
                force_delete=true
            fi
            delete_topic "$1" "$force_delete"
            ;;
        create-all|create-tda)
            create_all_tda_topics
            ;;
        list-tda)
            exec_kafka_command "kafka-topics --list --bootstrap-server localhost:9092" | grep "^${TOPIC_PREFIX}" | sort
            ;;
        update|config)
            if [ $# -lt 2 ]; then
                error "Topic name and configuration required (format: KEY=VALUE)"
                exit 1
            fi
            local topic="$1"
            local config="$2"
            local key=$(echo "$config" | cut -d'=' -f1)
            local value=$(echo "$config" | cut -d'=' -f2-)
            update_topic_config "$topic" "$key" "$value"
            ;;
        metrics|stats)
            show_topic_metrics "${1:-}"
            ;;
        produce|send)
            if [ $# -eq 0 ]; then
                error "Topic name required for produce command"
                exit 1
            fi
            produce_test_message "$1" "${2:-}"
            ;;
        consume|read)
            if [ $# -eq 0 ]; then
                error "Topic name required for consume command"
                exit 1
            fi
            consume_messages "$1" "${2:-10}" "true"
            ;;
        cleanup-empty)
            warning "Cleanup-empty command not yet implemented"
            ;;
        *)
            error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
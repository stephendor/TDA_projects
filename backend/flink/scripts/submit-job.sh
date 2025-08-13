#!/bin/bash
set -e

# =============================================================================
# TDA Flink Job Submission Script
# =============================================================================
#
# This script submits TDA streaming jobs to Apache Flink cluster with
# proper configuration, dependency management, and monitoring setup.
#
# Usage:
#   ./submit-job.sh --job JOB_FILE [OPTIONS]
#   ./submit-job.sh --list
#   ./submit-job.sh --cancel JOB_ID
#   ./submit-job.sh --status [JOB_ID]
#
# Examples:
#   ./submit-job.sh --job tda-streaming-job.py --config sample-config.json
#   ./submit-job.sh --job tda-streaming-job.py --parallelism 8 --window-size 200
#   ./submit-job.sh --list
#   ./submit-job.sh --cancel abc123
#   ./submit-job.sh --status abc123
#
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FLINK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JOBS_DIR="$FLINK_DIR/jobs"

# Default values
JOB_FILE=""
CONFIG_FILE=""
JOB_NAME=""
PARALLELISM=""
WINDOW_SIZE=""
SLIDE_INTERVAL=""
ACTION="submit"
JOB_ID=""
ENVIRONMENT="${TDA_ENVIRONMENT:-dev}"
FLINK_REST_URL="${FLINK_REST_URL:-http://localhost:8082}"
VERBOSE=false
DRY_RUN=false

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

log_debug() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
TDA Flink Job Submission Script

USAGE:
    $0 --job JOB_FILE [OPTIONS]
    $0 --list
    $0 --cancel JOB_ID
    $0 --status [JOB_ID]

ACTIONS:
    --job FILE          Submit a job file (required for submit action)
    --list              List all running jobs
    --cancel ID         Cancel a running job by ID
    --status [ID]       Show status of job(s)

OPTIONS:
    --config FILE       Job configuration file (JSON)
    --job-name NAME     Override job name
    --parallelism N     Set job parallelism
    --window-size N     Set window size for streaming jobs
    --slide-interval N  Set window slide interval
    --environment ENV   Target environment (dev, staging, prod)
    --flink-url URL     Flink REST API URL
    --dry-run           Show what would be done without executing
    --verbose           Enable verbose output
    -h, --help          Show this help message

EXAMPLES:
    # Submit with configuration file
    $0 --job tda-streaming-job.py --config sample-config.json

    # Submit with command line parameters
    $0 --job tda-streaming-job.py --parallelism 8 --window-size 200

    # List all jobs
    $0 --list

    # Cancel a job
    $0 --cancel abc123-def456-789

    # Check job status
    $0 --status abc123-def456-789

ENVIRONMENT VARIABLES:
    TDA_ENVIRONMENT     Target environment (dev, staging, prod)
    FLINK_REST_URL      Flink REST API URL
    KAFKA_BOOTSTRAP_SERVERS  Kafka cluster endpoints
    SCHEMA_REGISTRY_URL      Schema registry URL

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --job)
            JOB_FILE="$2"
            ACTION="submit"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --parallelism)
            PARALLELISM="$2"
            shift 2
            ;;
        --window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --slide-interval)
            SLIDE_INTERVAL="$2"
            shift 2
            ;;
        --list)
            ACTION="list"
            shift
            ;;
        --cancel)
            ACTION="cancel"
            JOB_ID="$2"
            shift 2
            ;;
        --status)
            ACTION="status"
            if [[ $# -gt 1 && ! $2 =~ ^-- ]]; then
                JOB_ID="$2"
                shift 2
            else
                shift
            fi
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --flink-url)
            FLINK_REST_URL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Utility Functions
# =============================================================================

check_prerequisites() {
    log_debug "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("curl" "jq" "python3")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check if Flink is accessible
    if ! curl -s "$FLINK_REST_URL/overview" &> /dev/null; then
        log_error "Flink cluster is not accessible at: $FLINK_REST_URL"
        log_info "Make sure Flink cluster is running:"
        log_info "  cd $PROJECT_ROOT/backend/docker"
        log_info "  docker-compose -f docker-compose.flink.yml up -d"
        exit 1
    fi
    
    log_debug "Prerequisites check passed"
}

# Flink REST API functions
flink_api_call() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    
    local url="$FLINK_REST_URL$endpoint"
    log_debug "API call: $method $url"
    
    if [ "$method" = "GET" ]; then
        curl -s -X GET "$url"
    elif [ "$method" = "POST" ]; then
        if [ -n "$data" ]; then
            curl -s -X POST -H "Content-Type: application/json" -d "$data" "$url"
        else
            curl -s -X POST "$url"
        fi
    elif [ "$method" = "DELETE" ]; then
        curl -s -X DELETE "$url"
    else
        log_error "Unsupported HTTP method: $method"
        return 1
    fi
}

get_cluster_overview() {
    flink_api_call "GET" "/overview"
}

list_jobs() {
    flink_api_call "GET" "/jobs"
}

get_job_details() {
    local job_id="$1"
    flink_api_call "GET" "/jobs/$job_id"
}

cancel_job() {
    local job_id="$1"
    flink_api_call "DELETE" "/jobs/$job_id"
}

# =============================================================================
# Action Functions
# =============================================================================

action_list_jobs() {
    log_info "Listing Flink jobs..."
    
    local jobs_response
    jobs_response=$(list_jobs)
    
    if [ $? -ne 0 ]; then
        log_error "Failed to retrieve jobs list"
        exit 1
    fi
    
    echo "$jobs_response" | jq -r '
        if .jobs | length == 0 then
            "No jobs found"
        else
            "ID\tNAME\tSTATE\tSTART_TIME\tDURATION",
            (.jobs[] | 
                "\(.id)\t\(.name)\t\(.state)\t\(.["start-time"] | strftime("%Y-%m-%d %H:%M:%S"))\t\(.duration)")
        end
    ' | column -t -s $'\t'
}

action_job_status() {
    if [ -z "$JOB_ID" ]; then
        log_info "Showing status of all jobs..."
        action_list_jobs
        return 0
    fi
    
    log_info "Getting status for job: $JOB_ID"
    
    local job_details
    job_details=$(get_job_details "$JOB_ID")
    
    if [ $? -ne 0 ]; then
        log_error "Failed to get job details for: $JOB_ID"
        exit 1
    fi
    
    echo "$job_details" | jq -r '
        "Job Details:",
        "  ID: \(.jid)",
        "  Name: \(.name)",
        "  State: \(.state)",
        "  Start Time: \(.["start-time"] | strftime("%Y-%m-%d %H:%M:%S"))",
        "  Duration: \(.duration)",
        "  Parallelism: \(.vertices[0].parallelism // "N/A")",
        "",
        "Vertices:",
        (.vertices[] | "  - \(.name) (\(.parallelism) tasks)")
    '
}

action_cancel_job() {
    if [ -z "$JOB_ID" ]; then
        log_error "Job ID is required for cancel action"
        exit 1
    fi
    
    log_info "Cancelling job: $JOB_ID"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would cancel job: $JOB_ID"
        return 0
    fi
    
    local response
    response=$(cancel_job "$JOB_ID")
    
    if [ $? -eq 0 ]; then
        log_success "Job cancelled successfully"
    else
        log_error "Failed to cancel job: $JOB_ID"
        exit 1
    fi
}

action_submit_job() {
    if [ -z "$JOB_FILE" ]; then
        log_error "Job file is required for submit action"
        exit 1
    fi
    
    # Resolve job file path
    if [ ! -f "$JOB_FILE" ]; then
        # Try relative to jobs directory
        if [ -f "$JOBS_DIR/$JOB_FILE" ]; then
            JOB_FILE="$JOBS_DIR/$JOB_FILE"
        else
            log_error "Job file not found: $JOB_FILE"
            exit 1
        fi
    fi
    
    log_info "Submitting job: $(basename "$JOB_FILE")"
    
    # Load configuration
    local job_config="{}"
    if [ -n "$CONFIG_FILE" ]; then
        if [ ! -f "$CONFIG_FILE" ]; then
            # Try relative to jobs directory
            if [ -f "$JOBS_DIR/$CONFIG_FILE" ]; then
                CONFIG_FILE="$JOBS_DIR/$CONFIG_FILE"
            else
                log_error "Configuration file not found: $CONFIG_FILE"
                exit 1
            fi
        fi
        
        log_info "Using configuration file: $(basename "$CONFIG_FILE")"
        job_config=$(cat "$CONFIG_FILE")
    fi
    
    # Override configuration with command line parameters
    if [ -n "$JOB_NAME" ]; then
        job_config=$(echo "$job_config" | jq --arg name "$JOB_NAME" '.job_name = $name')
    fi
    
    if [ -n "$PARALLELISM" ]; then
        job_config=$(echo "$job_config" | jq --arg p "$PARALLELISM" '.parallelism = ($p | tonumber)')
    fi
    
    if [ -n "$WINDOW_SIZE" ]; then
        job_config=$(echo "$job_config" | jq --arg w "$WINDOW_SIZE" '.window_size = ($w | tonumber)')
    fi
    
    if [ -n "$SLIDE_INTERVAL" ]; then
        job_config=$(echo "$job_config" | jq --arg s "$SLIDE_INTERVAL" '.slide_interval = ($s | tonumber)')
    fi
    
    # Set environment-specific defaults
    case $ENVIRONMENT in
        dev)
            job_config=$(echo "$job_config" | jq '.parallelism = 2 | .checkpoint_interval = 60000')
            ;;
        staging)
            job_config=$(echo "$job_config" | jq '.parallelism = 4 | .checkpoint_interval = 30000')
            ;;
        prod)
            job_config=$(echo "$job_config" | jq '.parallelism = 8 | .checkpoint_interval = 30000')
            ;;
    esac
    
    log_debug "Final job configuration:"
    if [ "$VERBOSE" = true ]; then
        echo "$job_config" | jq '.'
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would submit job with configuration:"
        echo "$job_config" | jq '.'
        return 0
    fi
    
    # Prepare job submission
    local job_name
    job_name=$(echo "$job_config" | jq -r '.job_name // "tda-streaming-job"')
    
    # Create temporary configuration file for the job
    local temp_config
    temp_config=$(mktemp)
    echo "$job_config" > "$temp_config"
    
    # Submit the job
    log_info "Submitting job to Flink cluster..."
    
    # For PyFlink jobs, we need to execute them directly
    # In a production environment, this would upload and submit via REST API
    
    local submit_command="python3 '$JOB_FILE' --config '$temp_config' --job-name '$job_name'"
    
    if [ "$ENVIRONMENT" != "dev" ]; then
        # Use docker exec for containerized environments
        submit_command="docker-compose -f $PROJECT_ROOT/backend/docker/docker-compose.flink.yml exec -T flink-jobmanager $submit_command"
    fi
    
    log_debug "Executing: $submit_command"
    
    # Execute the job submission
    if eval "$submit_command"; then
        log_success "Job submitted successfully: $job_name"
        
        # Wait a moment and show job status
        sleep 5
        log_info "Job status:"
        local jobs
        jobs=$(list_jobs)
        echo "$jobs" | jq -r '.jobs[] | select(.name == "'$job_name'") | "  ID: \(.id)\n  State: \(.state)\n  Start Time: \(.["start-time"] | strftime("%Y-%m-%d %H:%M:%S"))"'
    else
        log_error "Job submission failed"
        exit 1
    fi
    
    # Cleanup
    rm -f "$temp_config"
}

# =============================================================================
# Main execution
# =============================================================================

main() {
    # Set Flink URL based on environment
    case $ENVIRONMENT in
        dev)
            FLINK_REST_URL="${FLINK_REST_URL:-http://localhost:8082}"
            ;;
        staging|prod)
            FLINK_REST_URL="${FLINK_REST_URL:-http://flink-jobmanager:8081}"
            ;;
    esac
    
    log_debug "Environment: $ENVIRONMENT"
    log_debug "Flink URL: $FLINK_REST_URL"
    log_debug "Action: $ACTION"
    
    check_prerequisites
    
    case $ACTION in
        submit)
            action_submit_job
            ;;
        list)
            action_list_jobs
            ;;
        status)
            action_job_status
            ;;
        cancel)
            action_cancel_job
            ;;
        *)
            log_error "Unknown action: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# Error handling
trap 'log_error "Script failed on line $LINENO. Exit code: $?"' ERR

# Run main function
main "$@"
#!/bin/bash

# TDA Kafka Setup Validation Script
# Validates configuration files and setup completeness

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"

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
    echo -e "${BLUE}[VALIDATE]${NC} $1"
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

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

validate_files() {
    log "Validating setup files..."
    
    local required_files=(
        "$DOCKER_DIR/docker-compose.kafka.yml"
        "$DOCKER_DIR/kafka.env"
        "$SCRIPT_DIR/start-kafka.sh"
        "$SCRIPT_DIR/stop-kafka.sh"
        "$SCRIPT_DIR/kafka-topics.sh"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            success "Found: $(basename "$file")"
        else
            error "Missing: $file"
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        success "All required files present"
        return 0
    else
        error "Missing ${#missing_files[@]} required files"
        return 1
    fi
}

validate_permissions() {
    log "Validating script permissions..."
    
    local scripts=(
        "$SCRIPT_DIR/start-kafka.sh"
        "$SCRIPT_DIR/stop-kafka.sh"
        "$SCRIPT_DIR/kafka-topics.sh"
        "$SCRIPT_DIR/validate-setup.sh"
    )
    
    local non_executable=()
    
    for script in "${scripts[@]}"; do
        if [ -x "$script" ]; then
            success "Executable: $(basename "$script")"
        else
            error "Not executable: $script"
            non_executable+=("$script")
        fi
    done
    
    if [ ${#non_executable[@]} -eq 0 ]; then
        success "All scripts are executable"
        return 0
    else
        error "${#non_executable[@]} scripts are not executable"
        echo "Fix with: chmod +x ${non_executable[*]}"
        return 1
    fi
}

validate_docker_compose() {
    log "Validating Docker Compose configuration..."
    
    local compose_file="$DOCKER_DIR/docker-compose.kafka.yml"
    
    # Check if docker-compose can parse the file
    if command -v docker-compose &> /dev/null; then
        if docker-compose -f "$compose_file" config &>/dev/null; then
            success "Docker Compose file is valid"
        else
            error "Docker Compose file has syntax errors"
            echo "Check with: docker-compose -f $compose_file config"
            return 1
        fi
    elif docker compose version &> /dev/null; then
        if docker compose -f "$compose_file" config &>/dev/null; then
            success "Docker Compose file is valid"
        else
            error "Docker Compose file has syntax errors"
            echo "Check with: docker compose -f $compose_file config"
            return 1
        fi
    else
        warning "Docker Compose not available, skipping syntax validation"
    fi
    
    # Validate service definitions
    local required_services=(
        "zookeeper-1" "zookeeper-2" "zookeeper-3"
        "kafka-1" "kafka-2" "kafka-3"
        "schema-registry" "kafka-ui"
    )
    
    for service in "${required_services[@]}"; do
        if grep -q "^  ${service}:" "$compose_file"; then
            success "Service defined: $service"
        else
            error "Missing service: $service"
        fi
    done
    
    return 0
}

validate_environment() {
    log "Validating environment configuration..."
    
    local env_file="$DOCKER_DIR/kafka.env"
    
    if [ ! -f "$env_file" ]; then
        error "Environment file not found: $env_file"
        return 1
    fi
    
    # Source the environment file
    set -a
    source "$env_file"
    set +a
    
    # Check critical variables
    local critical_vars=(
        "KAFKA_BOOTSTRAP_SERVERS"
        "TDA_TOPICS"
        "KAFKA_DEFAULT_PARTITIONS"
        "KAFKA_DEFAULT_REPLICATION_FACTOR"
    )
    
    for var in "${critical_vars[@]}"; do
        if [ -n "${!var:-}" ]; then
            success "Environment variable set: $var=${!var}"
        else
            warning "Environment variable not set: $var"
        fi
    done
    
    # Validate topic configuration
    if [ -n "${TDA_TOPICS:-}" ]; then
        IFS=',' read -ra topics <<< "$TDA_TOPICS"
        success "Configured ${#topics[@]} TDA topics: ${TDA_TOPICS}"
    fi
    
    return 0
}

validate_prerequisites() {
    log "Validating prerequisites..."
    
    # Check Docker
    if command -v docker &> /dev/null; then
        success "Docker is installed"
        
        # Check Docker daemon
        if docker info &> /dev/null; then
            success "Docker daemon is running"
        else
            error "Docker daemon is not running"
            return 1
        fi
    else
        error "Docker is not installed"
        return 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        success "Docker Compose (standalone) is available"
    elif docker compose version &> /dev/null; then
        success "Docker Compose (plugin) is available"
    else
        error "Docker Compose is not available"
        return 1
    fi
    
    # Check available resources
    local available_memory
    available_memory=$(free -m | awk 'NR==2{print $7}')
    if [ "$available_memory" -gt 8192 ]; then
        success "Sufficient memory available: ${available_memory}MB"
    else
        warning "Low memory available: ${available_memory}MB (recommended: 8GB+)"
    fi
    
    local available_disk
    available_disk=$(df /var/lib/docker 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}' || echo "0")
    if [ "$available_disk" -gt 10 ]; then
        success "Sufficient disk space: ${available_disk}GB"
    else
        warning "Low disk space: ${available_disk}GB (recommended: 10GB+)"
    fi
    
    return 0
}

validate_network_ports() {
    log "Validating network port availability..."
    
    local required_ports=(2181 2182 2183 19092 19093 19094 8080 8081 8083 8088 6379)
    local occupied_ports=()
    
    for port in "${required_ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            warning "Port $port is already in use"
            occupied_ports+=("$port")
        else
            success "Port $port is available"
        fi
    done
    
    if [ ${#occupied_ports[@]} -eq 0 ]; then
        success "All required ports are available"
        return 0
    else
        warning "${#occupied_ports[@]} ports are occupied: ${occupied_ports[*]}"
        echo "This may cause conflicts when starting the cluster"
        return 1
    fi
}

validate_script_syntax() {
    log "Validating script syntax..."
    
    local scripts=(
        "$SCRIPT_DIR/start-kafka.sh"
        "$SCRIPT_DIR/stop-kafka.sh"
        "$SCRIPT_DIR/kafka-topics.sh"
    )
    
    for script in "${scripts[@]}"; do
        if bash -n "$script" 2>/dev/null; then
            success "Syntax OK: $(basename "$script")"
        else
            error "Syntax error in: $script"
            bash -n "$script"
            return 1
        fi
    done
    
    return 0
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
TDA Kafka Setup Validation Script

Usage: $0 [OPTIONS]

Options:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    --fix          Attempt to fix common issues
    --quick        Skip resource-intensive checks

Validation Areas:
    - File presence and permissions
    - Docker Compose configuration syntax
    - Environment variable configuration
    - System prerequisites and resources
    - Network port availability
    - Script syntax validation

Examples:
    $0                  # Full validation
    $0 --quick          # Quick validation
    $0 --fix            # Validation with auto-fix

EOF
}

fix_common_issues() {
    log "Attempting to fix common issues..."
    
    # Fix script permissions
    chmod +x "$SCRIPT_DIR"/*.sh
    success "Fixed script permissions"
    
    # Create missing directories
    mkdir -p "$DOCKER_DIR/logs"
    success "Created logs directory"
    
    return 0
}

main() {
    local verbose=false
    local fix_issues=false
    local quick_mode=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            --fix)
                fix_issues=true
                shift
                ;;
            --quick)
                quick_mode=true
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
    echo "  TDA Kafka Setup Validation"
    echo "  Mode: $([ "$quick_mode" = "true" ] && echo "Quick" || echo "Full")"
    echo "======================================================================="
    
    local validation_results=()
    local failed_count=0
    
    # Run validations
    if validate_files; then
        validation_results+=("✓ Files")
    else
        validation_results+=("✗ Files")
        ((failed_count++))
    fi
    
    if validate_permissions; then
        validation_results+=("✓ Permissions")
    else
        validation_results+=("✗ Permissions")
        ((failed_count++))
    fi
    
    if validate_docker_compose; then
        validation_results+=("✓ Docker Compose")
    else
        validation_results+=("✗ Docker Compose")
        ((failed_count++))
    fi
    
    if validate_environment; then
        validation_results+=("✓ Environment")
    else
        validation_results+=("✗ Environment")
        ((failed_count++))
    fi
    
    if [ "$quick_mode" != "true" ]; then
        if validate_prerequisites; then
            validation_results+=("✓ Prerequisites")
        else
            validation_results+=("✗ Prerequisites")
            ((failed_count++))
        fi
        
        if validate_network_ports; then
            validation_results+=("✓ Network Ports")
        else
            validation_results+=("✗ Network Ports")
            ((failed_count++))
        fi
        
        if validate_script_syntax; then
            validation_results+=("✓ Script Syntax")
        else
            validation_results+=("✗ Script Syntax")
            ((failed_count++))
        fi
    fi
    
    # Fix issues if requested
    if [ "$fix_issues" = "true" ] && [ $failed_count -gt 0 ]; then
        echo ""
        fix_common_issues
        echo ""
        warning "Re-run validation to verify fixes"
    fi
    
    # Summary
    echo ""
    echo "======================================================================="
    echo "  Validation Summary"
    echo "======================================================================="
    
    for result in "${validation_results[@]}"; do
        echo "  $result"
    done
    
    echo ""
    if [ $failed_count -eq 0 ]; then
        success "All validations passed! Setup is ready."
        echo ""
        echo "Next steps:"
        echo "  1. Start cluster: ./scripts/start-kafka.sh"
        echo "  2. Verify health: ./scripts/start-kafka.sh --health-only"
        echo "  3. Access Kafka UI: http://localhost:8080"
        exit 0
    else
        error "$failed_count validation(s) failed"
        echo ""
        echo "Fix issues and re-run validation"
        if [ "$fix_issues" != "true" ]; then
            echo "Use --fix to attempt automatic fixes"
        fi
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
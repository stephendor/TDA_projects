#!/bin/bash
# TDA Platform Deployment Script

set -e

# Configuration
ENVIRONMENT=${1:-development}
VERSION=${2:-latest}
COMPOSE_FILE="docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
validate_environment() {
    log_info "Validating deployment environment: $ENVIRONMENT"
    
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
        exit 1
    fi
    
    # Check required commands
    for cmd in docker docker-compose; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is not installed"
            exit 1
        fi
    done
    
    log_info "Environment validation passed"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available disk space (minimum 5GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [ $available_space -lt 5242880 ]; then  # 5GB in KB
        log_warn "Low disk space detected. Available: $(($available_space / 1024 / 1024))GB"
    fi
    
    # Check if required ports are available
    for port in 8000 5432 6379 3000 9090; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            log_warn "Port $port is already in use"
        fi
    done
    
    log_info "Pre-deployment checks completed"
}

# Backup existing data (for production)
backup_data() {
    if [ "$ENVIRONMENT" = "production" ]; then
        log_info "Creating backup of production data..."
        
        backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup database
        if docker-compose ps postgres | grep -q "Up"; then
            docker-compose exec -T postgres pg_dump -U tda_user tda_platform > "$backup_dir/database.sql"
            log_info "Database backup created: $backup_dir/database.sql"
        fi
        
        # Backup volumes
        docker run --rm -v tda_projects_postgres_data:/data -v $(pwd)/$backup_dir:/backup alpine tar czf /backup/postgres_data.tar.gz -C /data .
        
        log_info "Data backup completed"
    fi
}

# Deploy services
deploy_services() {
    log_info "Deploying TDA Platform services..."
    
    # Set environment-specific configuration
    export ENVIRONMENT=$ENVIRONMENT
    export VERSION=$VERSION
    
    # Use environment-specific compose file if it exists
    if [ -f "docker-compose.$ENVIRONMENT.yml" ]; then
        COMPOSE_FILE="docker-compose.yml:docker-compose.$ENVIRONMENT.yml"
        log_info "Using environment-specific configuration: docker-compose.$ENVIRONMENT.yml"
    fi
    
    # Pull latest images
    log_info "Pulling latest container images..."
    docker-compose -f $COMPOSE_FILE pull
    
    # Build and start services
    log_info "Building and starting services..."
    docker-compose -f $COMPOSE_FILE up -d --build
    
    log_info "Services deployment initiated"
}

# Health checks
health_checks() {
    log_info "Running health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    for i in {1..30}; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            log_info "API service is healthy"
            break
        else
            log_warn "Waiting for API service... (attempt $i/30)"
            sleep 10
        fi
        
        if [ $i -eq 30 ]; then
            log_error "API service health check failed"
            exit 1
        fi
    done
    
    # Check database connectivity
    if docker-compose exec -T postgres pg_isready -U tda_user > /dev/null; then
        log_info "Database is ready"
    else
        log_error "Database connectivity check failed"
        exit 1
    fi
    
    # Check Redis connectivity
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        log_info "Redis is ready"
    else
        log_error "Redis connectivity check failed"
        exit 1
    fi
    
    log_info "All health checks passed"
}

# Post-deployment tasks
post_deployment() {
    log_info "Running post-deployment tasks..."
    
    # Run database migrations (if any)
    if [ -f "migrations/migrate.py" ]; then
        log_info "Running database migrations..."
        docker-compose exec tda-api python migrations/migrate.py
    fi
    
    # Warm up caches
    log_info "Warming up application caches..."
    curl -s http://localhost:8000/api/v1/warmup > /dev/null || true
    
    # Create initial monitoring dashboards
    if [ "$ENVIRONMENT" = "production" ]; then
        log_info "Setting up production monitoring..."
        # Import Grafana dashboards, configure alerts, etc.
    fi
    
    log_info "Post-deployment tasks completed"
}

# Display deployment information
show_deployment_info() {
    log_info "Deployment Summary"
    echo "=================================="
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Timestamp: $(date)"
    echo ""
    echo "Service URLs:"
    echo "  API Server: http://localhost:8000"
    echo "  Grafana: http://localhost:3000 (admin/tda_admin_password)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    echo "Service Status:"
    docker-compose ps
    echo "=================================="
}

# Rollback function
rollback() {
    log_warn "Initiating rollback..."
    
    if [ -f "docker-compose.rollback.yml" ]; then
        docker-compose -f docker-compose.rollback.yml up -d
        log_info "Rollback completed"
    else
        log_error "No rollback configuration found"
        exit 1
    fi
}

# Main deployment flow
main() {
    log_info "Starting TDA Platform deployment..."
    
    validate_environment
    pre_deployment_checks
    backup_data
    deploy_services
    health_checks
    post_deployment
    show_deployment_info
    
    log_info "🎉 Deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy"|"")
        main
        ;;
    "rollback")
        rollback
        ;;
    "health")
        health_checks
        ;;
    "status")
        docker-compose ps
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|health|status] [environment] [version]"
        echo "  deploy: Deploy the platform (default)"
        echo "  rollback: Rollback to previous version"
        echo "  health: Run health checks only"
        echo "  status: Show service status"
        echo ""
        echo "Environments: development (default), staging, production"
        exit 1
        ;;
esac
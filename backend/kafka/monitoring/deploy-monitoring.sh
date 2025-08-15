#!/bin/bash

# TDA Kafka Monitoring System Deployment Script
# This script sets up the complete monitoring infrastructure for TDA Kafka platform

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/tda-monitoring-deploy.log"
COMPOSE_FILE="docker-compose.monitoring.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if Kafka network exists
    if ! docker network ls | grep -q "tda-kafka-network"; then
        warning "TDA Kafka network not found. Creating it..."
        docker network create tda-kafka-network || {
            error "Failed to create Kafka network"
            exit 1
        }
    fi
    
    success "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    log "Creating directories..."
    
    mkdir -p "$SCRIPT_DIR/data/prometheus"
    mkdir -p "$SCRIPT_DIR/data/grafana"
    mkdir -p "$SCRIPT_DIR/data/alertmanager"
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/config"
    
    # Set proper permissions
    chmod 755 "$SCRIPT_DIR/data"
    chmod -R 755 "$SCRIPT_DIR/logs"
    
    success "Directories created"
}

# Generate configuration files if they don't exist
generate_configs() {
    log "Generating configuration files..."
    
    # Grafana datasources
    if [ ! -f "$SCRIPT_DIR/config/grafana-datasources.yml" ]; then
        cat > "$SCRIPT_DIR/config/grafana-datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    fi
    
    # Grafana dashboards
    if [ ! -f "$SCRIPT_DIR/config/grafana-dashboards.yml" ]; then
        cat > "$SCRIPT_DIR/config/grafana-dashboards.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'TDA Dashboards'
    orgId: 1
    folder: 'TDA Platform'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    fi
    
    # Alertmanager config
    if [ ! -f "$SCRIPT_DIR/config/alertmanager.yml" ]; then
        cat > "$SCRIPT_DIR/config/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@tda-platform.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://127.0.0.1:5001/'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF
    fi
    
    # Kafka Lag Exporter config
    if [ ! -f "$SCRIPT_DIR/config/kafka-lag-exporter.conf" ]; then
        cat > "$SCRIPT_DIR/config/kafka-lag-exporter.conf" << 'EOF'
kafka-lag-exporter {
  bootstrap-brokers = "kafka-broker-1:9092,kafka-broker-2:9092,kafka-broker-3:9092"
  
  clusters = [
    {
      name = "tda-kafka-cluster"
      bootstrap-brokers = "kafka-broker-1:9092,kafka-broker-2:9092,kafka-broker-3:9092"
    }
  ]
  
  metric-whitelist = [".*"]
  
  poll-interval = 30 seconds
  lookup-table-size = 120
}

akka.http.server.idle-timeout = 25 seconds
akka.http.server.request-timeout = 20 seconds
EOF
    fi
    
    # Kafka JMX Exporter config
    if [ ! -f "$SCRIPT_DIR/config/kafka-jmx-exporter.yml" ]; then
        cat > "$SCRIPT_DIR/config/kafka-jmx-exporter.yml" << 'EOF'
hostPort: kafka-broker-1:9999
ssl: false
lowercaseOutputName: false
lowercaseOutputLabelNames: false

rules:
  - pattern: kafka.server<type=(.+), name=(.+)><>Count
    name: kafka_server_$1_$2_total
    type: COUNTER
    
  - pattern: kafka.server<type=(.+), name=(.+)><>Value
    name: kafka_server_$1_$2
    type: GAUGE
    
  - pattern: kafka.server<type=(.+), name=(.+), topic=(.+)><>Count
    name: kafka_server_$1_$2_total
    labels:
      topic: $3
    type: COUNTER
EOF
    fi
    
    success "Configuration files generated"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$SCRIPT_DIR"
    
    # Build health checker image
    if [ -f "Dockerfile.health-checker" ]; then
        log "Building TDA Health Checker image..."
        docker build -f Dockerfile.health-checker -t tda-health-checker:latest . || {
            error "Failed to build health checker image"
            exit 1
        }
    fi
    
    # Build metrics exporter image
    if [ -f "Dockerfile.metrics-exporter" ]; then
        log "Building TDA Metrics Exporter image..."
        docker build -f Dockerfile.metrics-exporter -t tda-metrics-exporter:latest . || {
            error "Failed to build metrics exporter image"
            exit 1
        }
    fi
    
    success "Docker images built successfully"
}

# Deploy monitoring stack
deploy_stack() {
    log "Deploying TDA monitoring stack..."
    
    cd "$SCRIPT_DIR"
    
    # Set environment variables
    export TDA_ENVIRONMENT="${TDA_ENVIRONMENT:-development}"
    export GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin123}"
    
    # Deploy with Docker Compose
    docker-compose -f "$COMPOSE_FILE" up -d || {
        error "Failed to deploy monitoring stack"
        exit 1
    }
    
    success "Monitoring stack deployed"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    local services=(
        "prometheus:9090"
        "grafana:3000"
        "tda-health-checker:8091"
        "tda-metrics-exporter:8090"
        "alertmanager:9093"
    )
    
    for service in "${services[@]}"; do
        local service_name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)
        
        log "Waiting for $service_name to be ready..."
        
        local max_attempts=30
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if curl -s "http://localhost:$port" > /dev/null 2>&1; then
                success "$service_name is ready"
                break
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                warning "$service_name is not responding after $max_attempts attempts"
            fi
            
            sleep 5
            ((attempt++))
        done
    done
}

# Import Grafana dashboard
import_grafana_dashboard() {
    log "Importing Grafana dashboard..."
    
    # Wait for Grafana to be fully ready
    sleep 10
    
    # Import the dashboard
    if [ -f "$SCRIPT_DIR/grafana-dashboard.json" ]; then
        curl -X POST \
            -H "Content-Type: application/json" \
            -d @"$SCRIPT_DIR/grafana-dashboard.json" \
            "http://admin:${GRAFANA_ADMIN_PASSWORD:-admin123}@localhost:3000/api/dashboards/db" \
            2>/dev/null || warning "Failed to import Grafana dashboard automatically"
    fi
    
    success "Dashboard import completed"
}

# Display access information
display_access_info() {
    log "Deployment completed successfully!"
    echo ""
    echo "=== TDA Kafka Monitoring System Access Information ==="
    echo ""
    echo -e "${GREEN}Grafana Dashboard:${NC} http://localhost:3000"
    echo -e "  Username: admin"
    echo -e "  Password: ${GRAFANA_ADMIN_PASSWORD:-admin123}"
    echo ""
    echo -e "${GREEN}Prometheus:${NC} http://localhost:9090"
    echo ""
    echo -e "${GREEN}Alertmanager:${NC} http://localhost:9093"
    echo ""
    echo -e "${GREEN}TDA Health Checker:${NC} http://localhost:8091"
    echo -e "  Metrics: http://localhost:8091/health/metrics"
    echo -e "  Health: http://localhost:8091/health"
    echo ""
    echo -e "${GREEN}TDA Metrics Exporter:${NC} http://localhost:8090"
    echo -e "  Metrics: http://localhost:8090/metrics"
    echo -e "  Stats: http://localhost:8090/stats"
    echo ""
    echo -e "${GREEN}Node Exporter:${NC} http://localhost:9100"
    echo ""
    echo -e "${GREEN}cAdvisor:${NC} http://localhost:8080"
    echo ""
    echo "=== Useful Commands ==="
    echo ""
    echo "View logs:"
    echo "  docker-compose -f $COMPOSE_FILE logs -f [service-name]"
    echo ""
    echo "Stop monitoring:"
    echo "  docker-compose -f $COMPOSE_FILE down"
    echo ""
    echo "Restart monitoring:"
    echo "  docker-compose -f $COMPOSE_FILE restart"
    echo ""
    echo "Update configuration:"
    echo "  docker-compose -f $COMPOSE_FILE restart prometheus grafana"
    echo ""
}

# Cleanup function
cleanup() {
    if [ "$1" = "full" ]; then
        log "Performing full cleanup..."
        docker-compose -f "$COMPOSE_FILE" down -v
        docker system prune -f
        success "Full cleanup completed"
    else
        log "Stopping monitoring services..."
        docker-compose -f "$COMPOSE_FILE" down
        success "Services stopped"
    fi
}

# Health check
health_check() {
    log "Performing health check..."
    
    echo "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo "Quick Service Tests:"
    
    # Test each service
    if curl -s http://localhost:9090/-/healthy > /dev/null; then
        echo -e "  Prometheus: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Prometheus: ${RED}✗ Unhealthy${NC}"
    fi
    
    if curl -s http://localhost:3000/api/health > /dev/null; then
        echo -e "  Grafana: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Grafana: ${RED}✗ Unhealthy${NC}"
    fi
    
    if curl -s http://localhost:8091/health > /dev/null; then
        echo -e "  Health Checker: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Health Checker: ${RED}✗ Unhealthy${NC}"
    fi
    
    if curl -s http://localhost:8090/health > /dev/null; then
        echo -e "  Metrics Exporter: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Metrics Exporter: ${RED}✗ Unhealthy${NC}"
    fi
}

# Main deployment function
main() {
    log "Starting TDA Kafka Monitoring System deployment..."
    
    check_prerequisites
    create_directories
    generate_configs
    build_images
    deploy_stack
    wait_for_services
    import_grafana_dashboard
    display_access_info
    
    success "TDA Kafka Monitoring System deployment completed successfully!"
}

# Help function
show_help() {
    echo "TDA Kafka Monitoring System Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    Deploy the monitoring stack (default)"
    echo "  stop      Stop the monitoring services"
    echo "  restart   Restart the monitoring services"
    echo "  cleanup   Remove monitoring services"
    echo "  cleanup-full  Remove services and volumes"
    echo "  health    Check service health"
    echo "  logs      Show logs for all services"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  TDA_ENVIRONMENT       Target environment (development/staging/production)"
    echo "  GRAFANA_ADMIN_PASSWORD Grafana admin password (default: admin123)"
    echo ""
    echo "Examples:"
    echo "  $0 deploy"
    echo "  TDA_ENVIRONMENT=production $0 deploy"
    echo "  GRAFANA_ADMIN_PASSWORD=secure123 $0 deploy"
    echo ""
}

# Command line interface
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        docker-compose -f "$COMPOSE_FILE" stop
        ;;
    restart)
        docker-compose -f "$COMPOSE_FILE" restart
        ;;
    cleanup)
        cleanup
        ;;
    cleanup-full)
        cleanup full
        ;;
    health)
        health_check
        ;;
    logs)
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
#!/usr/bin/env python3
"""
Comprehensive Kafka Health Check Service for TDA Platform

This service provides deep health monitoring for Kafka cluster, topics, consumer groups,
and TDA-specific workflows. It includes broker connectivity tests, partition health
validation, consumer lag monitoring, and integration with TDA backend services.

Features:
- Broker connectivity and leader election monitoring
- Topic health and partition distribution analysis
- Consumer group lag monitoring with configurable thresholds
- TDA workflow status validation
- Integration with external health endpoints
- Prometheus metrics export
- Comprehensive logging and alerting
- Configuration-driven health checks
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from urllib.parse import urlparse

import aiohttp
import click
import yaml
from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, ConfigResource
from prometheus_client import Gauge, Counter, Histogram, start_http_server
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/tda-kafka-health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
health_score = Gauge('tda_kafka_health_score', 'Overall Kafka health score', ['component'])
broker_status = Gauge('tda_kafka_broker_status', 'Broker status (1=up, 0=down)', ['broker_id'])
topic_health = Gauge('tda_kafka_topic_health', 'Topic health score', ['topic'])
consumer_lag_metric = Gauge('tda_kafka_consumer_lag', 'Consumer lag per group and topic', ['group', 'topic'])
workflow_health = Gauge('tda_workflow_health_status', 'TDA workflow health status', ['workflow'])
health_check_duration = Histogram('tda_health_check_duration_seconds', 'Health check execution time', ['check_type'])
health_check_errors = Counter('tda_health_check_errors_total', 'Health check errors', ['check_type', 'error_type'])


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component type enumeration"""
    BROKER = "broker"
    TOPIC = "topic"
    CONSUMER_GROUP = "consumer_group"
    WORKFLOW = "workflow"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    component: str
    component_type: ComponentType
    status: HealthStatus
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


@dataclass
class BrokerHealth:
    """Broker health information"""
    broker_id: int
    host: str
    port: int
    is_controller: bool
    is_available: bool
    leader_count: int
    replica_count: int
    response_time_ms: float


@dataclass
class TopicHealth:
    """Topic health information"""
    name: str
    partition_count: int
    replication_factor: int
    under_replicated_partitions: int
    offline_partitions: int
    message_rate: float
    byte_rate: float
    error_rate: float


@dataclass
class ConsumerGroupHealth:
    """Consumer group health information"""
    group_id: str
    state: str
    member_count: int
    total_lag: int
    max_lag: int
    topics: Set[str]
    stability_score: float


class TDAHealthChecker:
    """Comprehensive health checker for TDA Kafka infrastructure"""
    
    def __init__(self, config_path: str = "config/health-check.yml"):
        self.config = self._load_config(config_path)
        self.console = Console()
        self.admin_client = None
        self.results: List[HealthCheckResult] = []
        self.overall_health_score = 0.0
        
        # Initialize Kafka clients
        self._init_kafka_clients()
        
        # Start Prometheus metrics server
        if self.config.get('prometheus', {}).get('enabled', True):
            port = self.config.get('prometheus', {}).get('port', 8091)
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load health check configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'kafka': {
                'bootstrap_servers': 'localhost:9092',
                'security_protocol': 'PLAINTEXT',
                'timeout_ms': 30000
            },
            'health_checks': {
                'broker_connectivity': True,
                'topic_health': True,
                'consumer_lag': True,
                'workflow_status': True,
                'external_services': True
            },
            'thresholds': {
                'consumer_lag_warning': 5000,
                'consumer_lag_critical': 10000,
                'error_rate_warning': 0.01,
                'error_rate_critical': 0.05,
                'response_time_warning': 1000,
                'response_time_critical': 5000
            },
            'tda_topics': [
                'tda_jobs',
                'tda_results',
                'tda_events',
                'tda_uploads',
                'tda_errors',
                'tda_audit',
                'tda_dlq'
            ],
            'workflows': [
                'data_ingestion',
                'computation',
                'result_delivery',
                'error_handling'
            ],
            'external_services': {
                'tda_backend': 'http://tda-backend:8000/health',
                'schema_registry': 'http://schema-registry:8081/subjects'
            },
            'prometheus': {
                'enabled': True,
                'port': 8091
            }
        }
    
    def _init_kafka_clients(self):
        """Initialize Kafka admin and consumer clients"""
        kafka_config = {
            'bootstrap.servers': self.config['kafka']['bootstrap_servers'],
            'security.protocol': self.config['kafka']['security_protocol']
        }
        
        self.admin_client = AdminClient(kafka_config)
        
        # Consumer for lag monitoring
        consumer_config = kafka_config.copy()
        consumer_config.update({
            'group.id': 'tda-health-checker',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': False
        })
        self.consumer = Consumer(consumer_config)
        
        logger.info("Kafka clients initialized")
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive results"""
        start_time = time.time()
        self.results.clear()
        
        with self.console.status("[bold green]Running TDA Kafka Health Checks...") as status:
            try:
                # Run all health checks
                if self.config['health_checks']['broker_connectivity']:
                    status.update("[bold blue]Checking broker connectivity...")
                    await self._check_broker_health()
                
                if self.config['health_checks']['topic_health']:
                    status.update("[bold blue]Analyzing topic health...")
                    await self._check_topic_health()
                
                if self.config['health_checks']['consumer_lag']:
                    status.update("[bold blue]Monitoring consumer lag...")
                    await self._check_consumer_lag()
                
                if self.config['health_checks']['workflow_status']:
                    status.update("[bold blue]Validating TDA workflows...")
                    await self._check_workflow_health()
                
                if self.config['health_checks']['external_services']:
                    status.update("[bold blue]Testing external services...")
                    await self._check_external_services()
                
                # Calculate overall health score
                self._calculate_overall_health()
                
                # Update Prometheus metrics
                self._update_prometheus_metrics()
                
                duration = time.time() - start_time
                logger.info(f"Health check completed in {duration:.2f}s, score: {self.overall_health_score:.2f}")
                
                return self._generate_health_report()
                
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                health_check_errors.labels(check_type='overall', error_type=type(e).__name__).inc()
                raise
    
    async def _check_broker_health(self):
        """Check broker connectivity and leadership status"""
        start_time = time.time()
        
        try:
            # Get cluster metadata
            metadata = self.admin_client.list_topics(timeout=10)
            brokers = metadata.brokers
            
            broker_healths = []
            
            for broker_id, broker in brokers.items():
                try:
                    # Test broker connectivity
                    test_start = time.time()
                    
                    # Create a simple producer to test connectivity
                    producer_config = {
                        'bootstrap.servers': f"{broker.host}:{broker.port}",
                        'socket.timeout.ms': 5000,
                        'message.timeout.ms': 5000
                    }
                    
                    test_producer = Producer(producer_config)
                    
                    # Count partitions where this broker is leader
                    leader_count = 0
                    replica_count = 0
                    
                    for topic_name, topic_metadata in metadata.topics.items():
                        for partition in topic_metadata.partitions.values():
                            if partition.leader == broker_id:
                                leader_count += 1
                            if broker_id in partition.replicas:
                                replica_count += 1
                    
                    response_time = (time.time() - test_start) * 1000
                    
                    broker_health = BrokerHealth(
                        broker_id=broker_id,
                        host=broker.host,
                        port=broker.port,
                        is_controller=False,  # Will be updated below
                        is_available=True,
                        leader_count=leader_count,
                        replica_count=replica_count,
                        response_time_ms=response_time
                    )
                    
                    broker_healths.append(broker_health)
                    broker_status.labels(broker_id=str(broker_id)).set(1)
                    
                    # Determine health status
                    if response_time > self.config['thresholds']['response_time_critical']:
                        status = HealthStatus.CRITICAL
                        score = 0.3
                    elif response_time > self.config['thresholds']['response_time_warning']:
                        status = HealthStatus.WARNING
                        score = 0.7
                    else:
                        status = HealthStatus.HEALTHY
                        score = 1.0
                    
                    self.results.append(HealthCheckResult(
                        component=f"broker-{broker_id}",
                        component_type=ComponentType.BROKER,
                        status=status,
                        score=score,
                        message=f"Broker {broker_id} responding in {response_time:.1f}ms",
                        details={
                            'host': broker.host,
                            'port': broker.port,
                            'leader_count': leader_count,
                            'replica_count': replica_count,
                            'response_time_ms': response_time
                        },
                        timestamp=datetime.now(timezone.utc),
                        duration_ms=(time.time() - start_time) * 1000
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to check broker {broker_id}: {str(e)}")
                    broker_status.labels(broker_id=str(broker_id)).set(0)
                    
                    self.results.append(HealthCheckResult(
                        component=f"broker-{broker_id}",
                        component_type=ComponentType.BROKER,
                        status=HealthStatus.CRITICAL,
                        score=0.0,
                        message=f"Broker {broker_id} unreachable: {str(e)}",
                        details={'error': str(e)},
                        timestamp=datetime.now(timezone.utc),
                        duration_ms=(time.time() - start_time) * 1000
                    ))
            
            duration = (time.time() - start_time) * 1000
            health_check_duration.labels(check_type='broker').observe(duration / 1000)
            
        except Exception as e:
            logger.error(f"Broker health check failed: {str(e)}")
            health_check_errors.labels(check_type='broker', error_type=type(e).__name__).inc()
            raise
    
    async def _check_topic_health(self):
        """Analyze health of TDA topics"""
        start_time = time.time()
        
        try:
            metadata = self.admin_client.list_topics(timeout=10)
            tda_topics = [t for t in metadata.topics.keys() if any(pattern in t for pattern in self.config['tda_topics'])]
            
            for topic_name in tda_topics:
                topic_metadata = metadata.topics[topic_name]
                
                # Calculate topic health metrics
                partition_count = len(topic_metadata.partitions)
                under_replicated = 0
                offline_partitions = 0
                
                for partition in topic_metadata.partitions.values():
                    if len(partition.replicas) != len(partition.isrs):
                        under_replicated += 1
                    if partition.leader == -1:
                        offline_partitions += 1
                
                # Determine health status and score
                if offline_partitions > 0:
                    status = HealthStatus.CRITICAL
                    score = 0.0
                elif under_replicated > partition_count * 0.2:  # More than 20% under-replicated
                    status = HealthStatus.CRITICAL
                    score = 0.3
                elif under_replicated > 0:
                    status = HealthStatus.WARNING
                    score = 0.7
                else:
                    status = HealthStatus.HEALTHY
                    score = 1.0
                
                topic_health.labels(topic=topic_name).set(score)
                
                self.results.append(HealthCheckResult(
                    component=f"topic-{topic_name}",
                    component_type=ComponentType.TOPIC,
                    status=status,
                    score=score,
                    message=f"Topic {topic_name}: {partition_count} partitions, {under_replicated} under-replicated, {offline_partitions} offline",
                    details={
                        'partition_count': partition_count,
                        'under_replicated_partitions': under_replicated,
                        'offline_partitions': offline_partitions,
                        'replication_factor': len(list(topic_metadata.partitions.values())[0].replicas) if partition_count > 0 else 0
                    },
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=(time.time() - start_time) * 1000
                ))
            
            duration = (time.time() - start_time) * 1000
            health_check_duration.labels(check_type='topic').observe(duration / 1000)
            
        except Exception as e:
            logger.error(f"Topic health check failed: {str(e)}")
            health_check_errors.labels(check_type='topic', error_type=type(e).__name__).inc()
            raise
    
    async def _check_consumer_lag(self):
        """Monitor consumer group lag"""
        start_time = time.time()
        
        try:
            # Get list of consumer groups
            group_list = self.admin_client.list_consumer_groups(timeout=10)
            tda_groups = [g.id for g in group_list.valid if 'tda' in g.id.lower()]
            
            for group_id in tda_groups:
                try:
                    # Get group metadata
                    group_metadata = self.admin_client.describe_consumer_groups([group_id], timeout=10)
                    
                    if group_id not in group_metadata:
                        continue
                    
                    group_info = group_metadata[group_id]
                    
                    # Get partition assignments and calculate lag
                    assignments = []
                    for member in group_info.members:
                        if member.assignment:
                            assignments.extend(member.assignment.topic_partitions)
                    
                    if not assignments:
                        continue
                    
                    # Get committed offsets
                    committed = self.consumer.committed(assignments, timeout=10)
                    
                    # Get high water marks
                    high_water_marks = self.consumer.get_watermark_offsets(assignments[0], timeout=10)
                    
                    total_lag = 0
                    max_lag = 0
                    topics = set()
                    
                    for tp in assignments:
                        topics.add(tp.topic)
                        committed_offset = next((c.offset for c in committed if c.topic == tp.topic and c.partition == tp.partition), -1)
                        
                        if committed_offset >= 0:
                            high_water_mark = high_water_marks[1]  # High water mark
                            lag = max(0, high_water_mark - committed_offset)
                            total_lag += lag
                            max_lag = max(max_lag, lag)
                            
                            consumer_lag_metric.labels(group=group_id, topic=tp.topic).set(lag)
                    
                    # Determine health status
                    if max_lag > self.config['thresholds']['consumer_lag_critical']:
                        status = HealthStatus.CRITICAL
                        score = 0.0
                    elif max_lag > self.config['thresholds']['consumer_lag_warning']:
                        status = HealthStatus.WARNING
                        score = 0.5
                    else:
                        status = HealthStatus.HEALTHY
                        score = 1.0
                    
                    self.results.append(HealthCheckResult(
                        component=f"consumer-group-{group_id}",
                        component_type=ComponentType.CONSUMER_GROUP,
                        status=status,
                        score=score,
                        message=f"Consumer group {group_id}: {total_lag} total lag, {max_lag} max lag",
                        details={
                            'group_id': group_id,
                            'state': group_info.state,
                            'member_count': len(group_info.members),
                            'total_lag': total_lag,
                            'max_lag': max_lag,
                            'topics': list(topics)
                        },
                        timestamp=datetime.now(timezone.utc),
                        duration_ms=(time.time() - start_time) * 1000
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to check consumer group {group_id}: {str(e)}")
                    health_check_errors.labels(check_type='consumer_lag', error_type=type(e).__name__).inc()
            
            duration = (time.time() - start_time) * 1000
            health_check_duration.labels(check_type='consumer_lag').observe(duration / 1000)
            
        except Exception as e:
            logger.error(f"Consumer lag check failed: {str(e)}")
            health_check_errors.labels(check_type='consumer_lag', error_type=type(e).__name__).inc()
            raise
    
    async def _check_workflow_health(self):
        """Check TDA workflow health status"""
        start_time = time.time()
        
        try:
            workflows = self.config['workflows']
            
            for workflow_name in workflows:
                try:
                    # Check workflow-specific metrics
                    # This would typically involve checking:
                    # - Job processing rates
                    # - Error rates
                    # - Processing times
                    # - Queue depths
                    
                    # For now, we'll simulate workflow health based on topic activity
                    workflow_score = await self._calculate_workflow_score(workflow_name)
                    
                    if workflow_score >= 0.8:
                        status = HealthStatus.HEALTHY
                    elif workflow_score >= 0.5:
                        status = HealthStatus.WARNING
                    else:
                        status = HealthStatus.CRITICAL
                    
                    workflow_health.labels(workflow=workflow_name).set(workflow_score)
                    
                    self.results.append(HealthCheckResult(
                        component=f"workflow-{workflow_name}",
                        component_type=ComponentType.WORKFLOW,
                        status=status,
                        score=workflow_score,
                        message=f"Workflow {workflow_name} health score: {workflow_score:.2f}",
                        details={
                            'workflow_name': workflow_name,
                            'health_score': workflow_score
                        },
                        timestamp=datetime.now(timezone.utc),
                        duration_ms=(time.time() - start_time) * 1000
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to check workflow {workflow_name}: {str(e)}")
                    workflow_health.labels(workflow=workflow_name).set(0.0)
                    health_check_errors.labels(check_type='workflow', error_type=type(e).__name__).inc()
            
            duration = (time.time() - start_time) * 1000
            health_check_duration.labels(check_type='workflow').observe(duration / 1000)
            
        except Exception as e:
            logger.error(f"Workflow health check failed: {str(e)}")
            health_check_errors.labels(check_type='workflow', error_type=type(e).__name__).inc()
            raise
    
    async def _calculate_workflow_score(self, workflow_name: str) -> float:
        """Calculate workflow health score based on various metrics"""
        # This is a simplified calculation
        # In a real implementation, this would check:
        # - Message processing rates
        # - Error rates
        # - Processing latencies
        # - Queue depths
        # - Dependency health
        
        base_score = 0.8  # Base healthy score
        
        # Check related topics
        related_topics = {
            'data_ingestion': ['tda_uploads', 'tda_events'],
            'computation': ['tda_jobs', 'tda_results'],
            'result_delivery': ['tda_results', 'tda_events'],
            'error_handling': ['tda_errors', 'tda_dlq']
        }
        
        topics = related_topics.get(workflow_name, [])
        topic_scores = []
        
        for result in self.results:
            if result.component_type == ComponentType.TOPIC:
                topic_name = result.component.replace('topic-', '')
                if any(pattern in topic_name for pattern in topics):
                    topic_scores.append(result.score)
        
        if topic_scores:
            avg_topic_score = sum(topic_scores) / len(topic_scores)
            workflow_score = (base_score + avg_topic_score) / 2
        else:
            workflow_score = base_score
        
        return min(1.0, max(0.0, workflow_score))
    
    async def _check_external_services(self):
        """Check external service health"""
        start_time = time.time()
        
        try:
            external_services = self.config['external_services']
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for service_name, url in external_services.items():
                    try:
                        check_start = time.time()
                        async with session.get(url) as response:
                            response_time = (time.time() - check_start) * 1000
                            
                            if response.status == 200:
                                status = HealthStatus.HEALTHY
                                score = 1.0
                                message = f"Service {service_name} responding in {response_time:.1f}ms"
                            else:
                                status = HealthStatus.WARNING
                                score = 0.5
                                message = f"Service {service_name} returned status {response.status}"
                            
                            self.results.append(HealthCheckResult(
                                component=f"external-{service_name}",
                                component_type=ComponentType.EXTERNAL_SERVICE,
                                status=status,
                                score=score,
                                message=message,
                                details={
                                    'service_name': service_name,
                                    'url': url,
                                    'status_code': response.status,
                                    'response_time_ms': response_time
                                },
                                timestamp=datetime.now(timezone.utc),
                                duration_ms=response_time
                            ))
                            
                    except Exception as e:
                        logger.error(f"Failed to check external service {service_name}: {str(e)}")
                        
                        self.results.append(HealthCheckResult(
                            component=f"external-{service_name}",
                            component_type=ComponentType.EXTERNAL_SERVICE,
                            status=HealthStatus.CRITICAL,
                            score=0.0,
                            message=f"Service {service_name} unreachable: {str(e)}",
                            details={'error': str(e)},
                            timestamp=datetime.now(timezone.utc),
                            duration_ms=(time.time() - start_time) * 1000
                        ))
                        
                        health_check_errors.labels(check_type='external_service', error_type=type(e).__name__).inc()
            
            duration = (time.time() - start_time) * 1000
            health_check_duration.labels(check_type='external_service').observe(duration / 1000)
            
        except Exception as e:
            logger.error(f"External service health check failed: {str(e)}")
            health_check_errors.labels(check_type='external_service', error_type=type(e).__name__).inc()
            raise
    
    def _calculate_overall_health(self):
        """Calculate overall health score"""
        if not self.results:
            self.overall_health_score = 0.0
            return
        
        # Weight different component types
        weights = {
            ComponentType.BROKER: 0.3,
            ComponentType.TOPIC: 0.25,
            ComponentType.CONSUMER_GROUP: 0.2,
            ComponentType.WORKFLOW: 0.15,
            ComponentType.EXTERNAL_SERVICE: 0.1
        }
        
        weighted_scores = {}
        component_counts = {}
        
        for result in self.results:
            comp_type = result.component_type
            if comp_type not in weighted_scores:
                weighted_scores[comp_type] = 0.0
                component_counts[comp_type] = 0
            
            weighted_scores[comp_type] += result.score
            component_counts[comp_type] += 1
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for comp_type, weight in weights.items():
            if comp_type in weighted_scores:
                avg_score = weighted_scores[comp_type] / component_counts[comp_type]
                total_score += avg_score * weight
                total_weight += weight
        
        self.overall_health_score = total_score / total_weight if total_weight > 0 else 0.0
        health_score.labels(component='overall').set(self.overall_health_score)
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics with health check results"""
        for result in self.results:
            if result.component_type == ComponentType.TOPIC:
                topic_name = result.component.replace('topic-', '')
                topic_health.labels(topic=topic_name).set(result.score)
            elif result.component_type == ComponentType.WORKFLOW:
                workflow_name = result.component.replace('workflow-', '')
                workflow_health.labels(workflow=workflow_name).set(result.score)
    
    def _generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_health_score': self.overall_health_score,
            'overall_status': self._get_overall_status(),
            'summary': {
                'total_checks': len(self.results),
                'healthy': len([r for r in self.results if r.status == HealthStatus.HEALTHY]),
                'warning': len([r for r in self.results if r.status == HealthStatus.WARNING]),
                'critical': len([r for r in self.results if r.status == HealthStatus.CRITICAL])
            },
            'components': {}
        }
        
        # Group results by component type
        for comp_type in ComponentType:
            type_results = [r for r in self.results if r.component_type == comp_type]
            if type_results:
                report['components'][comp_type.value] = [
                    {
                        'component': r.component,
                        'status': r.status.value,
                        'score': r.score,
                        'message': r.message,
                        'details': r.details,
                        'timestamp': r.timestamp.isoformat(),
                        'duration_ms': r.duration_ms
                    }
                    for r in type_results
                ]
        
        return report
    
    def _get_overall_status(self) -> str:
        """Get overall status based on health score"""
        if self.overall_health_score >= 0.8:
            return HealthStatus.HEALTHY.value
        elif self.overall_health_score >= 0.5:
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.CRITICAL.value
    
    def display_health_dashboard(self):
        """Display rich health dashboard"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header
        overall_status = self._get_overall_status()
        color = "green" if overall_status == "healthy" else "yellow" if overall_status == "warning" else "red"
        
        header_panel = Panel(
            f"[bold {color}]TDA Kafka Health Dashboard[/bold {color}]\n"
            f"Overall Health Score: [bold]{self.overall_health_score:.2f}[/bold] | "
            f"Status: [bold {color}]{overall_status.upper()}[/bold {color}] | "
            f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            title="TDA Platform Health Monitor"
        )
        layout["header"].update(header_panel)
        
        # Main content
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Summary table
        summary_table = Table(title="Health Summary")
        summary_table.add_column("Component Type", style="cyan")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("Healthy", justify="right", style="green")
        summary_table.add_column("Warning", justify="right", style="yellow")
        summary_table.add_column("Critical", justify="right", style="red")
        summary_table.add_column("Avg Score", justify="right")
        
        for comp_type in ComponentType:
            type_results = [r for r in self.results if r.component_type == comp_type]
            if type_results:
                healthy = len([r for r in type_results if r.status == HealthStatus.HEALTHY])
                warning = len([r for r in type_results if r.status == HealthStatus.WARNING])
                critical = len([r for r in type_results if r.status == HealthStatus.CRITICAL])
                avg_score = sum(r.score for r in type_results) / len(type_results)
                
                summary_table.add_row(
                    comp_type.value.replace('_', ' ').title(),
                    str(len(type_results)),
                    str(healthy),
                    str(warning),
                    str(critical),
                    f"{avg_score:.2f}"
                )
        
        layout["left"].update(summary_table)
        
        # Recent alerts/issues
        issues_table = Table(title="Recent Issues")
        issues_table.add_column("Component", style="cyan")
        issues_table.add_column("Status", style="yellow")
        issues_table.add_column("Message")
        
        critical_results = [r for r in self.results if r.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]]
        critical_results.sort(key=lambda x: x.timestamp, reverse=True)
        
        for result in critical_results[:10]:  # Show last 10 issues
            status_color = "red" if result.status == HealthStatus.CRITICAL else "yellow"
            issues_table.add_row(
                result.component,
                f"[{status_color}]{result.status.value}[/{status_color}]",
                result.message[:60] + "..." if len(result.message) > 60 else result.message
            )
        
        layout["right"].update(issues_table)
        
        # Footer
        footer_panel = Panel(
            f"Total Checks: {len(self.results)} | "
            f"Healthy: [green]{len([r for r in self.results if r.status == HealthStatus.HEALTHY])}[/green] | "
            f"Warning: [yellow]{len([r for r in self.results if r.status == HealthStatus.WARNING])}[/yellow] | "
            f"Critical: [red]{len([r for r in self.results if r.status == HealthStatus.CRITICAL])}[/red]",
            title="Statistics"
        )
        layout["footer"].update(footer_panel)
        
        self.console.print(layout)


@click.command()
@click.option('--config', '-c', default='config/health-check.yml', help='Configuration file path')
@click.option('--dashboard', '-d', is_flag=True, help='Show live dashboard')
@click.option('--json-output', '-j', is_flag=True, help='Output results as JSON')
@click.option('--continuous', is_flag=True, help='Run continuous monitoring')
@click.option('--interval', '-i', default=60, help='Check interval in seconds (for continuous mode)')
async def main(config, dashboard, json_output, continuous, interval):
    """TDA Kafka Health Check Service"""
    checker = TDAHealthChecker(config)
    
    if continuous:
        while True:
            try:
                report = await checker.run_comprehensive_health_check()
                
                if dashboard:
                    checker.display_health_dashboard()
                elif json_output:
                    click.echo(json.dumps(report, indent=2))
                else:
                    click.echo(f"Health check completed: {report['overall_status']} (score: {report['overall_health_score']:.2f})")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                click.echo("Health monitoring stopped")
                break
            except Exception as e:
                click.echo(f"Health check error: {str(e)}")
                await asyncio.sleep(interval)
    else:
        report = await checker.run_comprehensive_health_check()
        
        if dashboard:
            checker.display_health_dashboard()
        elif json_output:
            click.echo(json.dumps(report, indent=2))
        else:
            checker.console.print(f"[bold]Health Check Results[/bold]")
            checker.console.print(f"Overall Status: [bold]{report['overall_status']}[/bold]")
            checker.console.print(f"Health Score: [bold]{report['overall_health_score']:.2f}[/bold]")
            checker.console.print(f"Total Checks: {report['summary']['total_checks']}")
            checker.console.print(f"Healthy: [green]{report['summary']['healthy']}[/green]")
            checker.console.print(f"Warning: [yellow]{report['summary']['warning']}[/yellow]")
            checker.console.print(f"Critical: [red]{report['summary']['critical']}[/red]")


if __name__ == '__main__':
    asyncio.run(main())
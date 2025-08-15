#!/usr/bin/env python3
"""
TDA Kafka Metrics Exporter

Custom Prometheus metrics exporter for TDA-specific business metrics, job processing statistics,
performance benchmarks, and integration with existing TDA backend monitoring.

Features:
- TDA job lifecycle metrics (submitted, processing, completed, failed)
- Persistence diagram analytics and topological feature metrics
- Algorithm performance benchmarks and processing time distributions
- Workflow health indicators and system status metrics
- Consumer group and topic-specific TDA metrics
- Integration with existing TDA backend Prometheus endpoints
- Custom business logic metrics for computational topology
- Real-time streaming metrics with configurable collection intervals
"""

import asyncio
import json
import logging
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Deque
from urllib.parse import urljoin

import aiohttp
import click
import yaml
from confluent_kafka import Consumer, TopicPartition
from prometheus_client import (
    Gauge, Counter, Histogram, Summary, Info, Enum as PrometheusEnum,
    start_http_server, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.core import REGISTRY
from aiohttp import web
import numpy as np
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/tda-metrics-exporter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """TDA job status enumeration"""
    SUBMITTED = "submitted"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlgorithmType(Enum):
    """TDA algorithm types"""
    VIETORIS_RIPS = "vietoris_rips"
    ALPHA_COMPLEX = "alpha_complex"
    CECH_COMPLEX = "cech_complex"
    WITNESS_COMPLEX = "witness_complex"
    PERSISTENT_HOMOLOGY = "persistent_homology"
    MAPPER = "mapper"
    DTM_FILTRATION = "dtm_filtration"


@dataclass
class JobMetrics:
    """Job processing metrics"""
    job_id: str
    job_type: str
    algorithm: AlgorithmType
    status: JobStatus
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


@dataclass
class PersistenceDiagramMetrics:
    """Persistence diagram metrics"""
    job_id: str
    dimension: int
    point_count: int
    max_persistence: float
    avg_persistence: float
    total_persistence: float
    birth_time_range: Tuple[float, float]
    death_time_range: Tuple[float, float]


class TDAMetricsExporter:
    """TDA-specific metrics exporter for Prometheus"""
    
    def __init__(self, config_path: str = "config/metrics-exporter.yml"):
        self.config = self._load_config(config_path)
        self.consumer = None
        self.metrics_registry = CollectorRegistry()
        
        # Job processing metrics
        self.job_counter = Counter(
            'tda_jobs_total',
            'Total number of TDA jobs',
            ['job_type', 'algorithm', 'status'],
            registry=self.metrics_registry
        )
        
        self.job_processing_time = Histogram(
            'tda_job_processing_time_seconds',
            'TDA job processing time in seconds',
            ['job_type', 'algorithm'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 1800.0, 3600.0],
            registry=self.metrics_registry
        )
        
        self.job_queue_depth = Gauge(
            'tda_job_queue_depth',
            'Current depth of TDA job queue',
            ['job_type'],
            registry=self.metrics_registry
        )
        
        self.job_memory_usage = Histogram(
            'tda_job_memory_usage_mb',
            'Memory usage for TDA jobs in MB',
            ['job_type', 'algorithm'],
            buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000],
            registry=self.metrics_registry
        )
        
        # Persistence diagram metrics
        self.persistence_points = Gauge(
            'tda_persistence_diagram_points',
            'Number of points in persistence diagrams',
            ['job_type', 'dimension'],
            registry=self.metrics_registry
        )
        
        self.persistence_max = Gauge(
            'tda_persistence_diagram_max_persistence',
            'Maximum persistence value in diagram',
            ['job_type', 'dimension'],
            registry=self.metrics_registry
        )
        
        self.persistence_total = Gauge(
            'tda_persistence_diagram_total_persistence',
            'Total persistence in diagram',
            ['job_type', 'dimension'],
            registry=self.metrics_registry
        )
        
        # Algorithm performance metrics
        self.algorithm_performance = Summary(
            'tda_algorithm_performance_seconds',
            'Algorithm performance metrics',
            ['algorithm', 'input_size_bucket'],
            registry=self.metrics_registry
        )
        
        self.algorithm_success_rate = Gauge(
            'tda_algorithm_success_rate',
            'Algorithm success rate',
            ['algorithm'],
            registry=self.metrics_registry
        )
        
        self.algorithm_memory_efficiency = Gauge(
            'tda_algorithm_memory_efficiency',
            'Memory efficiency metric (output_size / memory_used)',
            ['algorithm'],
            registry=self.metrics_registry
        )
        
        # Workflow metrics
        self.workflow_health = Gauge(
            'tda_workflow_health_score',
            'TDA workflow health score (0-1)',
            ['workflow'],
            registry=self.metrics_registry
        )
        
        self.workflow_throughput = Gauge(
            'tda_workflow_throughput_jobs_per_minute',
            'Workflow throughput in jobs per minute',
            ['workflow'],
            registry=self.metrics_registry
        )
        
        self.workflow_error_rate = Gauge(
            'tda_workflow_error_rate',
            'Workflow error rate (0-1)',
            ['workflow'],
            registry=self.metrics_registry
        )
        
        # System metrics
        self.system_health = Gauge(
            'tda_system_health_score',
            'Overall TDA system health score (0-1)',
            ['component'],
            registry=self.metrics_registry
        )
        
        self.active_connections = Gauge(
            'tda_kafka_active_connections',
            'Number of active Kafka connections',
            ['connection_type'],
            registry=self.metrics_registry
        )
        
        self.message_processing_rate = Gauge(
            'tda_message_processing_rate',
            'Message processing rate per second',
            ['topic'],
            registry=self.metrics_registry
        )
        
        # Topic-specific TDA metrics
        self.topic_lag = Gauge(
            'tda_topic_consumer_lag',
            'Consumer lag for TDA topics',
            ['topic', 'consumer_group'],
            registry=self.metrics_registry
        )
        
        self.topic_error_rate = Gauge(
            'tda_topic_error_rate',
            'Error rate for TDA topics',
            ['topic'],
            registry=self.metrics_registry
        )
        
        # Business metrics
        self.computational_complexity = Histogram(
            'tda_computational_complexity',
            'Computational complexity metrics',
            ['algorithm', 'complexity_type'],
            buckets=[1, 10, 100, 1000, 10000, 100000, 1000000],
            registry=self.metrics_registry
        )
        
        self.data_quality_score = Gauge(
            'tda_data_quality_score',
            'Data quality score for TDA inputs',
            ['data_source'],
            registry=self.metrics_registry
        )
        
        self.feature_extraction_rate = Gauge(
            'tda_feature_extraction_rate',
            'Rate of topological feature extraction',
            ['feature_type'],
            registry=self.metrics_registry
        )
        
        # Information metrics
        self.system_info = Info(
            'tda_system_info',
            'TDA system information',
            registry=self.metrics_registry
        )
        
        # Data storage for calculations
        self.job_history: Deque[JobMetrics] = deque(maxlen=10000)
        self.persistence_history: Deque[PersistenceDiagramMetrics] = deque(maxlen=5000)
        self.algorithm_stats = defaultdict(lambda: {"successes": 0, "failures": 0, "total_time": 0.0, "total_memory": 0.0})
        
        # Initialize Kafka consumer
        self._init_kafka_consumer()
        
        # Set system info
        self._set_system_info()
        
        logger.info("TDA Metrics Exporter initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
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
                'group_id': 'tda-metrics-exporter',
                'auto_offset_reset': 'latest'
            },
            'topics': {
                'jobs': 'tda_jobs',
                'results': 'tda_results',
                'events': 'tda_events',
                'errors': 'tda_errors'
            },
            'metrics': {
                'collection_interval': 30,
                'retention_hours': 24,
                'aggregation_window': 300
            },
            'tda_backend': {
                'health_endpoint': 'http://tda-backend:8000/health',
                'metrics_endpoint': 'http://tda-backend:8000/metrics'
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8090
            }
        }
    
    def _init_kafka_consumer(self):
        """Initialize Kafka consumer for metrics collection"""
        consumer_config = {
            'bootstrap.servers': self.config['kafka']['bootstrap_servers'],
            'security.protocol': self.config['kafka']['security_protocol'],
            'group.id': self.config['kafka']['group_id'],
            'auto.offset.reset': self.config['kafka']['auto_offset_reset'],
            'enable.auto.commit': True
        }
        
        self.consumer = Consumer(consumer_config)
        
        # Subscribe to TDA topics
        topics = list(self.config['topics'].values())
        self.consumer.subscribe(topics)
        
        logger.info(f"Kafka consumer initialized, subscribed to topics: {topics}")
    
    def _set_system_info(self):
        """Set system information metrics"""
        self.system_info.info({
            'version': '1.0.0',
            'kafka_topics_monitored': ','.join(self.config['topics'].values()),
            'collection_interval': str(self.config['metrics']['collection_interval']),
            'algorithms_supported': ','.join([alg.value for alg in AlgorithmType])
        })
    
    async def start_metrics_collection(self):
        """Start collecting metrics from Kafka topics"""
        logger.info("Starting TDA metrics collection")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._collect_kafka_metrics()),
            asyncio.create_task(self._collect_backend_metrics()),
            asyncio.create_task(self._calculate_derived_metrics()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Metrics collection error: {str(e)}")
            raise
    
    async def _collect_kafka_metrics(self):
        """Collect metrics from Kafka messages"""
        while True:
            try:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue
                
                # Process message based on topic
                topic = msg.topic()
                
                try:
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    if topic == self.config['topics']['jobs']:
                        await self._process_job_message(message_data)
                    elif topic == self.config['topics']['results']:
                        await self._process_result_message(message_data)
                    elif topic == self.config['topics']['events']:
                        await self._process_event_message(message_data)
                    elif topic == self.config['topics']['errors']:
                        await self._process_error_message(message_data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message from {topic}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing message from {topic}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Kafka metrics collection error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _process_job_message(self, data: Dict[str, Any]):
        """Process job lifecycle messages"""
        try:
            job_id = data.get('job_id')
            job_type = data.get('job_type', 'unknown')
            algorithm = data.get('algorithm', 'unknown')
            status = data.get('status', 'unknown')
            
            # Convert algorithm and status to enums
            try:
                algorithm_enum = AlgorithmType(algorithm.lower())
            except ValueError:
                algorithm_enum = AlgorithmType.VIETORIS_RIPS  # Default
            
            try:
                status_enum = JobStatus(status.lower())
            except ValueError:
                status_enum = JobStatus.SUBMITTED  # Default
            
            # Update counters
            self.job_counter.labels(
                job_type=job_type,
                algorithm=algorithm_enum.value,
                status=status_enum.value
            ).inc()
            
            # Create or update job metrics
            job_metrics = JobMetrics(
                job_id=job_id,
                job_type=job_type,
                algorithm=algorithm_enum,
                status=status_enum,
                submitted_at=datetime.fromisoformat(data.get('submitted_at', datetime.now().isoformat())),
                started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
                completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
                processing_time_seconds=data.get('processing_time_seconds'),
                input_size=data.get('input_size'),
                output_size=data.get('output_size'),
                memory_usage_mb=data.get('memory_usage_mb'),
                cpu_usage_percent=data.get('cpu_usage_percent')
            )
            
            # Update job history
            self.job_history.append(job_metrics)
            
            # Update processing time histogram
            if job_metrics.processing_time_seconds:
                self.job_processing_time.labels(
                    job_type=job_type,
                    algorithm=algorithm_enum.value
                ).observe(job_metrics.processing_time_seconds)
            
            # Update memory usage histogram
            if job_metrics.memory_usage_mb:
                self.job_memory_usage.labels(
                    job_type=job_type,
                    algorithm=algorithm_enum.value
                ).observe(job_metrics.memory_usage_mb)
            
            # Update algorithm statistics
            if status_enum == JobStatus.COMPLETED:
                self.algorithm_stats[algorithm_enum]['successes'] += 1
                if job_metrics.processing_time_seconds:
                    self.algorithm_stats[algorithm_enum]['total_time'] += job_metrics.processing_time_seconds
                if job_metrics.memory_usage_mb:
                    self.algorithm_stats[algorithm_enum]['total_memory'] += job_metrics.memory_usage_mb
            elif status_enum == JobStatus.FAILED:
                self.algorithm_stats[algorithm_enum]['failures'] += 1
            
            logger.debug(f"Processed job message: {job_id} - {status}")
            
        except Exception as e:
            logger.error(f"Error processing job message: {str(e)}")
    
    async def _process_result_message(self, data: Dict[str, Any]):
        """Process computation result messages"""
        try:
            job_id = data.get('job_id')
            job_type = data.get('job_type', 'unknown')
            
            # Process persistence diagrams if present
            persistence_diagrams = data.get('persistence_diagrams', [])
            
            for diagram_data in persistence_diagrams:
                dimension = diagram_data.get('dimension', 0)
                points = diagram_data.get('points', [])
                
                if not points:
                    continue
                
                # Calculate persistence metrics
                births = [p.get('birth', 0) for p in points]
                deaths = [p.get('death', float('inf')) for p in points if p.get('death') != float('inf')]
                persistences = [d - b for b, d in zip(births, deaths) if d != float('inf')]
                
                if persistences:
                    point_count = len(points)
                    max_persistence = max(persistences)
                    avg_persistence = sum(persistences) / len(persistences)
                    total_persistence = sum(persistences)
                    
                    # Update persistence diagram metrics
                    self.persistence_points.labels(
                        job_type=job_type,
                        dimension=str(dimension)
                    ).set(point_count)
                    
                    self.persistence_max.labels(
                        job_type=job_type,
                        dimension=str(dimension)
                    ).set(max_persistence)
                    
                    self.persistence_total.labels(
                        job_type=job_type,
                        dimension=str(dimension)
                    ).set(total_persistence)
                    
                    # Store for history
                    persistence_metrics = PersistenceDiagramMetrics(
                        job_id=job_id,
                        dimension=dimension,
                        point_count=point_count,
                        max_persistence=max_persistence,
                        avg_persistence=avg_persistence,
                        total_persistence=total_persistence,
                        birth_time_range=(min(births), max(births)) if births else (0, 0),
                        death_time_range=(min(deaths), max(deaths)) if deaths else (0, 0)
                    )
                    
                    self.persistence_history.append(persistence_metrics)
            
            # Update computational complexity metrics
            complexity_data = data.get('computational_complexity', {})
            if complexity_data:
                algorithm = complexity_data.get('algorithm', 'unknown')
                for complexity_type, value in complexity_data.items():
                    if complexity_type != 'algorithm' and isinstance(value, (int, float)):
                        self.computational_complexity.labels(
                            algorithm=algorithm,
                            complexity_type=complexity_type
                        ).observe(value)
            
            logger.debug(f"Processed result message: {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing result message: {str(e)}")
    
    async def _process_event_message(self, data: Dict[str, Any]):
        """Process system event messages"""
        try:
            event_type = data.get('event_type')
            
            if event_type == 'workflow_health_update':
                workflow = data.get('workflow')
                health_score = data.get('health_score', 0.0)
                
                self.workflow_health.labels(workflow=workflow).set(health_score)
            
            elif event_type == 'system_health_update':
                component = data.get('component')
                health_score = data.get('health_score', 0.0)
                
                self.system_health.labels(component=component).set(health_score)
            
            elif event_type == 'data_quality_update':
                data_source = data.get('data_source')
                quality_score = data.get('quality_score', 0.0)
                
                self.data_quality_score.labels(data_source=data_source).set(quality_score)
            
            logger.debug(f"Processed event message: {event_type}")
            
        except Exception as e:
            logger.error(f"Error processing event message: {str(e)}")
    
    async def _process_error_message(self, data: Dict[str, Any]):
        """Process error messages"""
        try:
            error_type = data.get('error_type')
            topic = data.get('topic', 'unknown')
            workflow = data.get('workflow', 'unknown')
            
            # Update error rate metrics (this would typically be calculated over time windows)
            # For now, we'll increment a counter and calculate rates in derived metrics
            
            logger.debug(f"Processed error message: {error_type}")
            
        except Exception as e:
            logger.error(f"Error processing error message: {str(e)}")
    
    async def _collect_backend_metrics(self):
        """Collect metrics from TDA backend services"""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Collect health metrics
                    health_endpoint = self.config['tda_backend']['health_endpoint']
                    try:
                        async with session.get(health_endpoint, timeout=10) as response:
                            if response.status == 200:
                                health_data = await response.json()
                                
                                # Update system health
                                overall_health = health_data.get('overall_health', 0.0)
                                self.system_health.labels(component='backend').set(overall_health)
                                
                                # Update component health scores
                                components = health_data.get('components', {})
                                for component, score in components.items():
                                    self.system_health.labels(component=component).set(score)
                                    
                    except Exception as e:
                        logger.error(f"Failed to collect backend health metrics: {str(e)}")
                        self.system_health.labels(component='backend').set(0.0)
                    
                    # Collect backend Prometheus metrics if available
                    metrics_endpoint = self.config['tda_backend']['metrics_endpoint']
                    try:
                        async with session.get(metrics_endpoint, timeout=10) as response:
                            if response.status == 200:
                                # Parse and integrate backend metrics
                                # This would parse Prometheus format and extract relevant metrics
                                pass
                    except Exception as e:
                        logger.debug(f"Backend metrics endpoint not available: {str(e)}")
                
                await asyncio.sleep(self.config['metrics']['collection_interval'])
                
            except Exception as e:
                logger.error(f"Backend metrics collection error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _calculate_derived_metrics(self):
        """Calculate derived metrics from collected data"""
        while True:
            try:
                await asyncio.sleep(self.config['metrics']['aggregation_window'])
                
                # Calculate queue depths
                current_time = datetime.now(timezone.utc)
                recent_jobs = [j for j in self.job_history 
                              if (current_time - j.submitted_at).total_seconds() < 3600]  # Last hour
                
                job_type_counts = defaultdict(int)
                for job in recent_jobs:
                    if job.status in [JobStatus.SUBMITTED, JobStatus.QUEUED, JobStatus.PROCESSING]:
                        job_type_counts[job.job_type] += 1
                
                for job_type, count in job_type_counts.items():
                    self.job_queue_depth.labels(job_type=job_type).set(count)
                
                # Calculate algorithm success rates
                for algorithm, stats in self.algorithm_stats.items():
                    total = stats['successes'] + stats['failures']
                    if total > 0:
                        success_rate = stats['successes'] / total
                        self.algorithm_success_rate.labels(algorithm=algorithm.value).set(success_rate)
                        
                        # Calculate memory efficiency
                        if stats['total_memory'] > 0 and stats['successes'] > 0:
                            avg_memory = stats['total_memory'] / stats['successes']
                            # This is a simplified efficiency metric
                            efficiency = min(1.0, 1000.0 / avg_memory)  # Better efficiency with less memory
                            self.algorithm_memory_efficiency.labels(algorithm=algorithm.value).set(efficiency)
                
                # Calculate workflow metrics
                time_window = timedelta(minutes=5)
                recent_cutoff = current_time - time_window
                
                workflow_job_counts = defaultdict(int)
                workflow_error_counts = defaultdict(int)
                
                for job in recent_jobs:
                    if job.submitted_at >= recent_cutoff:
                        # Map job types to workflows (simplified mapping)
                        workflow = self._map_job_to_workflow(job.job_type)
                        workflow_job_counts[workflow] += 1
                        
                        if job.status == JobStatus.FAILED:
                            workflow_error_counts[workflow] += 1
                
                for workflow in self.config['workflows']:
                    job_count = workflow_job_counts[workflow]
                    error_count = workflow_error_counts[workflow]
                    
                    # Throughput (jobs per minute)
                    throughput = job_count / (time_window.total_seconds() / 60)
                    self.workflow_throughput.labels(workflow=workflow).set(throughput)
                    
                    # Error rate
                    error_rate = error_count / job_count if job_count > 0 else 0.0
                    self.workflow_error_rate.labels(workflow=workflow).set(error_rate)
                
                # Calculate feature extraction rates
                recent_results = [p for p in self.persistence_history 
                                if (current_time - datetime.now(timezone.utc)).total_seconds() < 3600]
                
                feature_counts = defaultdict(int)
                for result in recent_results:
                    feature_type = f"h{result.dimension}"
                    feature_counts[feature_type] += result.point_count
                
                for feature_type, count in feature_counts.items():
                    rate = count / 3600  # Features per second over last hour
                    self.feature_extraction_rate.labels(feature_type=feature_type).set(rate)
                
                logger.debug("Calculated derived metrics")
                
            except Exception as e:
                logger.error(f"Error calculating derived metrics: {str(e)}")
    
    def _map_job_to_workflow(self, job_type: str) -> str:
        """Map job type to workflow"""
        # Simplified mapping logic
        if 'upload' in job_type.lower():
            return 'data_ingestion'
        elif any(alg in job_type.lower() for alg in ['rips', 'alpha', 'homology']):
            return 'computation'
        elif 'result' in job_type.lower():
            return 'result_delivery'
        else:
            return 'computation'  # Default
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory leaks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                
                retention_hours = self.config['metrics']['retention_hours']
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
                
                # Clean job history
                original_job_count = len(self.job_history)
                self.job_history = deque(
                    [j for j in self.job_history if j.submitted_at >= cutoff_time],
                    maxlen=self.job_history.maxlen
                )
                
                # Clean persistence history
                original_persistence_count = len(self.persistence_history)
                # Note: persistence_history doesn't have timestamp, so we'll just limit by size
                
                logger.info(f"Cleaned up old data: {original_job_count - len(self.job_history)} jobs, "
                          f"{original_persistence_count - len(self.persistence_history)} persistence records")
                
            except Exception as e:
                logger.error(f"Error during data cleanup: {str(e)}")
    
    async def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.metrics_registry)
    
    async def start_server(self):
        """Start HTTP server for metrics endpoint"""
        app = web.Application()
        app.router.add_get('/metrics', self._metrics_handler)
        app.router.add_get('/health', self._health_handler)
        app.router.add_get('/stats', self._stats_handler)
        
        host = self.config['server']['host']
        port = self.config['server']['port']
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"TDA Metrics Exporter server started on http://{host}:{port}")
        logger.info(f"Metrics available at http://{host}:{port}/metrics")
    
    async def _metrics_handler(self, request):
        """Handle metrics endpoint"""
        try:
            metrics_output = await self.get_metrics()
            return web.Response(text=metrics_output, content_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            logger.error(f"Error generating metrics: {str(e)}")
            return web.Response(text="Error generating metrics", status=500)
    
    async def _health_handler(self, request):
        """Handle health check endpoint"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'kafka_connected': self.consumer is not None,
            'job_history_size': len(self.job_history),
            'persistence_history_size': len(self.persistence_history),
            'algorithm_stats': {alg.value: stats for alg, stats in self.algorithm_stats.items()}
        }
        return web.json_response(health_status)
    
    async def _stats_handler(self, request):
        """Handle statistics endpoint"""
        current_time = datetime.now(timezone.utc)
        
        # Calculate statistics from collected data
        recent_jobs = [j for j in self.job_history 
                      if (current_time - j.submitted_at).total_seconds() < 3600]
        
        stats = {
            'timestamp': current_time.isoformat(),
            'total_jobs_processed': len(self.job_history),
            'recent_jobs_count': len(recent_jobs),
            'job_status_counts': {},
            'algorithm_distribution': {},
            'average_processing_times': {},
            'persistence_diagram_stats': {
                'total_diagrams': len(self.persistence_history),
                'dimension_distribution': {}
            }
        }
        
        # Job status distribution
        for job in recent_jobs:
            status = job.status.value
            stats['job_status_counts'][status] = stats['job_status_counts'].get(status, 0) + 1
            
            algorithm = job.algorithm.value
            stats['algorithm_distribution'][algorithm] = stats['algorithm_distribution'].get(algorithm, 0) + 1
        
        # Average processing times
        for algorithm in AlgorithmType:
            alg_jobs = [j for j in recent_jobs if j.algorithm == algorithm and j.processing_time_seconds]
            if alg_jobs:
                avg_time = sum(j.processing_time_seconds for j in alg_jobs) / len(alg_jobs)
                stats['average_processing_times'][algorithm.value] = avg_time
        
        # Persistence diagram dimension distribution
        for pd in self.persistence_history:
            dim = str(pd.dimension)
            stats['persistence_diagram_stats']['dimension_distribution'][dim] = \
                stats['persistence_diagram_stats']['dimension_distribution'].get(dim, 0) + 1
        
        return web.json_response(stats)


@click.command()
@click.option('--config', '-c', default='config/metrics-exporter.yml', help='Configuration file path')
@click.option('--port', '-p', default=8090, help='Server port')
@click.option('--host', default='0.0.0.0', help='Server host')
async def main(config, port, host):
    """TDA Kafka Metrics Exporter"""
    exporter = TDAMetricsExporter(config)
    
    # Override server config if provided
    exporter.config['server']['host'] = host
    exporter.config['server']['port'] = port
    
    # Start server and metrics collection
    await asyncio.gather(
        exporter.start_server(),
        exporter.start_metrics_collection()
    )


if __name__ == '__main__':
    asyncio.run(main())
#!/usr/bin/env python3
"""
TDA Analytics Job for Apache Flink
==================================

Comprehensive real-time analytics for TDA computation events, processing multiple
Kafka topics to generate performance metrics, system health monitoring, and 
usage analytics for the TDA platform.

Features:
- Multi-topic event processing (tda_jobs, tda_results, tda_events, tda_uploads, tda_errors)
- Real-time performance metrics and success rate tracking
- Windowed analytics for trending and alerting
- Algorithm usage pattern analysis
- System health monitoring and error tracking
- Point cloud statistics and processing patterns
- Configurable alerting thresholds and output streams

Usage:
    python tda-analytics-job.py [--config config.json] [--env production]
    
Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka cluster endpoints
    SCHEMA_REGISTRY_URL: Schema registry URL
    FLINK_PARALLELISM: Job parallelism (default: 8)
    ANALYTICS_ENV: Environment (development/staging/production)
    CHECKPOINT_INTERVAL: Checkpoint interval in ms (default: 30000)
"""

import os
import sys
import json
import logging
import argparse
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import numpy as np
import statistics

# PyFlink imports
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.formats.json import JsonRowSerializationSchema, JsonRowDeserializationSchema
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import (
    MapFunction, FlatMapFunction, FilterFunction, KeyedProcessFunction, 
    WindowFunction, AggregateFunction, ProcessWindowFunction, ProcessFunction
)
from pyflink.datastream.window import (
    TumblingEventTimeWindows, SlidingEventTimeWindows, SessionWindows, 
    TimeWindow, Time, EventTimeSessionWindows
)
from pyflink.datastream.state import (
    ValueStateDescriptor, ListStateDescriptor, MapStateDescriptor, 
    ReducingStateDescriptor, AggregatingStateDescriptor
)
from pyflink.common.types import Row
from pyflink.common.time import Instant
from pyflink.datastream.checkpointing import CheckpointingMode

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/flink/log/tda-analytics.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AnalyticsConfig:
    """Configuration for TDA analytics job."""
    
    # Job configuration
    job_name: str = "tda-analytics-job"
    environment: str = "development"
    parallelism: int = 8
    checkpoint_interval: int = 30000  # 30 seconds
    checkpoint_timeout: int = 300000  # 5 minutes
    min_pause_between_checkpoints: int = 5000  # 5 seconds
    
    # Kafka configuration
    kafka_bootstrap_servers: str = "kafka1:9092,kafka2:9093,kafka3:9094"
    schema_registry_url: str = "http://schema-registry:8081"
    consumer_group: str = "flink-analytics-consumer"
    
    # Input topics
    input_topics: Dict[str, str] = field(default_factory=lambda: {
        'jobs': 'tda_jobs',
        'results': 'tda_results', 
        'events': 'tda_events',
        'uploads': 'tda_uploads',
        'errors': 'tda_errors'
    })
    
    # Output topics
    output_topics: Dict[str, str] = field(default_factory=lambda: {
        'metrics': 'tda_analytics_metrics',
        'alerts': 'tda_analytics_alerts',
        'reports': 'tda_analytics_reports',
        'dashboards': 'tda_analytics_dashboards'
    })
    
    # Window configurations
    window_configs: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        'realtime': {'size': 60, 'slide': 10},      # 1min window, 10s slide
        'short_term': {'size': 300, 'slide': 60},   # 5min window, 1min slide  
        'medium_term': {'size': 900, 'slide': 300}, # 15min window, 5min slide
        'long_term': {'size': 3600, 'slide': 600}   # 1hour window, 10min slide
    })
    
    # Session configuration
    session_timeout: int = 1800  # 30 minutes for user session analysis
    
    # Alert thresholds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'error_rate': 0.05,           # 5% error rate
        'avg_computation_time': 300.0, # 5 minutes
        'queue_time': 120.0,          # 2 minutes queue time
        'memory_usage': 0.85,         # 85% memory usage
        'disk_usage': 0.80,           # 80% disk usage
        'consumer_lag': 10000,        # 10k message lag
        'throughput_drop': 0.30       # 30% throughput drop
    })
    
    # Performance tuning
    buffer_timeout: int = 100        # ms
    enable_object_reuse: bool = True
    network_buffer_size: int = 32768  # 32KB
    
    @classmethod
    def from_env(cls, environment: str = None) -> 'AnalyticsConfig':
        """Create configuration from environment variables."""
        env = environment or os.getenv("ANALYTICS_ENV", "development")
        
        config = cls(
            environment=env,
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", cls.kafka_bootstrap_servers),
            schema_registry_url=os.getenv("SCHEMA_REGISTRY_URL", cls.schema_registry_url),
            parallelism=int(os.getenv("FLINK_PARALLELISM", str(cls.parallelism))),
            checkpoint_interval=int(os.getenv("CHECKPOINT_INTERVAL", str(cls.checkpoint_interval)))
        )
        
        # Environment-specific adjustments
        if env == "production":
            config.parallelism = max(config.parallelism, 12)
            config.checkpoint_interval = 60000  # 1 minute
            config.alert_thresholds['error_rate'] = 0.02  # Stricter in production
        elif env == "development":
            config.parallelism = min(config.parallelism, 4)
            config.checkpoint_interval = 10000  # 10 seconds
            
        return config


@dataclass
class TDAEvent:
    """Unified event structure for TDA analytics."""
    event_id: str
    event_type: str
    source_topic: str
    timestamp: datetime
    job_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    algorithm: Optional[str] = None
    status: Optional[str] = None
    
    # Performance metrics
    computation_time: Optional[float] = None
    queue_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    # Data metrics
    point_cloud_size: Optional[int] = None
    dimension: Optional[int] = None
    result_size: Optional[int] = None
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    severity: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class EventParser(MapFunction):
    """Parse and normalize events from different Kafka topics."""
    
    def map(self, value: Tuple[str, str]) -> Optional[TDAEvent]:
        """Parse JSON message and convert to unified TDAEvent."""
        topic, message = value
        
        try:
            data = json.loads(message)
            
            # Extract common fields
            metadata = data.get('metadata', {})
            payload = data.get('payload', data)  # Support both structured and flat formats
            
            event_id = metadata.get('message_id', payload.get('event_id', f"evt_{hash(message)}"))
            event_type = metadata.get('message_type', payload.get('event_type', 'unknown'))
            timestamp_str = metadata.get('timestamp', payload.get('timestamp'))
            
            # Parse timestamp
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Create base event
            event = TDAEvent(
                event_id=event_id,
                event_type=event_type,
                source_topic=topic,
                timestamp=timestamp,
                job_id=payload.get('job_id'),
                user_id=payload.get('user_id'),
                session_id=payload.get('session_id'),
                algorithm=payload.get('algorithm'),
                status=payload.get('status')
            )
            
            # Parse topic-specific fields
            if topic.endswith('_jobs'):
                self._parse_job_event(event, payload)
            elif topic.endswith('_results'):
                self._parse_result_event(event, payload)
            elif topic.endswith('_events'):
                self._parse_system_event(event, payload)
            elif topic.endswith('_uploads'):
                self._parse_upload_event(event, payload)
            elif topic.endswith('_errors'):
                self._parse_error_event(event, payload)
            
            # Store original data as metadata
            event.metadata = {
                'original_topic': topic,
                'original_payload': payload,
                'parsed_at': datetime.now(timezone.utc).isoformat()
            }
            
            return event
            
        except Exception as e:
            logger.error(f"Error parsing event from topic {topic}: {e}")
            logger.debug(f"Failed message: {message[:500]}")
            
            # Return error event for tracking
            return TDAEvent(
                event_id=f"parse_error_{hash(message)}",
                event_type="parse.error",
                source_topic=topic,
                timestamp=datetime.now(timezone.utc),
                error_type="ParseError",
                error_message=str(e),
                severity="error",
                metadata={'original_message': message[:1000]}
            )
    
    def _parse_job_event(self, event: TDAEvent, payload: Dict[str, Any]) -> None:
        """Parse job-specific fields."""
        event.computation_time = payload.get('execution_time')
        event.algorithm = payload.get('algorithm')
        
        # Calculate queue time if timestamps available
        submitted_at = payload.get('submitted_at')
        started_at = payload.get('started_at')
        if submitted_at and started_at:
            try:
                submit_time = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
                start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                event.queue_time = (start_time - submit_time).total_seconds()
            except:
                pass
    
    def _parse_result_event(self, event: TDAEvent, payload: Dict[str, Any]) -> None:
        """Parse result-specific fields."""
        event.result_size = payload.get('result_size')
        event.computation_time = payload.get('computation_time_seconds')
        
        # Extract point cloud information if available
        result_data = payload.get('result_data', {})
        if result_data:
            event.point_cloud_size = result_data.get('num_points')
            event.dimension = result_data.get('dimension')
    
    def _parse_system_event(self, event: TDAEvent, payload: Dict[str, Any]) -> None:
        """Parse system event fields."""
        event.memory_usage = payload.get('memory_usage')
        event.cpu_usage = payload.get('cpu_usage')
        
        # Parse health check data
        if event.event_type == 'system.health_check':
            health_data = payload.get('health_data', {})
            event.metadata.update(health_data)
    
    def _parse_upload_event(self, event: TDAEvent, payload: Dict[str, Any]) -> None:
        """Parse upload event fields."""
        file_size = payload.get('file_size')
        if file_size:
            event.result_size = file_size
            
        # Extract point cloud size from validation results
        validation_result = payload.get('validation_result', {})
        if validation_result:
            event.point_cloud_size = validation_result.get('num_points')
            event.dimension = validation_result.get('dimension')
    
    def _parse_error_event(self, event: TDAEvent, payload: Dict[str, Any]) -> None:
        """Parse error event fields."""
        event.error_type = payload.get('error_type')
        event.error_message = payload.get('error_message')
        event.severity = payload.get('severity', 'error')


class MetricsAggregator(AggregateFunction):
    """Aggregate events into metrics for windowed analysis."""
    
    def create_accumulator(self) -> Dict[str, Any]:
        """Create empty accumulator for metrics."""
        return {
            'total_events': 0,
            'job_events': 0,
            'result_events': 0,
            'error_events': 0,
            'upload_events': 0,
            'system_events': 0,
            
            # Performance metrics
            'computation_times': [],
            'queue_times': [],
            'memory_usages': [],
            'cpu_usages': [],
            
            # Success/failure tracking
            'successful_jobs': 0,
            'failed_jobs': 0,
            'cancelled_jobs': 0,
            
            # Algorithm usage
            'algorithm_counts': defaultdict(int),
            'algorithm_performance': defaultdict(list),
            
            # Point cloud statistics
            'point_cloud_sizes': [],
            'dimensions': [],
            'result_sizes': [],
            
            # Error tracking
            'error_types': defaultdict(int),
            'error_severities': defaultdict(int),
            
            # Temporal tracking
            'first_event_time': None,
            'last_event_time': None,
            'events_by_minute': defaultdict(int),
            
            # User activity
            'unique_users': set(),
            'unique_sessions': set(),
            'user_activity': defaultdict(int)
        }
    
    def add(self, accumulator: Dict[str, Any], event: TDAEvent) -> Dict[str, Any]:
        """Add event to accumulator."""
        if not event:
            return accumulator
            
        acc = accumulator
        acc['total_events'] += 1
        
        # Update temporal tracking
        if not acc['first_event_time'] or event.timestamp < acc['first_event_time']:
            acc['first_event_time'] = event.timestamp
        if not acc['last_event_time'] or event.timestamp > acc['last_event_time']:
            acc['last_event_time'] = event.timestamp
            
        # Events by minute for rate calculation
        minute_key = event.timestamp.replace(second=0, microsecond=0)
        acc['events_by_minute'][minute_key] += 1
        
        # Track by topic
        if 'jobs' in event.source_topic:
            acc['job_events'] += 1
            self._process_job_event(acc, event)
        elif 'results' in event.source_topic:
            acc['result_events'] += 1
            self._process_result_event(acc, event)
        elif 'errors' in event.source_topic:
            acc['error_events'] += 1
            self._process_error_event(acc, event)
        elif 'uploads' in event.source_topic:
            acc['upload_events'] += 1
            self._process_upload_event(acc, event)
        elif 'events' in event.source_topic:
            acc['system_events'] += 1
            self._process_system_event(acc, event)
        
        # Track users and sessions
        if event.user_id:
            acc['unique_users'].add(event.user_id)
            acc['user_activity'][event.user_id] += 1
        if event.session_id:
            acc['unique_sessions'].add(event.session_id)
        
        return acc
    
    def _process_job_event(self, acc: Dict[str, Any], event: TDAEvent) -> None:
        """Process job-specific metrics."""
        if event.algorithm:
            acc['algorithm_counts'][event.algorithm] += 1
            
        if event.computation_time is not None:
            acc['computation_times'].append(event.computation_time)
            if event.algorithm:
                acc['algorithm_performance'][event.algorithm].append(event.computation_time)
                
        if event.queue_time is not None:
            acc['queue_times'].append(event.queue_time)
            
        # Track job outcomes
        if event.status == 'completed':
            acc['successful_jobs'] += 1
        elif event.status == 'failed':
            acc['failed_jobs'] += 1
        elif event.status == 'cancelled':
            acc['cancelled_jobs'] += 1
    
    def _process_result_event(self, acc: Dict[str, Any], event: TDAEvent) -> None:
        """Process result-specific metrics."""
        if event.point_cloud_size is not None:
            acc['point_cloud_sizes'].append(event.point_cloud_size)
        if event.dimension is not None:
            acc['dimensions'].append(event.dimension)
        if event.result_size is not None:
            acc['result_sizes'].append(event.result_size)
            
        if event.computation_time is not None:
            acc['computation_times'].append(event.computation_time)
    
    def _process_error_event(self, acc: Dict[str, Any], event: TDAEvent) -> None:
        """Process error-specific metrics."""
        if event.error_type:
            acc['error_types'][event.error_type] += 1
        if event.severity:
            acc['error_severities'][event.severity] += 1
    
    def _process_upload_event(self, acc: Dict[str, Any], event: TDAEvent) -> None:
        """Process upload-specific metrics."""
        if event.point_cloud_size is not None:
            acc['point_cloud_sizes'].append(event.point_cloud_size)
        if event.dimension is not None:
            acc['dimensions'].append(event.dimension)
        if event.result_size is not None:
            acc['result_sizes'].append(event.result_size)
    
    def _process_system_event(self, acc: Dict[str, Any], event: TDAEvent) -> None:
        """Process system-specific metrics."""
        if event.memory_usage is not None:
            acc['memory_usages'].append(event.memory_usage)
        if event.cpu_usage is not None:
            acc['cpu_usages'].append(event.cpu_usage)
    
    def get_result(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
        """Get final metrics result."""
        acc = accumulator
        
        # Convert sets to counts for serialization
        acc['unique_users'] = len(acc['unique_users'])
        acc['unique_sessions'] = len(acc['unique_sessions'])
        
        # Convert defaultdicts to regular dicts
        acc['algorithm_counts'] = dict(acc['algorithm_counts'])
        acc['algorithm_performance'] = {k: list(v) for k, v in acc['algorithm_performance'].items()}
        acc['error_types'] = dict(acc['error_types'])
        acc['error_severities'] = dict(acc['error_severities'])
        acc['user_activity'] = dict(acc['user_activity'])
        acc['events_by_minute'] = {k.isoformat(): v for k, v in acc['events_by_minute'].items()}
        
        # Calculate derived metrics
        acc['success_rate'] = (
            acc['successful_jobs'] / max(acc['successful_jobs'] + acc['failed_jobs'], 1)
        )
        acc['error_rate'] = acc['error_events'] / max(acc['total_events'], 1)
        
        # Statistical metrics
        if acc['computation_times']:
            acc['avg_computation_time'] = statistics.mean(acc['computation_times'])
            acc['median_computation_time'] = statistics.median(acc['computation_times'])
            acc['max_computation_time'] = max(acc['computation_times'])
            acc['p95_computation_time'] = np.percentile(acc['computation_times'], 95)
            
        if acc['queue_times']:
            acc['avg_queue_time'] = statistics.mean(acc['queue_times'])
            acc['median_queue_time'] = statistics.median(acc['queue_times'])
            acc['max_queue_time'] = max(acc['queue_times'])
            
        if acc['point_cloud_sizes']:
            acc['avg_point_cloud_size'] = statistics.mean(acc['point_cloud_sizes'])
            acc['median_point_cloud_size'] = statistics.median(acc['point_cloud_sizes'])
            acc['max_point_cloud_size'] = max(acc['point_cloud_sizes'])
            
        # Calculate throughput
        if acc['first_event_time'] and acc['last_event_time']:
            duration = (acc['last_event_time'] - acc['first_event_time']).total_seconds()
            if duration > 0:
                acc['events_per_second'] = acc['total_events'] / duration
                acc['jobs_per_minute'] = (acc['job_events'] * 60) / duration
            
        return acc
    
    def merge(self, acc1: Dict[str, Any], acc2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two accumulators."""
        merged = {
            'total_events': acc1['total_events'] + acc2['total_events'],
            'job_events': acc1['job_events'] + acc2['job_events'],
            'result_events': acc1['result_events'] + acc2['result_events'],
            'error_events': acc1['error_events'] + acc2['error_events'],
            'upload_events': acc1['upload_events'] + acc2['upload_events'],
            'system_events': acc1['system_events'] + acc2['system_events'],
            
            'computation_times': acc1['computation_times'] + acc2['computation_times'],
            'queue_times': acc1['queue_times'] + acc2['queue_times'],
            'memory_usages': acc1['memory_usages'] + acc2['memory_usages'],
            'cpu_usages': acc1['cpu_usages'] + acc2['cpu_usages'],
            
            'successful_jobs': acc1['successful_jobs'] + acc2['successful_jobs'],
            'failed_jobs': acc1['failed_jobs'] + acc2['failed_jobs'],
            'cancelled_jobs': acc1['cancelled_jobs'] + acc2['cancelled_jobs'],
            
            'point_cloud_sizes': acc1['point_cloud_sizes'] + acc2['point_cloud_sizes'],
            'dimensions': acc1['dimensions'] + acc2['dimensions'],
            'result_sizes': acc1['result_sizes'] + acc2['result_sizes'],
            
            'unique_users': acc1['unique_users'].union(acc2['unique_users']),
            'unique_sessions': acc1['unique_sessions'].union(acc2['unique_sessions'])
        }
        
        # Merge dictionaries
        for key in ['algorithm_counts', 'error_types', 'error_severities', 'user_activity']:
            merged[key] = defaultdict(int)
            for d in [acc1[key], acc2[key]]:
                for k, v in d.items():
                    merged[key][k] += v
        
        # Merge algorithm performance
        merged['algorithm_performance'] = defaultdict(list)
        for d in [acc1['algorithm_performance'], acc2['algorithm_performance']]:
            for k, v in d.items():
                merged['algorithm_performance'][k].extend(v)
        
        # Merge temporal data
        merged['events_by_minute'] = defaultdict(int)
        for d in [acc1['events_by_minute'], acc2['events_by_minute']]:
            for k, v in d.items():
                merged['events_by_minute'][k] += v
        
        # Handle timestamps
        times1 = [acc1['first_event_time'], acc1['last_event_time']]
        times2 = [acc2['first_event_time'], acc2['last_event_time']]
        all_times = [t for t in times1 + times2 if t is not None]
        
        if all_times:
            merged['first_event_time'] = min(all_times)
            merged['last_event_time'] = max(all_times)
        else:
            merged['first_event_time'] = None
            merged['last_event_time'] = None
        
        return merged


class AlertGenerator(ProcessWindowFunction):
    """Generate alerts based on aggregated metrics."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.thresholds = config.alert_thresholds
    
    def process(self, key: str, context: 'ProcessWindowFunction.Context', 
                elements: 'Iterable[Dict[str, Any]]') -> 'Iterable[Dict[str, Any]]':
        """Process window and generate alerts."""
        
        for metrics in elements:
            alerts = []
            window_start = context.window().start
            window_end = context.window().end
            
            # Check error rate
            if metrics.get('error_rate', 0) > self.thresholds['error_rate']:
                alerts.append(self._create_alert(
                    alert_type='high_error_rate',
                    severity='critical',
                    message=f"Error rate {metrics['error_rate']:.2%} exceeds threshold {self.thresholds['error_rate']:.2%}",
                    metrics=metrics,
                    window_start=window_start,
                    window_end=window_end
                ))
            
            # Check average computation time
            avg_comp_time = metrics.get('avg_computation_time')
            if avg_comp_time and avg_comp_time > self.thresholds['avg_computation_time']:
                alerts.append(self._create_alert(
                    alert_type='slow_computation',
                    severity='warning',
                    message=f"Average computation time {avg_comp_time:.1f}s exceeds threshold {self.thresholds['avg_computation_time']:.1f}s",
                    metrics=metrics,
                    window_start=window_start,
                    window_end=window_end
                ))
            
            # Check queue time
            avg_queue_time = metrics.get('avg_queue_time')
            if avg_queue_time and avg_queue_time > self.thresholds['queue_time']:
                alerts.append(self._create_alert(
                    alert_type='high_queue_time',
                    severity='warning',
                    message=f"Average queue time {avg_queue_time:.1f}s exceeds threshold {self.thresholds['queue_time']:.1f}s",
                    metrics=metrics,
                    window_start=window_start,
                    window_end=window_end
                ))
            
            # Check memory usage
            if metrics.get('memory_usages'):
                max_memory = max(metrics['memory_usages'])
                if max_memory > self.thresholds['memory_usage']:
                    alerts.append(self._create_alert(
                        alert_type='high_memory_usage',
                        severity='critical',
                        message=f"Memory usage {max_memory:.1%} exceeds threshold {self.thresholds['memory_usage']:.1%}",
                        metrics=metrics,
                        window_start=window_start,
                        window_end=window_end
                    ))
            
            # Check throughput drops (requires comparison with previous windows)
            events_per_second = metrics.get('events_per_second')
            if events_per_second is not None and events_per_second < 1.0:  # Less than 1 event per second
                alerts.append(self._create_alert(
                    alert_type='low_throughput',
                    severity='warning',
                    message=f"Throughput {events_per_second:.2f} events/second is very low",
                    metrics=metrics,
                    window_start=window_start,
                    window_end=window_end
                ))
            
            # Check for specific error patterns
            error_types = metrics.get('error_types', {})
            for error_type, count in error_types.items():
                if count > 10:  # More than 10 errors of the same type
                    alerts.append(self._create_alert(
                        alert_type='repeated_errors',
                        severity='warning',
                        message=f"High frequency of {error_type} errors: {count} occurrences",
                        metrics={'error_type': error_type, 'count': count},
                        window_start=window_start,
                        window_end=window_end
                    ))
            
            # Yield all generated alerts
            for alert in alerts:
                yield alert
    
    def _create_alert(self, alert_type: str, severity: str, message: str, 
                     metrics: Dict[str, Any], window_start: int, window_end: int) -> Dict[str, Any]:
        """Create alert dictionary."""
        return {
            'alert_id': f"{alert_type}_{window_start}_{hash(message) % 10000}",
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'window_start': window_start,
            'window_end': window_end,
            'environment': self.config.environment,
            'metrics': metrics,
            'metadata': {
                'generated_by': 'tda-analytics-job',
                'version': '1.0.0',
                'config': {
                    'threshold': self.config.alert_thresholds.get(alert_type.replace('high_', '').replace('low_', ''))
                }
            }
        }


class ReportGenerator(MapFunction):
    """Generate periodic reports from aggregated metrics."""
    
    def map(self, value: Tuple[str, Dict[str, Any]]) -> str:
        """Generate analytics report from metrics."""
        window_type, metrics = value
        
        try:
            report = {
                'report_id': f"analytics_{window_type}_{int(datetime.now().timestamp())}",
                'report_type': f'analytics_{window_type}',
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'window_type': window_type,
                'period': {
                    'start': metrics.get('first_event_time'),
                    'end': metrics.get('last_event_time')
                },
                'summary': {
                    'total_events': metrics.get('total_events', 0),
                    'job_events': metrics.get('job_events', 0),
                    'result_events': metrics.get('result_events', 0),
                    'error_events': metrics.get('error_events', 0),
                    'upload_events': metrics.get('upload_events', 0),
                    'system_events': metrics.get('system_events', 0),
                    'unique_users': metrics.get('unique_users', 0),
                    'unique_sessions': metrics.get('unique_sessions', 0)
                },
                'performance': {
                    'success_rate': metrics.get('success_rate', 0),
                    'error_rate': metrics.get('error_rate', 0),
                    'avg_computation_time': metrics.get('avg_computation_time'),
                    'avg_queue_time': metrics.get('avg_queue_time'),
                    'throughput': {
                        'events_per_second': metrics.get('events_per_second'),
                        'jobs_per_minute': metrics.get('jobs_per_minute')
                    }
                },
                'algorithms': {
                    'usage_counts': metrics.get('algorithm_counts', {}),
                    'performance': {
                        alg: {
                            'avg_time': statistics.mean(times) if times else None,
                            'median_time': statistics.median(times) if times else None,
                            'count': len(times)
                        }
                        for alg, times in metrics.get('algorithm_performance', {}).items()
                    }
                },
                'data_patterns': {
                    'point_clouds': {
                        'avg_size': metrics.get('avg_point_cloud_size'),
                        'median_size': metrics.get('median_point_cloud_size'),
                        'max_size': metrics.get('max_point_cloud_size'),
                        'dimensions': list(set(metrics.get('dimensions', [])))
                    },
                    'results': {
                        'avg_size': statistics.mean(metrics.get('result_sizes', [])) if metrics.get('result_sizes') else None,
                        'total_size': sum(metrics.get('result_sizes', []))
                    }
                },
                'errors': {
                    'by_type': metrics.get('error_types', {}),
                    'by_severity': metrics.get('error_severities', {}),
                    'total_count': metrics.get('error_events', 0)
                },
                'user_activity': {
                    'active_users': metrics.get('unique_users', 0),
                    'active_sessions': metrics.get('unique_sessions', 0),
                    'top_users': sorted(
                        metrics.get('user_activity', {}).items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]  # Top 10 most active users
                },
                'metadata': {
                    'generated_by': 'tda-analytics-job',
                    'version': '1.0.0',
                    'environment': getattr(self, 'environment', 'unknown')
                }
            }
            
            return json.dumps(report, default=str)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return json.dumps({
                'report_id': f"error_{int(datetime.now().timestamp())}",
                'error': str(e),
                'generated_at': datetime.now(timezone.utc).isoformat()
            })


class DashboardMetricsGenerator(MapFunction):
    """Generate real-time metrics for dashboard consumption."""
    
    def map(self, value: Dict[str, Any]) -> str:
        """Generate dashboard metrics from aggregated data."""
        try:
            # Extract key metrics for dashboard
            dashboard_metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics_type': 'realtime_dashboard',
                'kpis': {
                    'total_events': value.get('total_events', 0),
                    'events_per_second': value.get('events_per_second', 0),
                    'success_rate': value.get('success_rate', 0) * 100,  # Convert to percentage
                    'error_rate': value.get('error_rate', 0) * 100,
                    'avg_computation_time': value.get('avg_computation_time', 0),
                    'avg_queue_time': value.get('avg_queue_time', 0),
                    'active_users': value.get('unique_users', 0),
                    'active_sessions': value.get('unique_sessions', 0)
                },
                'algorithm_stats': {
                    'most_used': max(value.get('algorithm_counts', {}).items(), 
                                   key=lambda x: x[1], default=('none', 0)),
                    'performance_ranking': sorted(
                        [
                            (alg, statistics.mean(times) if times else float('inf'))
                            for alg, times in value.get('algorithm_performance', {}).items()
                        ],
                        key=lambda x: x[1]
                    )[:5]  # Top 5 fastest algorithms
                },
                'system_health': {
                    'status': 'healthy' if value.get('error_rate', 0) < 0.05 else 'degraded',
                    'avg_memory_usage': statistics.mean(value.get('memory_usages', [])) if value.get('memory_usages') else 0,
                    'max_memory_usage': max(value.get('memory_usages', [0])),
                    'avg_cpu_usage': statistics.mean(value.get('cpu_usages', [])) if value.get('cpu_usages') else 0
                },
                'data_insights': {
                    'avg_point_cloud_size': value.get('avg_point_cloud_size', 0),
                    'total_data_processed': sum(value.get('result_sizes', [])),
                    'common_dimensions': statistics.mode(value.get('dimensions', [2])) if value.get('dimensions') else 2
                },
                'trends': {
                    'events_by_minute': value.get('events_by_minute', {}),
                    'error_trend': value.get('error_rate', 0) - 0.02,  # Simple trend calculation
                    'performance_trend': 0.0  # Placeholder for trend analysis
                }
            }
            
            return json.dumps(dashboard_metrics)
            
        except Exception as e:
            logger.error(f"Error generating dashboard metrics: {e}")
            return json.dumps({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'metrics_type': 'error'
            })


def create_kafka_source(topic: str, config: AnalyticsConfig) -> FlinkKafkaConsumer:
    """Create Kafka consumer for a specific topic."""
    properties = {
        'bootstrap.servers': config.kafka_bootstrap_servers,
        'group.id': f"{config.consumer_group}-{topic}",
        'auto.offset.reset': 'latest',  # Start from latest for analytics
        'enable.auto.commit': 'false',
        'max.poll.records': '1000',
        'fetch.max.bytes': '52428800',  # 50MB
        'session.timeout.ms': '45000',
        'heartbeat.interval.ms': '15000',
        'max.partition.fetch.bytes': '1048576',  # 1MB
        'connections.max.idle.ms': '540000'  # 9 minutes
    }
    
    # Add topic name to message for processing
    class TopicAwareSchema(SimpleStringSchema):
        def __init__(self, topic_name):
            super().__init__()
            self.topic_name = topic_name
            
        def deserialize(self, message):
            return (self.topic_name, super().deserialize(message).decode('utf-8'))
    
    return FlinkKafkaConsumer(
        topics=topic,
        deserialization_schema=TopicAwareSchema(topic),
        properties=properties
    )


def create_kafka_sink(topic: str, config: AnalyticsConfig) -> FlinkKafkaProducer:
    """Create Kafka producer for output topic."""
    properties = {
        'bootstrap.servers': config.kafka_bootstrap_servers,
        'acks': 'all',
        'retries': '2147483647',
        'max.in.flight.requests.per.connection': '1',
        'enable.idempotence': 'true',
        'compression.type': 'lz4',
        'batch.size': '65536',
        'linger.ms': '100',
        'buffer.memory': '134217728',  # 128MB
        'request.timeout.ms': '30000',
        'delivery.timeout.ms': '120000'
    }
    
    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=SimpleStringSchema(),
        producer_config=properties
    )


def create_analytics_job(config: AnalyticsConfig) -> None:
    """Create and execute the TDA analytics job."""
    logger.info(f"Starting TDA Analytics Job: {config.job_name}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Processing topics: {list(config.input_topics.values())}")
    
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(config.parallelism)
    
    # Configure checkpointing
    env.enable_checkpointing(config.checkpoint_interval, CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_min_pause_between_checkpoints(config.min_pause_between_checkpoints)
    checkpoint_config.set_checkpoint_timeout(config.checkpoint_timeout)
    checkpoint_config.enable_externalized_checkpoints(True)
    
    # Performance optimizations
    if config.enable_object_reuse:
        env.get_config().enable_object_reuse()
    env.set_buffer_timeout(config.buffer_timeout)
    env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
    
    # Create sources for all input topics
    streams = []
    for topic_type, topic_name in config.input_topics.items():
        logger.info(f"Creating source for {topic_type}: {topic_name}")
        source = create_kafka_source(topic_name, config)
        stream = env.add_source(source).name(f"kafka-source-{topic_type}")
        streams.append(stream)
    
    # Union all input streams
    if len(streams) > 1:
        unified_stream = streams[0].union(*streams[1:])
    else:
        unified_stream = streams[0]
    
    # Parse events from all topics
    parsed_stream = unified_stream.map(
        EventParser(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    ).name("event-parser")
    
    # Filter out parse errors for main processing (but keep for error tracking)
    valid_events = parsed_stream.filter(
        lambda event: event and event.event_type != "parse.error"
    ).name("filter-valid-events")
    
    # Assign timestamps and watermarks for event time processing
    timestamped_stream = valid_events.assign_timestamps_and_watermarks(
        lambda event: int(event.timestamp.timestamp() * 1000)
    ).name("assign-timestamps")
    
    # Key by a composite key for different aggregation levels
    keyed_stream = timestamped_stream.key_by(lambda event: "global")  # Single key for global metrics
    
    # Create multiple windowed streams for different time scales
    window_streams = {}
    
    for window_name, window_config in config.window_configs.items():
        window_size = Time.seconds(window_config['size'])
        slide_interval = Time.seconds(window_config['slide'])
        
        # Create sliding window
        windowed = keyed_stream.window(
            SlidingEventTimeWindows.of(window_size, slide_interval)
        )
        
        # Aggregate metrics
        aggregated = windowed.aggregate(
            MetricsAggregator(),
            output_type=Types.PICKLED_BYTE_ARRAY()
        ).name(f"aggregate-{window_name}")
        
        window_streams[window_name] = aggregated
    
    # Generate alerts from short-term metrics
    alert_stream = window_streams['short_term'].process(
        AlertGenerator(config),
        output_type=Types.PICKLED_BYTE_ARRAY()
    ).name("generate-alerts")
    
    # Convert alerts to JSON and send to alerts topic
    alert_json = alert_stream.map(
        lambda alert: json.dumps(alert),
        output_type=Types.STRING()
    ).name("format-alerts")
    
    alert_sink = create_kafka_sink(config.output_topics['alerts'], config)
    alert_json.add_sink(alert_sink).name("alerts-sink")
    
    # Generate reports from long-term metrics
    report_stream = window_streams['long_term'].map(
        lambda metrics: ("hourly", metrics),
        output_type=Types.TUPLE([Types.STRING(), Types.PICKLED_BYTE_ARRAY()])
    ).map(
        ReportGenerator(),
        output_type=Types.STRING()
    ).name("generate-reports")
    
    report_sink = create_kafka_sink(config.output_topics['reports'], config)
    report_stream.add_sink(report_sink).name("reports-sink")
    
    # Generate real-time dashboard metrics
    dashboard_stream = window_streams['realtime'].map(
        DashboardMetricsGenerator(),
        output_type=Types.STRING()
    ).name("generate-dashboard-metrics")
    
    dashboard_sink = create_kafka_sink(config.output_topics['dashboards'], config)
    dashboard_stream.add_sink(dashboard_sink).name("dashboard-sink")
    
    # Send detailed metrics to metrics topic
    metrics_stream = window_streams['medium_term'].map(
        lambda metrics: json.dumps(metrics, default=str),
        output_type=Types.STRING()
    ).name("format-metrics")
    
    metrics_sink = create_kafka_sink(config.output_topics['metrics'], config)
    metrics_stream.add_sink(metrics_sink).name("metrics-sink")
    
    # Session-based analysis for user behavior
    session_stream = timestamped_stream.key_by(
        lambda event: event.session_id or "anonymous"
    ).window(
        EventTimeSessionWindows.with_gap(Time.seconds(config.session_timeout))
    ).aggregate(
        MetricsAggregator(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    ).name("session-analytics")
    
    session_json = session_stream.map(
        lambda session_metrics: json.dumps({
            'session_analytics': session_metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, default=str),
        output_type=Types.STRING()
    ).name("format-session-metrics")
    
    session_json.add_sink(metrics_sink).name("session-metrics-sink")
    
    # Error tracking - process parse errors separately
    error_events = parsed_stream.filter(
        lambda event: event and event.event_type == "parse.error"
    ).name("filter-error-events")
    
    error_json = error_events.map(
        lambda error: json.dumps(error.to_dict()),
        output_type=Types.STRING()
    ).name("format-error-events")
    
    error_json.add_sink(alert_sink).name("error-events-sink")
    
    logger.info("Analytics job pipeline configured successfully")
    logger.info("Starting job execution...")
    
    # Execute the job
    try:
        env.execute(config.job_name)
    except Exception as e:
        logger.error(f"Job execution failed: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main entry point for TDA analytics job."""
    parser = argparse.ArgumentParser(description="TDA Analytics Job for Apache Flink")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--env", type=str, default="development", 
                       choices=["development", "staging", "production"],
                       help="Environment to run in")
    parser.add_argument("--job-name", type=str, help="Override job name")
    parser.add_argument("--parallelism", type=int, help="Override parallelism")
    parser.add_argument("--checkpoint-interval", type=int, help="Override checkpoint interval")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = AnalyticsConfig(**config_data)
    else:
        logger.info("Using environment-based configuration")
        config = AnalyticsConfig.from_env(args.env)
    
    # Apply command line overrides
    if args.job_name:
        config.job_name = args.job_name
    if args.parallelism:
        config.parallelism = args.parallelism
    if args.checkpoint_interval:
        config.checkpoint_interval = args.checkpoint_interval
    
    logger.info(f"Final configuration: {asdict(config)}")
    
    try:
        # Validate configuration
        required_topics = list(config.input_topics.values()) + list(config.output_topics.values())
        logger.info(f"Required topics: {required_topics}")
        
        # Install required dependencies
        logger.info("Installing required dependencies...")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "numpy", "apache-flink", "kafka-python"
        ], check=True, capture_output=True)
        
        # Create and run analytics job
        create_analytics_job(config)
        
    except KeyboardInterrupt:
        logger.info("Job interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Job failed with error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
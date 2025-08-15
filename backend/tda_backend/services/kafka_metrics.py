"""
Kafka Metrics and Monitoring Service for TDA Backend.

This module provides comprehensive metrics collection, monitoring, and alerting
for Kafka operations including producer/consumer metrics, message throughput,
and error tracking.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client.registry import REGISTRY

from ..config import settings
from .kafka_producer import get_kafka_producer, MessageType
from .kafka_schemas import get_message_validator


logger = logging.getLogger(__name__)


# Prometheus Metrics
KAFKA_MESSAGES_PRODUCED_TOTAL = Counter(
    'tda_kafka_messages_produced_total',
    'Total number of messages produced to Kafka',
    ['topic', 'message_type', 'status']
)

KAFKA_MESSAGES_CONSUMED_TOTAL = Counter(
    'tda_kafka_messages_consumed_total',
    'Total number of messages consumed from Kafka',
    ['topic', 'message_type', 'status']
)

KAFKA_MESSAGE_SIZE_BYTES = Histogram(
    'tda_kafka_message_size_bytes',
    'Size of messages in bytes',
    ['topic', 'message_type'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

KAFKA_PRODUCER_LATENCY_SECONDS = Histogram(
    'tda_kafka_producer_latency_seconds',
    'Time taken to send messages to Kafka',
    ['topic', 'message_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

KAFKA_CONSUMER_PROCESSING_TIME_SECONDS = Histogram(
    'tda_kafka_consumer_processing_time_seconds',
    'Time taken to process consumed messages',
    ['topic', 'message_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

KAFKA_CONNECTION_STATUS = Gauge(
    'tda_kafka_connection_status',
    'Status of Kafka connection (1=healthy, 0=unhealthy)',
    ['component']  # producer, consumer, schema_registry
)

KAFKA_SCHEMA_VALIDATION_ERRORS_TOTAL = Counter(
    'tda_kafka_schema_validation_errors_total',
    'Total number of schema validation errors',
    ['message_type']
)

KAFKA_CLUSTER_INFO = Info(
    'tda_kafka_cluster_info',
    'Information about the Kafka cluster'
)

# Business Logic Metrics
TDA_JOB_EVENTS_TOTAL = Counter(
    'tda_job_events_total',
    'Total number of TDA job events',
    ['event_type']  # submitted, started, completed, failed
)

TDA_FILE_OPERATIONS_TOTAL = Counter(
    'tda_file_operations_total',
    'Total number of file operations',
    ['operation_type']  # uploaded, validated, processed
)

TDA_COMPUTATION_RESULTS_TOTAL = Counter(
    'tda_computation_results_total',
    'Total number of computation results',
    ['result_type']  # persistence_diagram, betti_numbers, etc.
)


@dataclass
class MetricsSummary:
    """Summary of Kafka metrics over a time period."""
    time_window: timedelta
    messages_produced: int = 0
    messages_consumed: int = 0
    bytes_produced: int = 0
    bytes_consumed: int = 0
    avg_producer_latency: float = 0.0
    avg_consumer_processing_time: float = 0.0
    error_count: int = 0
    success_rate: float = 0.0
    topic_breakdown: Dict[str, int] = field(default_factory=dict)
    message_type_breakdown: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class KafkaMetricsCollector:
    """Collector for Kafka metrics and monitoring."""
    
    def __init__(self):
        self.producer = get_kafka_producer()
        self.validator = get_message_validator()
        self._is_running = False
        self._collection_task: Optional[asyncio.Task] = None
        self.collection_interval = 30.0  # seconds
        
        # In-memory metrics store
        self._metrics_history: List[MetricsSummary] = []
        self._max_history_size = 1440  # 24 hours at 1-minute intervals
        
        # Alert thresholds
        self.error_rate_threshold = 0.05  # 5%
        self.latency_threshold = 1.0  # 1 second
        self.throughput_threshold = 1000  # messages per minute
    
    async def start_collection(self):
        """Start metrics collection."""
        if self._is_running:
            logger.warning("Metrics collection is already running")
            return
        
        self._is_running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started Kafka metrics collection")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped Kafka metrics collection")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        logger.info("Starting metrics collection loop")
        
        try:
            while self._is_running:
                try:
                    await self._collect_metrics()
                    await asyncio.sleep(self.collection_interval)
                    
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                    await asyncio.sleep(5)  # Brief pause before retrying
                    
        except asyncio.CancelledError:
            logger.info("Metrics collection loop cancelled")
            raise
    
    async def _collect_metrics(self):
        """Collect current metrics snapshot."""
        try:
            # Get producer metrics
            producer_metrics = self.producer.get_metrics()
            
            # Update Prometheus metrics
            self._update_connection_status()
            
            # Create metrics summary
            summary = MetricsSummary(
                time_window=timedelta(seconds=self.collection_interval),
                messages_produced=producer_metrics.get('messages_sent', 0),
                messages_consumed=0,  # TODO: Add consumer metrics when available
                bytes_produced=producer_metrics.get('bytes_sent', 0),
                avg_producer_latency=producer_metrics.get('average_send_duration', 0.0),
                error_count=producer_metrics.get('messages_failed', 0),
                success_rate=producer_metrics.get('success_rate', 0.0)
            )
            
            # Store in history
            self._metrics_history.append(summary)
            
            # Trim history to max size
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history = self._metrics_history[-self._max_history_size:]
            
            # Check for alerts
            await self._check_alerts(summary)
            
            logger.debug(f"Collected metrics: {summary}")
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    def _update_connection_status(self):
        """Update connection status metrics."""
        try:
            # Producer status
            producer_healthy = self.producer._is_started and self.producer.producer is not None
            KAFKA_CONNECTION_STATUS.labels(component='producer').set(1 if producer_healthy else 0)
            
            # Schema Registry status (simple check)
            # This would be more sophisticated in production
            KAFKA_CONNECTION_STATUS.labels(component='schema_registry').set(1)
            
        except Exception as e:
            logger.error(f"Failed to update connection status: {e}")
    
    async def _check_alerts(self, summary: MetricsSummary):
        """Check metrics against alert thresholds."""
        alerts = []
        
        # Error rate alert
        if summary.error_count > 0 and summary.success_rate < (1 - self.error_rate_threshold):
            alerts.append({
                'type': 'high_error_rate',
                'message': f"High error rate: {(1-summary.success_rate)*100:.1f}%",
                'severity': 'warning' if summary.success_rate > 0.9 else 'critical'
            })
        
        # Latency alert
        if summary.avg_producer_latency > self.latency_threshold:
            alerts.append({
                'type': 'high_latency',
                'message': f"High producer latency: {summary.avg_producer_latency:.3f}s",
                'severity': 'warning' if summary.avg_producer_latency < 2.0 else 'critical'
            })
        
        # Log alerts
        for alert in alerts:
            severity = alert['severity']
            message = alert['message']
            
            if severity == 'critical':
                logger.error(f"CRITICAL ALERT: {message}")
            else:
                logger.warning(f"WARNING ALERT: {message}")
    
    def record_message_produced(
        self,
        topic: str,
        message_type: MessageType,
        size_bytes: int,
        latency_seconds: float,
        success: bool
    ):
        """Record metrics for a produced message."""
        status = 'success' if success else 'error'
        
        KAFKA_MESSAGES_PRODUCED_TOTAL.labels(
            topic=topic,
            message_type=message_type.value,
            status=status
        ).inc()
        
        if success:
            KAFKA_MESSAGE_SIZE_BYTES.labels(
                topic=topic,
                message_type=message_type.value
            ).observe(size_bytes)
            
            KAFKA_PRODUCER_LATENCY_SECONDS.labels(
                topic=topic,
                message_type=message_type.value
            ).observe(latency_seconds)
        
        # Business logic metrics
        if message_type in [MessageType.JOB_SUBMITTED, MessageType.JOB_STARTED, 
                           MessageType.JOB_COMPLETED, MessageType.JOB_FAILED]:
            event_type = message_type.value.split('.')[1]  # Extract event type
            TDA_JOB_EVENTS_TOTAL.labels(event_type=event_type).inc()
        
        elif message_type in [MessageType.FILE_UPLOADED, MessageType.FILE_VALIDATED, 
                             MessageType.FILE_PROCESSED]:
            operation_type = message_type.value.split('.')[1]  # Extract operation type
            TDA_FILE_OPERATIONS_TOTAL.labels(operation_type=operation_type).inc()
        
        elif message_type in [MessageType.RESULT_GENERATED, MessageType.RESULT_CACHED]:
            result_type = 'computation_result'
            TDA_COMPUTATION_RESULTS_TOTAL.labels(result_type=result_type).inc()
    
    def record_message_consumed(
        self,
        topic: str,
        message_type: MessageType,
        processing_time_seconds: float,
        success: bool
    ):
        """Record metrics for a consumed message."""
        status = 'success' if success else 'error'
        
        KAFKA_MESSAGES_CONSUMED_TOTAL.labels(
            topic=topic,
            message_type=message_type.value,
            status=status
        ).inc()
        
        if success:
            KAFKA_CONSUMER_PROCESSING_TIME_SECONDS.labels(
                topic=topic,
                message_type=message_type.value
            ).observe(processing_time_seconds)
    
    def record_schema_validation_error(self, message_type: MessageType):
        """Record schema validation error."""
        KAFKA_SCHEMA_VALIDATION_ERRORS_TOTAL.labels(
            message_type=message_type.value
        ).inc()
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get Kafka cluster information."""
        try:
            if not self.producer._is_started:
                return {"status": "disconnected"}
            
            # Get cluster metadata
            metadata = await self.producer.producer.client.fetch_metadata()
            
            cluster_info = {
                "broker_count": len(metadata.brokers),
                "topic_count": len(metadata.topics),
                "brokers": [
                    {
                        "id": broker.nodeId,
                        "host": broker.host,
                        "port": broker.port
                    }
                    for broker in metadata.brokers
                ],
                "topics": list(metadata.topics.keys())
            }
            
            # Update Prometheus info metric
            KAFKA_CLUSTER_INFO.info({
                'broker_count': str(cluster_info['broker_count']),
                'topic_count': str(cluster_info['topic_count']),
                'brokers': ','.join([f"{b['host']}:{b['port']}" for b in cluster_info['brokers']])
            })
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {"error": str(e)}
    
    def get_metrics_summary(self, time_window: timedelta = None) -> MetricsSummary:
        """Get metrics summary for a time window."""
        if not self._metrics_history:
            return MetricsSummary(time_window=time_window or timedelta(minutes=5))
        
        # Default to last 5 minutes
        window = time_window or timedelta(minutes=5)
        cutoff_time = datetime.now(timezone.utc) - window
        
        # Filter metrics within time window
        recent_metrics = [
            m for m in self._metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return MetricsSummary(time_window=window)
        
        # Aggregate metrics
        total_produced = sum(m.messages_produced for m in recent_metrics)
        total_consumed = sum(m.messages_consumed for m in recent_metrics)
        total_bytes_produced = sum(m.bytes_produced for m in recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)
        
        # Calculate averages
        avg_latency = sum(m.avg_producer_latency for m in recent_metrics) / len(recent_metrics)
        success_rate = (total_produced - total_errors) / total_produced if total_produced > 0 else 1.0
        
        return MetricsSummary(
            time_window=window,
            messages_produced=total_produced,
            messages_consumed=total_consumed,
            bytes_produced=total_bytes_produced,
            avg_producer_latency=avg_latency,
            error_count=total_errors,
            success_rate=success_rate
        )
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        try:
            producer_metrics = self.producer.get_metrics()
            cluster_info = asyncio.create_task(self.get_cluster_info())
            
            recent_summary = self.get_metrics_summary(timedelta(minutes=5))
            hourly_summary = self.get_metrics_summary(timedelta(hours=1))
            
            return {
                "producer": producer_metrics,
                "cluster": cluster_info,
                "recent_summary": recent_summary,
                "hourly_summary": hourly_summary,
                "collection_status": {
                    "is_running": self._is_running,
                    "collection_interval": self.collection_interval,
                    "history_size": len(self._metrics_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get all metrics: {e}")
            return {"error": str(e)}


# Global metrics collector instance
_metrics_collector: Optional[KafkaMetricsCollector] = None


def get_metrics_collector() -> KafkaMetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = KafkaMetricsCollector()
    return _metrics_collector


async def start_metrics_collection():
    """Start global metrics collection."""
    collector = get_metrics_collector()
    await collector.start_collection()


async def stop_metrics_collection():
    """Stop global metrics collection."""
    global _metrics_collector
    if _metrics_collector:
        await _metrics_collector.stop_collection()
        _metrics_collector = None


# Decorator for automatic metrics collection
def kafka_metrics(message_type: MessageType = None):
    """
    Decorator to automatically collect metrics for Kafka operations.
    
    Usage:
        @kafka_metrics(MessageType.JOB_SUBMITTED)
        async def send_job_message(...):
            # Your message sending logic
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract metrics information from result or context
                # This is a simplified example - real implementation would 
                # extract actual values from the function context
                if message_type:
                    latency = time.time() - start_time
                    collector.record_message_produced(
                        topic="unknown",  # Would extract from context
                        message_type=message_type,
                        size_bytes=0,  # Would extract from message
                        latency_seconds=latency,
                        success=True
                    )
                
                return result
                
            except Exception as e:
                if message_type:
                    latency = time.time() - start_time
                    collector.record_message_produced(
                        topic="unknown",
                        message_type=message_type,
                        size_bytes=0,
                        latency_seconds=latency,
                        success=False
                    )
                raise
        
        return wrapper
    return decorator

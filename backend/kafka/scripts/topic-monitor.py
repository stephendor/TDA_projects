#!/usr/bin/env python3
"""
TDA Kafka Topic Monitor

Advanced topic monitoring and analytics tool that provides comprehensive
insights into topic performance, health, and usage patterns.

Features:
- Monitor topic lag and throughput in real-time
- Generate detailed topic usage reports
- Alert on configuration drift and anomalies
- Performance metrics collection and analysis
- Consumer group monitoring
- Data retention and cleanup tracking
- Automated health scoring
- Export metrics to various formats

Usage:
    python topic-monitor.py monitor --topics tda_jobs,tda_results
    python topic-monitor.py report --format json --output report.json
    python topic-monitor.py alerts --check-interval 60
    python topic-monitor.py dashboard --port 8080
"""

import asyncio
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import yaml
from confluent_kafka import Consumer, KafkaError, TopicPartition
from confluent_kafka.admin import AdminClient, ConfigResource
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("topic-monitor.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PartitionMetrics:
    """Metrics for a single partition."""
    
    partition_id: int
    high_watermark: int = 0
    low_watermark: int = 0
    message_count: int = 0
    size_bytes: int = 0
    leader: Optional[int] = None
    replicas: List[int] = field(default_factory=list)
    in_sync_replicas: List[int] = field(default_factory=list)
    lag: Dict[str, int] = field(default_factory=dict)  # consumer_group -> lag
    

@dataclass
class TopicMetrics:
    """Comprehensive metrics for a topic."""
    
    name: str
    partitions: Dict[int, PartitionMetrics] = field(default_factory=dict)
    total_messages: int = 0
    total_size_bytes: int = 0
    message_rate: float = 0.0  # messages per second
    byte_rate: float = 0.0  # bytes per second
    consumer_groups: Set[str] = field(default_factory=set)
    total_lag: int = 0
    max_lag: int = 0
    health_score: float = 100.0
    alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    

@dataclass
class ConsumerGroupMetrics:
    """Metrics for a consumer group."""
    
    group_id: str
    state: str = "unknown"
    members: int = 0
    topics: Set[str] = field(default_factory=set)
    total_lag: int = 0
    max_lag: int = 0
    partition_assignments: Dict[str, List[int]] = field(default_factory=dict)
    last_commit_time: Optional[datetime] = None
    

@dataclass
class ClusterMetrics:
    """Overall cluster metrics."""
    
    broker_count: int = 0
    topic_count: int = 0
    partition_count: int = 0
    total_messages: int = 0
    total_size_bytes: int = 0
    under_replicated_partitions: int = 0
    offline_partitions: int = 0
    active_consumer_groups: int = 0
    total_consumer_lag: int = 0
    

@dataclass
class Alert:
    """Alert information."""
    
    severity: str  # critical, warning, info
    topic: Optional[str]
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    

class TopicMonitor:
    """Advanced Kafka topic monitoring system."""
    
    def __init__(
        self,
        kafka_config: Optional[Dict[str, str]] = None,
        monitoring_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the topic monitor."""
        self.kafka_config = kafka_config or self._get_default_kafka_config()
        self.monitoring_config = monitoring_config or self._get_default_monitoring_config()
        
        # Initialize clients
        self.admin_client = AdminClient(self.kafka_config)
        self.consumer = Consumer({
            **self.kafka_config,
            "group.id": "topic-monitor",
            "enable.auto.commit": False,
            "auto.offset.reset": "latest"
        })
        
        # Metrics storage
        self.topic_metrics: Dict[str, TopicMetrics] = {}
        self.consumer_group_metrics: Dict[str, ConsumerGroupMetrics] = {}
        self.cluster_metrics = ClusterMetrics()
        self.alerts: List[Alert] = []
        
        # Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Historical data for rate calculations
        self.metric_history: Dict[str, List[Tuple[datetime, Any]]] = {}
        
    def _get_default_kafka_config(self) -> Dict[str, str]:
        """Get default Kafka configuration."""
        return {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "client.id": "tda-topic-monitor",
            "request.timeout.ms": "30000",
            "api.version.request": "true",
        }
    
    def _get_default_monitoring_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "collection_interval": 30,  # seconds
            "alert_thresholds": {
                "consumer_lag_critical": 10000,
                "consumer_lag_warning": 5000,
                "error_rate_warning": 0.05,
                "disk_usage_critical": 0.90,
                "disk_usage_warning": 0.80,
                "under_replicated_critical": 0.1,
            },
            "retention": {
                "metrics_history_hours": 24,
                "alerts_history_hours": 168,  # 1 week
            },
            "export": {
                "prometheus_enabled": True,
                "prometheus_port": 9090,
                "json_enabled": True,
                "csv_enabled": True,
            }
        }
    
    def setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics collectors."""
        self.registry = CollectorRegistry()
        
        # Topic metrics
        self.topic_messages_total = Counter(
            'kafka_topic_messages_total',
            'Total messages in topic',
            ['topic'],
            registry=self.registry
        )
        
        self.topic_size_bytes = Gauge(
            'kafka_topic_size_bytes',
            'Total size of topic in bytes',
            ['topic'],
            registry=self.registry
        )
        
        self.topic_message_rate = Gauge(
            'kafka_topic_message_rate',
            'Message rate per second',
            ['topic'],
            registry=self.registry
        )
        
        self.topic_consumer_lag = Gauge(
            'kafka_topic_consumer_lag',
            'Consumer lag per topic and group',
            ['topic', 'consumer_group'],
            registry=self.registry
        )
        
        self.topic_health_score = Gauge(
            'kafka_topic_health_score',
            'Topic health score (0-100)',
            ['topic'],
            registry=self.registry
        )
        
        # Partition metrics
        self.partition_high_watermark = Gauge(
            'kafka_partition_high_watermark',
            'Partition high watermark',
            ['topic', 'partition'],
            registry=self.registry
        )
        
        self.partition_lag = Gauge(
            'kafka_partition_lag',
            'Partition lag per consumer group',
            ['topic', 'partition', 'consumer_group'],
            registry=self.registry
        )
        
        # Consumer group metrics
        self.consumer_group_members = Gauge(
            'kafka_consumer_group_members',
            'Number of members in consumer group',
            ['consumer_group'],
            registry=self.registry
        )
        
        self.consumer_group_lag_total = Gauge(
            'kafka_consumer_group_lag_total',
            'Total lag for consumer group',
            ['consumer_group'],
            registry=self.registry
        )
        
        # Cluster metrics
        self.cluster_brokers = Gauge(
            'kafka_cluster_brokers',
            'Number of brokers in cluster',
            registry=self.registry
        )
        
        self.cluster_topics = Gauge(
            'kafka_cluster_topics',
            'Number of topics in cluster',
            registry=self.registry
        )
        
        self.cluster_under_replicated_partitions = Gauge(
            'kafka_cluster_under_replicated_partitions',
            'Number of under-replicated partitions',
            registry=self.registry
        )
    
    async def collect_all_metrics(self, topics: Optional[List[str]] = None) -> None:
        """Collect all metrics for specified topics or all topics."""
        try:
            # Get cluster metadata
            metadata = self.admin_client.list_topics(timeout=10)
            self.cluster_metrics.broker_count = len(metadata.brokers)
            self.cluster_metrics.topic_count = len(metadata.topics)
            
            # Determine topics to monitor
            topics_to_monitor = topics or list(metadata.topics.keys())
            topics_to_monitor = [t for t in topics_to_monitor if not t.startswith('__')]
            
            # Collect topic metrics
            await self._collect_topic_metrics(topics_to_monitor)
            
            # Collect consumer group metrics
            await self._collect_consumer_group_metrics()
            
            # Update cluster metrics
            self._update_cluster_metrics()
            
            # Calculate health scores
            self._calculate_health_scores()
            
            # Check for alerts
            self._check_alerts()
            
            # Update Prometheus metrics
            self._update_prometheus_metrics()
            
            # Clean up old data
            self._cleanup_old_data()
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            raise
    
    async def _collect_topic_metrics(self, topics: List[str]) -> None:
        """Collect metrics for specified topics."""
        for topic_name in topics:
            try:
                # Get topic metadata
                topic_metadata = self.admin_client.list_topics(topic=topic_name, timeout=10)
                if topic_name not in topic_metadata.topics:
                    continue
                
                topic_info = topic_metadata.topics[topic_name]
                
                # Initialize topic metrics if not exists
                if topic_name not in self.topic_metrics:
                    self.topic_metrics[topic_name] = TopicMetrics(name=topic_name)
                
                topic_metrics = self.topic_metrics[topic_name]
                topic_metrics.partitions.clear()
                
                total_messages = 0
                total_size = 0
                
                # Collect partition metrics
                for partition_id, partition_info in topic_info.partitions.items():
                    partition_metrics = PartitionMetrics(partition_id=partition_id)
                    
                    # Get watermarks
                    try:
                        tp = TopicPartition(topic_name, partition_id)
                        low, high = self.consumer.get_watermark_offsets(tp, timeout=10)
                        partition_metrics.low_watermark = low
                        partition_metrics.high_watermark = high
                        partition_metrics.message_count = high - low
                        total_messages += partition_metrics.message_count
                    except Exception as e:
                        logger.warning(f"Failed to get watermarks for {topic_name}:{partition_id}: {e}")
                    
                    # Get partition size (estimate)
                    partition_metrics.size_bytes = partition_metrics.message_count * 1024  # rough estimate
                    total_size += partition_metrics.size_bytes
                    
                    # Replica information
                    partition_metrics.leader = partition_info.leader
                    partition_metrics.replicas = partition_info.replicas
                    partition_metrics.in_sync_replicas = partition_info.isrs
                    
                    topic_metrics.partitions[partition_id] = partition_metrics
                
                # Update topic totals
                topic_metrics.total_messages = total_messages
                topic_metrics.total_size_bytes = total_size
                
                # Calculate rates
                await self._calculate_rates(topic_name, topic_metrics)
                
                topic_metrics.last_updated = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Failed to collect metrics for topic {topic_name}: {e}")
    
    async def _calculate_rates(self, topic_name: str, metrics: TopicMetrics) -> None:
        """Calculate message and byte rates for a topic."""
        current_time = datetime.utcnow()
        current_messages = metrics.total_messages
        current_bytes = metrics.total_size_bytes
        
        # Store current values in history
        history_key = f"{topic_name}_messages"
        if history_key not in self.metric_history:
            self.metric_history[history_key] = []
        
        self.metric_history[history_key].append((current_time, current_messages))
        
        bytes_history_key = f"{topic_name}_bytes"
        if bytes_history_key not in self.metric_history:
            self.metric_history[bytes_history_key] = []
        
        self.metric_history[bytes_history_key].append((current_time, current_bytes))
        
        # Calculate rates if we have previous data
        if len(self.metric_history[history_key]) >= 2:
            prev_time, prev_messages = self.metric_history[history_key][-2]
            time_diff = (current_time - prev_time).total_seconds()
            
            if time_diff > 0:
                message_diff = current_messages - prev_messages
                metrics.message_rate = message_diff / time_diff
        
        if len(self.metric_history[bytes_history_key]) >= 2:
            prev_time, prev_bytes = self.metric_history[bytes_history_key][-2]
            time_diff = (current_time - prev_time).total_seconds()
            
            if time_diff > 0:
                bytes_diff = current_bytes - prev_bytes
                metrics.byte_rate = bytes_diff / time_diff
    
    async def _collect_consumer_group_metrics(self) -> None:
        """Collect consumer group metrics."""
        try:
            # Get consumer groups
            groups = self.admin_client.list_consumer_groups()
            
            for group_info in groups:
                group_id = group_info.id
                
                try:
                    # Get group metadata
                    group_metadata = self.admin_client.describe_consumer_groups([group_id])
                    
                    for group, future in group_metadata.items():
                        try:
                            group_desc = future.result(timeout=10)
                            
                            # Initialize consumer group metrics
                            if group_id not in self.consumer_group_metrics:
                                self.consumer_group_metrics[group_id] = ConsumerGroupMetrics(group_id=group_id)
                            
                            group_metrics = self.consumer_group_metrics[group_id]
                            group_metrics.state = group_desc.state
                            group_metrics.members = len(group_desc.members)
                            
                            # Get topic assignments
                            group_metrics.topics.clear()
                            group_metrics.partition_assignments.clear()
                            
                            for member in group_desc.members:
                                for assignment in member.assignment:
                                    topic = assignment.topic
                                    group_metrics.topics.add(topic)
                                    
                                    if topic not in group_metrics.partition_assignments:
                                        group_metrics.partition_assignments[topic] = []
                                    group_metrics.partition_assignments[topic].extend(assignment.partitions)
                            
                            # Calculate lag
                            await self._calculate_consumer_lag(group_id, group_metrics)
                            
                        except Exception as e:
                            logger.warning(f"Failed to get details for consumer group {group_id}: {e}")
                
                except Exception as e:
                    logger.warning(f"Failed to describe consumer group {group_id}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to collect consumer group metrics: {e}")
    
    async def _calculate_consumer_lag(
        self, group_id: str, group_metrics: ConsumerGroupMetrics
    ) -> None:
        """Calculate consumer lag for a consumer group."""
        try:
            total_lag = 0
            max_lag = 0
            
            for topic in group_metrics.topics:
                if topic in self.topic_metrics:
                    topic_metrics = self.topic_metrics[topic]
                    
                    for partition_id, partition_metrics in topic_metrics.partitions.items():
                        try:
                            # Get committed offset
                            tp = TopicPartition(topic, partition_id)
                            committed = self.consumer.committed([tp], timeout=10)
                            
                            if committed and committed[0].offset >= 0:
                                lag = partition_metrics.high_watermark - committed[0].offset
                                partition_metrics.lag[group_id] = lag
                                total_lag += lag
                                max_lag = max(max_lag, lag)
                        
                        except Exception as e:
                            logger.warning(f"Failed to get lag for {topic}:{partition_id} group {group_id}: {e}")
            
            group_metrics.total_lag = total_lag
            group_metrics.max_lag = max_lag
            
            # Update topic metrics
            for topic in group_metrics.topics:
                if topic in self.topic_metrics:
                    self.topic_metrics[topic].consumer_groups.add(group_id)
                    topic_lag = sum(
                        p.lag.get(group_id, 0) 
                        for p in self.topic_metrics[topic].partitions.values()
                    )
                    self.topic_metrics[topic].total_lag = max(
                        self.topic_metrics[topic].total_lag, topic_lag
                    )
                    self.topic_metrics[topic].max_lag = max(
                        self.topic_metrics[topic].max_lag,
                        max(p.lag.get(group_id, 0) for p in self.topic_metrics[topic].partitions.values())
                    )
        
        except Exception as e:
            logger.error(f"Failed to calculate consumer lag for group {group_id}: {e}")
    
    def _update_cluster_metrics(self) -> None:
        """Update cluster-level metrics."""
        self.cluster_metrics.partition_count = sum(
            len(topic.partitions) for topic in self.topic_metrics.values()
        )
        
        self.cluster_metrics.total_messages = sum(
            topic.total_messages for topic in self.topic_metrics.values()
        )
        
        self.cluster_metrics.total_size_bytes = sum(
            topic.total_size_bytes for topic in self.topic_metrics.values()
        )
        
        # Count under-replicated partitions
        under_replicated = 0
        offline = 0
        
        for topic_metrics in self.topic_metrics.values():
            for partition_metrics in topic_metrics.partitions.values():
                replica_count = len(partition_metrics.replicas)
                isr_count = len(partition_metrics.in_sync_replicas)
                
                if isr_count < replica_count:
                    under_replicated += 1
                
                if isr_count == 0:
                    offline += 1
        
        self.cluster_metrics.under_replicated_partitions = under_replicated
        self.cluster_metrics.offline_partitions = offline
        self.cluster_metrics.active_consumer_groups = len(self.consumer_group_metrics)
        self.cluster_metrics.total_consumer_lag = sum(
            group.total_lag for group in self.consumer_group_metrics.values()
        )
    
    def _calculate_health_scores(self) -> None:
        """Calculate health scores for topics."""
        for topic_name, topic_metrics in self.topic_metrics.items():
            score = 100.0
            
            # Penalize for under-replicated partitions
            total_partitions = len(topic_metrics.partitions)
            under_replicated = sum(
                1 for p in topic_metrics.partitions.values()
                if len(p.in_sync_replicas) < len(p.replicas)
            )
            
            if total_partitions > 0:
                under_replicated_ratio = under_replicated / total_partitions
                score -= under_replicated_ratio * 30  # Up to 30 points penalty
            
            # Penalize for high consumer lag
            if topic_metrics.max_lag > self.monitoring_config["alert_thresholds"]["consumer_lag_critical"]:
                score -= 40  # Major penalty for critical lag
            elif topic_metrics.max_lag > self.monitoring_config["alert_thresholds"]["consumer_lag_warning"]:
                score -= 20  # Moderate penalty for warning lag
            
            # Penalize for offline partitions
            offline_partitions = sum(
                1 for p in topic_metrics.partitions.values()
                if len(p.in_sync_replicas) == 0
            )
            
            if offline_partitions > 0:
                score -= 50  # Major penalty for offline partitions
            
            # Ensure score is between 0 and 100
            topic_metrics.health_score = max(0.0, min(100.0, score))
    
    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        current_time = datetime.utcnow()
        new_alerts = []
        
        thresholds = self.monitoring_config["alert_thresholds"]
        
        # Check consumer lag alerts
        for topic_name, topic_metrics in self.topic_metrics.items():
            if topic_metrics.max_lag > thresholds["consumer_lag_critical"]:
                new_alerts.append(Alert(
                    severity="critical",
                    topic=topic_name,
                    message=f"Critical consumer lag: {topic_metrics.max_lag} messages",
                    timestamp=current_time,
                    details={"lag": topic_metrics.max_lag}
                ))
            elif topic_metrics.max_lag > thresholds["consumer_lag_warning"]:
                new_alerts.append(Alert(
                    severity="warning",
                    topic=topic_name,
                    message=f"High consumer lag: {topic_metrics.max_lag} messages",
                    timestamp=current_time,
                    details={"lag": topic_metrics.max_lag}
                ))
        
        # Check under-replicated partitions
        cluster_under_replicated_ratio = (
            self.cluster_metrics.under_replicated_partitions / 
            max(1, self.cluster_metrics.partition_count)
        )
        
        if cluster_under_replicated_ratio > thresholds["under_replicated_critical"]:
            new_alerts.append(Alert(
                severity="critical",
                topic=None,
                message=f"High under-replicated partitions: {self.cluster_metrics.under_replicated_partitions}",
                timestamp=current_time,
                details={"count": self.cluster_metrics.under_replicated_partitions}
            ))
        
        # Check offline partitions
        if self.cluster_metrics.offline_partitions > 0:
            new_alerts.append(Alert(
                severity="critical",
                topic=None,
                message=f"Offline partitions detected: {self.cluster_metrics.offline_partitions}",
                timestamp=current_time,
                details={"count": self.cluster_metrics.offline_partitions}
            ))
        
        # Add new alerts
        self.alerts.extend(new_alerts)
        
        # Log new alerts
        for alert in new_alerts:
            log_level = logging.CRITICAL if alert.severity == "critical" else logging.WARNING
            logger.log(log_level, f"ALERT [{alert.severity.upper()}] {alert.message}")
    
    def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics."""
        # Clear existing metrics
        self.topic_messages_total.clear()
        self.topic_size_bytes.clear()
        self.topic_message_rate.clear()
        self.topic_consumer_lag.clear()
        self.topic_health_score.clear()
        self.partition_high_watermark.clear()
        self.partition_lag.clear()
        self.consumer_group_members.clear()
        self.consumer_group_lag_total.clear()
        
        # Update topic metrics
        for topic_name, topic_metrics in self.topic_metrics.items():
            self.topic_messages_total.labels(topic=topic_name).inc(topic_metrics.total_messages)
            self.topic_size_bytes.labels(topic=topic_name).set(topic_metrics.total_size_bytes)
            self.topic_message_rate.labels(topic=topic_name).set(topic_metrics.message_rate)
            self.topic_health_score.labels(topic=topic_name).set(topic_metrics.health_score)
            
            # Partition metrics
            for partition_id, partition_metrics in topic_metrics.partitions.items():
                self.partition_high_watermark.labels(
                    topic=topic_name, 
                    partition=str(partition_id)
                ).set(partition_metrics.high_watermark)
                
                # Consumer lag per partition and group
                for group_id, lag in partition_metrics.lag.items():
                    self.partition_lag.labels(
                        topic=topic_name,
                        partition=str(partition_id),
                        consumer_group=group_id
                    ).set(lag)
                    
                    self.topic_consumer_lag.labels(
                        topic=topic_name,
                        consumer_group=group_id
                    ).set(lag)
        
        # Update consumer group metrics
        for group_id, group_metrics in self.consumer_group_metrics.items():
            self.consumer_group_members.labels(consumer_group=group_id).set(group_metrics.members)
            self.consumer_group_lag_total.labels(consumer_group=group_id).set(group_metrics.total_lag)
        
        # Update cluster metrics
        self.cluster_brokers.set(self.cluster_metrics.broker_count)
        self.cluster_topics.set(self.cluster_metrics.topic_count)
        self.cluster_under_replicated_partitions.set(self.cluster_metrics.under_replicated_partitions)
    
    def _cleanup_old_data(self) -> None:
        """Clean up old historical data and alerts."""
        current_time = datetime.utcnow()
        
        # Clean up metrics history
        retention_hours = self.monitoring_config["retention"]["metrics_history_hours"]
        cutoff_time = current_time - timedelta(hours=retention_hours)
        
        for key, history in self.metric_history.items():
            self.metric_history[key] = [
                (timestamp, value) for timestamp, value in history
                if timestamp > cutoff_time
            ]
        
        # Clean up old alerts
        alert_retention_hours = self.monitoring_config["retention"]["alerts_history_hours"]
        alert_cutoff_time = current_time - timedelta(hours=alert_retention_hours)
        
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > alert_cutoff_time
        ]
    
    def generate_report(self, format: str = "json") -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        current_time = datetime.utcnow()
        
        report = {
            "timestamp": current_time.isoformat(),
            "cluster": {
                "brokers": self.cluster_metrics.broker_count,
                "topics": self.cluster_metrics.topic_count,
                "partitions": self.cluster_metrics.partition_count,
                "total_messages": self.cluster_metrics.total_messages,
                "total_size_bytes": self.cluster_metrics.total_size_bytes,
                "under_replicated_partitions": self.cluster_metrics.under_replicated_partitions,
                "offline_partitions": self.cluster_metrics.offline_partitions,
                "active_consumer_groups": self.cluster_metrics.active_consumer_groups,
                "total_consumer_lag": self.cluster_metrics.total_consumer_lag,
            },
            "topics": {},
            "consumer_groups": {},
            "alerts": [],
            "summary": {
                "healthy_topics": 0,
                "warning_topics": 0,
                "critical_topics": 0,
                "average_health_score": 0.0,
            }
        }
        
        # Add topic details
        health_scores = []
        for topic_name, topic_metrics in self.topic_metrics.items():
            topic_data = {
                "name": topic_name,
                "partitions": len(topic_metrics.partitions),
                "total_messages": topic_metrics.total_messages,
                "total_size_bytes": topic_metrics.total_size_bytes,
                "message_rate": topic_metrics.message_rate,
                "byte_rate": topic_metrics.byte_rate,
                "consumer_groups": list(topic_metrics.consumer_groups),
                "total_lag": topic_metrics.total_lag,
                "max_lag": topic_metrics.max_lag,
                "health_score": topic_metrics.health_score,
                "last_updated": topic_metrics.last_updated.isoformat(),
            }
            
            report["topics"][topic_name] = topic_data
            health_scores.append(topic_metrics.health_score)
            
            # Categorize topics by health
            if topic_metrics.health_score >= 90:
                report["summary"]["healthy_topics"] += 1
            elif topic_metrics.health_score >= 70:
                report["summary"]["warning_topics"] += 1
            else:
                report["summary"]["critical_topics"] += 1
        
        # Calculate average health score
        if health_scores:
            report["summary"]["average_health_score"] = statistics.mean(health_scores)
        
        # Add consumer group details
        for group_id, group_metrics in self.consumer_group_metrics.items():
            group_data = {
                "group_id": group_id,
                "state": group_metrics.state,
                "members": group_metrics.members,
                "topics": list(group_metrics.topics),
                "total_lag": group_metrics.total_lag,
                "max_lag": group_metrics.max_lag,
            }
            
            report["consumer_groups"][group_id] = group_data
        
        # Add recent alerts
        recent_alerts = [
            {
                "severity": alert.severity,
                "topic": alert.topic,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "details": alert.details,
            }
            for alert in self.alerts[-50:]  # Last 50 alerts
        ]
        
        report["alerts"] = recent_alerts
        
        return report
    
    def create_dashboard_layout(self) -> Layout:
        """Create Rich dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        
        layout["left"].split_column(
            Layout(name="cluster"),
            Layout(name="topics"),
        )
        
        layout["right"].split_column(
            Layout(name="consumer_groups"),
            Layout(name="alerts"),
        )
        
        return layout
    
    def render_dashboard(self) -> Layout:
        """Render the monitoring dashboard."""
        layout = self.create_dashboard_layout()
        
        # Header
        header_text = f"🏥 TDA Kafka Topic Monitor - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        layout["header"].update(Panel(header_text, style="bold blue"))
        
        # Cluster overview
        cluster_table = Table(title="Cluster Overview")
        cluster_table.add_column("Metric", style="cyan")
        cluster_table.add_column("Value", justify="right")
        
        cluster_table.add_row("Brokers", str(self.cluster_metrics.broker_count))
        cluster_table.add_row("Topics", str(self.cluster_metrics.topic_count))
        cluster_table.add_row("Partitions", str(self.cluster_metrics.partition_count))
        cluster_table.add_row("Total Messages", f"{self.cluster_metrics.total_messages:,}")
        cluster_table.add_row("Total Size", f"{self.cluster_metrics.total_size_bytes:,} bytes")
        cluster_table.add_row("Under-replicated", str(self.cluster_metrics.under_replicated_partitions))
        cluster_table.add_row("Offline Partitions", str(self.cluster_metrics.offline_partitions))
        cluster_table.add_row("Consumer Groups", str(self.cluster_metrics.active_consumer_groups))
        cluster_table.add_row("Total Lag", f"{self.cluster_metrics.total_consumer_lag:,}")
        
        layout["cluster"].update(Panel(cluster_table, title="Cluster"))
        
        # Topics overview
        topics_table = Table(title="Topic Health")
        topics_table.add_column("Topic", style="cyan")
        topics_table.add_column("Health", justify="center")
        topics_table.add_column("Messages", justify="right")
        topics_table.add_column("Lag", justify="right")
        topics_table.add_column("Rate", justify="right")
        
        for topic_name, topic_metrics in sorted(self.topic_metrics.items()):
            # Health indicator
            if topic_metrics.health_score >= 90:
                health_icon = "🟢"
            elif topic_metrics.health_score >= 70:
                health_icon = "🟡"
            else:
                health_icon = "🔴"
            
            health_text = f"{health_icon} {topic_metrics.health_score:.1f}"
            
            topics_table.add_row(
                topic_name[:20],  # Truncate long names
                health_text,
                f"{topic_metrics.total_messages:,}",
                f"{topic_metrics.max_lag:,}",
                f"{topic_metrics.message_rate:.1f}/s"
            )
        
        layout["topics"].update(Panel(topics_table, title="Topics"))
        
        # Consumer groups
        groups_table = Table(title="Consumer Groups")
        groups_table.add_column("Group", style="cyan")
        groups_table.add_column("State")
        groups_table.add_column("Members", justify="right")
        groups_table.add_column("Lag", justify="right")
        
        for group_id, group_metrics in sorted(self.consumer_group_metrics.items()):
            state_color = "green" if group_metrics.state == "Stable" else "yellow"
            
            groups_table.add_row(
                group_id[:20],
                f"[{state_color}]{group_metrics.state}[/{state_color}]",
                str(group_metrics.members),
                f"{group_metrics.total_lag:,}"
            )
        
        layout["consumer_groups"].update(Panel(groups_table, title="Consumer Groups"))
        
        # Recent alerts
        alerts_table = Table(title="Recent Alerts")
        alerts_table.add_column("Time", style="dim")
        alerts_table.add_column("Severity")
        alerts_table.add_column("Message", style="yellow")
        
        recent_alerts = sorted(self.alerts[-10:], key=lambda x: x.timestamp, reverse=True)
        for alert in recent_alerts:
            severity_color = {
                "critical": "red",
                "warning": "yellow",
                "info": "blue"
            }.get(alert.severity, "white")
            
            alerts_table.add_row(
                alert.timestamp.strftime("%H:%M:%S"),
                f"[{severity_color}]{alert.severity.upper()}[/{severity_color}]",
                alert.message[:40]  # Truncate long messages
            )
        
        if not recent_alerts:
            alerts_table.add_row("-", "-", "No recent alerts")
        
        layout["alerts"].update(Panel(alerts_table, title="Alerts"))
        
        # Footer
        footer_text = f"Collection Interval: {self.monitoring_config['collection_interval']}s | " \
                     f"Prometheus: {self.monitoring_config['export']['prometheus_port']} | " \
                     f"Press Ctrl+C to exit"
        layout["footer"].update(Panel(footer_text, style="dim"))
        
        return layout
    
    async def start_monitoring(
        self,
        topics: Optional[List[str]] = None,
        interval: Optional[int] = None,
        dashboard: bool = False
    ) -> None:
        """Start continuous monitoring."""
        collection_interval = interval or self.monitoring_config["collection_interval"]
        
        console.print(f"🚀 Starting TDA Kafka Topic Monitor")
        console.print(f"📊 Collection interval: {collection_interval} seconds")
        
        if topics:
            console.print(f"🎯 Monitoring topics: {', '.join(topics)}")
        else:
            console.print("🎯 Monitoring all topics")
        
        if dashboard:
            console.print("📈 Dashboard mode enabled")
            
            # Start Prometheus server if enabled
            if self.monitoring_config["export"]["prometheus_enabled"]:
                prometheus_port = self.monitoring_config["export"]["prometheus_port"]
                start_http_server(prometheus_port, registry=self.registry)
                console.print(f"📊 Prometheus metrics available on port {prometheus_port}")
            
            # Run dashboard
            with Live(self.render_dashboard(), refresh_per_second=1, console=console) as live:
                try:
                    while True:
                        await self.collect_all_metrics(topics)
                        live.update(self.render_dashboard())
                        await asyncio.sleep(collection_interval)
                except KeyboardInterrupt:
                    console.print("\n👋 Monitoring stopped")
        else:
            # Run without dashboard
            try:
                while True:
                    await self.collect_all_metrics(topics)
                    console.print(f"✅ Metrics collected at {datetime.utcnow().strftime('%H:%M:%S')}")
                    await asyncio.sleep(collection_interval)
            except KeyboardInterrupt:
                console.print("\n👋 Monitoring stopped")


# CLI Commands
@click.group()
@click.option(
    "--kafka-config",
    help="Kafka configuration (JSON format)"
)
@click.option(
    "--monitoring-config",
    help="Monitoring configuration file path"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level"
)
@click.pass_context
def cli(ctx, kafka_config, monitoring_config, log_level):
    """TDA Kafka Topic Monitor - Advanced monitoring and analytics tool."""
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    kafka_conf = None
    if kafka_config:
        kafka_conf = json.loads(kafka_config)
    
    monitor_conf = None
    if monitoring_config:
        with open(monitoring_config, 'r') as f:
            monitor_conf = yaml.safe_load(f)
    
    ctx.ensure_object(dict)
    ctx.obj["monitor"] = TopicMonitor(
        kafka_config=kafka_conf,
        monitoring_config=monitor_conf
    )


@cli.command()
@click.option(
    "--topics",
    help="Comma-separated list of topics to monitor (default: all)"
)
@click.option(
    "--interval",
    type=int,
    help="Collection interval in seconds"
)
@click.option(
    "--dashboard",
    is_flag=True,
    help="Enable real-time dashboard"
)
@click.pass_context
def monitor(ctx, topics, interval, dashboard):
    """Start continuous monitoring."""
    monitor = ctx.obj["monitor"]
    
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    async def _monitor():
        await monitor.start_monitoring(
            topics=topic_list,
            interval=interval,
            dashboard=dashboard
        )
    
    asyncio.run(_monitor())


@cli.command()
@click.option(
    "--topics",
    help="Comma-separated list of topics to include in report (default: all)"
)
@click.option(
    "--format",
    default="json",
    type=click.Choice(["json", "yaml"]),
    help="Report format"
)
@click.option(
    "--output",
    help="Output file path (default: stdout)"
)
@click.pass_context
def report(ctx, topics, format, output):
    """Generate monitoring report."""
    monitor = ctx.obj["monitor"]
    
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    async def _generate_report():
        # Collect latest metrics
        await monitor.collect_all_metrics(topics=topic_list)
        
        # Generate report
        report_data = monitor.generate_report(format=format)
        
        if format == "json":
            report_text = json.dumps(report_data, indent=2)
        else:  # yaml
            report_text = yaml.dump(report_data, default_flow_style=False)
        
        if output:
            with open(output, 'w') as f:
                f.write(report_text)
            console.print(f"📊 Report saved to {output}")
        else:
            console.print(report_text)
    
    asyncio.run(_generate_report())


@cli.command()
@click.option(
    "--check-interval",
    default=60,
    type=int,
    help="Alert check interval in seconds"
)
@click.option(
    "--webhook-url",
    help="Webhook URL for alert notifications"
)
@click.pass_context
def alerts(ctx, check_interval, webhook_url):
    """Monitor and send alerts."""
    monitor = ctx.obj["monitor"]
    
    console.print(f"🚨 Starting alert monitoring (interval: {check_interval}s)")
    
    async def _alert_monitor():
        try:
            while True:
                await monitor.collect_all_metrics()
                
                # Send webhook notifications if configured
                if webhook_url:
                    # TODO: Implement webhook notifications
                    pass
                
                await asyncio.sleep(check_interval)
        except KeyboardInterrupt:
            console.print("\n👋 Alert monitoring stopped")
    
    asyncio.run(_alert_monitor())


@cli.command()
@click.option(
    "--port",
    default=8080,
    type=int,
    help="Dashboard port"
)
@click.option(
    "--topics",
    help="Comma-separated list of topics to monitor"
)
@click.pass_context
def dashboard(ctx, port, topics):
    """Start web dashboard (simplified version)."""
    monitor = ctx.obj["monitor"]
    
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    console.print(f"🌐 Starting dashboard on port {port}")
    
    async def _dashboard():
        await monitor.start_monitoring(
            topics=topic_list,
            dashboard=True
        )
    
    asyncio.run(_dashboard())


if __name__ == "__main__":
    cli()
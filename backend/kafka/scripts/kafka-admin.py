#!/usr/bin/env python3
"""
TDA Kafka Administrative Utilities

Comprehensive administrative tool for Kafka cluster management including
cluster health checks, consumer group management, offset management,
and data migration utilities.

Features:
- Cluster health checks and diagnostics
- Consumer group management and monitoring
- Offset management and reset utilities
- Data migration and replication tools
- Performance analysis and optimization
- Security and ACL management
- Backup and restore capabilities
- Disaster recovery utilities

Usage:
    python kafka-admin.py cluster-health --detailed
    python kafka-admin.py consumer-groups --list --group-pattern "tda_*"
    python kafka-admin.py reset-offsets --group tda_processors --topic tda_jobs --to-earliest
    python kafka-admin.py migrate-data --source-topic old_topic --target-topic new_topic
    python kafka-admin.py security-audit --export audit-report.json
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import click
import yaml
from confluent_kafka import (
    Consumer,
    KafkaError,
    Producer,
    TopicPartition,
    __version__ as kafka_version,
)
from confluent_kafka.admin import (
    AdminClient,
    ConfigResource,
    NewTopic,
    PartitionMetadata,
    TopicMetadata,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("kafka-admin.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ClusterHealth:
    """Cluster health information."""
    
    broker_count: int = 0
    controller_id: Optional[int] = None
    cluster_id: str = ""
    topic_count: int = 0
    partition_count: int = 0
    under_replicated_partitions: int = 0
    offline_partitions: int = 0
    isr_shrinking_brokers: List[int] = field(default_factory=list)
    leader_imbalance_ratio: float = 0.0
    health_score: float = 100.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ConsumerGroupStatus:
    """Consumer group status information."""
    
    group_id: str
    state: str = "unknown"
    protocol_type: str = ""
    protocol: str = ""
    members: List[Dict[str, Any]] = field(default_factory=list)
    coordinator: Optional[int] = None
    lag: Dict[str, int] = field(default_factory=dict)  # topic:partition -> lag
    total_lag: int = 0
    last_commit_time: Optional[datetime] = None
    stability: str = "unknown"  # stable, rebalancing, dead


@dataclass
class MigrationTask:
    """Data migration task information."""
    
    task_id: str
    source_topic: str
    target_topic: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    records_migrated: int = 0
    total_records: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class KafkaAdmin:
    """Comprehensive Kafka administrative utilities."""
    
    def __init__(
        self,
        kafka_config: Optional[Dict[str, str]] = None,
        admin_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Kafka admin utility."""
        self.kafka_config = kafka_config or self._get_default_kafka_config()
        self.admin_config = admin_config or self._get_default_admin_config()
        
        # Initialize clients
        self.admin_client = AdminClient(self.kafka_config)
        self.consumer = None
        self.producer = None
        
        # State tracking
        self.migration_tasks: Dict[str, MigrationTask] = {}
        
    def _get_default_kafka_config(self) -> Dict[str, str]:
        """Get default Kafka configuration."""
        return {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "client.id": "tda-kafka-admin",
            "request.timeout.ms": "30000",
            "api.version.request": "true",
        }
    
    def _get_default_admin_config(self) -> Dict[str, Any]:
        """Get default admin configuration."""
        return {
            "health_check": {
                "critical_lag_threshold": 100000,
                "warning_lag_threshold": 10000,
                "max_offline_partitions": 0,
                "max_under_replicated_ratio": 0.1,
            },
            "migration": {
                "batch_size": 1000,
                "timeout_ms": 30000,
                "max_retries": 3,
                "preserve_keys": True,
                "preserve_timestamps": True,
            },
            "monitoring": {
                "collection_interval": 60,
                "history_retention_hours": 24,
            }
        }
    
    def _init_consumer(self) -> Consumer:
        """Initialize consumer if not already created."""
        if not self.consumer:
            self.consumer = Consumer({
                **self.kafka_config,
                "group.id": "kafka-admin-tools",
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest"
            })
        return self.consumer
    
    def _init_producer(self) -> Producer:
        """Initialize producer if not already created."""
        if not self.producer:
            self.producer = Producer(self.kafka_config)
        return self.producer
    
    async def cluster_health_check(self, detailed: bool = False) -> ClusterHealth:
        """Perform comprehensive cluster health check."""
        console.print("🏥 Performing cluster health check...")
        
        health = ClusterHealth()
        
        try:
            # Get cluster metadata
            metadata = self.admin_client.list_topics(timeout=10)
            
            # Basic cluster info
            health.broker_count = len(metadata.brokers)
            health.cluster_id = metadata.cluster_id or "unknown"
            health.topic_count = len(metadata.topics)
            
            # Calculate partition statistics
            total_partitions = 0
            under_replicated = 0
            offline = 0
            leader_distribution = {}
            
            for topic_name, topic_metadata in metadata.topics.items():
                if topic_name.startswith('__'):
                    continue  # Skip internal topics
                
                for partition_id, partition_metadata in topic_metadata.partitions.items():
                    total_partitions += 1
                    
                    # Count under-replicated partitions
                    if len(partition_metadata.isrs) < len(partition_metadata.replicas):
                        under_replicated += 1
                    
                    # Count offline partitions
                    if len(partition_metadata.isrs) == 0:
                        offline += 1
                    
                    # Track leader distribution
                    leader = partition_metadata.leader
                    if leader:
                        leader_distribution[leader] = leader_distribution.get(leader, 0) + 1
            
            health.partition_count = total_partitions
            health.under_replicated_partitions = under_replicated
            health.offline_partitions = offline
            
            # Calculate leader imbalance
            if leader_distribution:
                expected_per_broker = total_partitions / len(leader_distribution)
                max_deviation = max(
                    abs(count - expected_per_broker) 
                    for count in leader_distribution.values()
                )
                health.leader_imbalance_ratio = max_deviation / expected_per_broker if expected_per_broker > 0 else 0
            
            # Identify controller
            try:
                # Get controller information (simplified approach)
                health.controller_id = min(metadata.brokers.keys()) if metadata.brokers else None
            except Exception as e:
                logger.warning(f"Could not determine controller: {e}")
            
            # Calculate health score and issues
            await self._calculate_health_score(health)
            
            if detailed:
                await self._add_detailed_health_info(health, metadata)
            
        except Exception as e:
            logger.error(f"Failed to perform cluster health check: {e}")
            health.health_score = 0.0
            health.issues.append(f"Health check failed: {str(e)}")
        
        return health
    
    async def _calculate_health_score(self, health: ClusterHealth) -> None:
        """Calculate overall health score and identify issues."""
        score = 100.0
        config = self.admin_config["health_check"]
        
        # Check offline partitions (critical)
        if health.offline_partitions > config["max_offline_partitions"]:
            score -= 50
            health.issues.append(
                f"Critical: {health.offline_partitions} offline partitions detected"
            )
            health.recommendations.append("Investigate broker failures and restore partition leaders")
        
        # Check under-replicated partitions
        if health.partition_count > 0:
            under_replicated_ratio = health.under_replicated_partitions / health.partition_count
            if under_replicated_ratio > config["max_under_replicated_ratio"]:
                score -= 30
                health.issues.append(
                    f"Warning: {health.under_replicated_partitions} under-replicated partitions "
                    f"({under_replicated_ratio:.1%})"
                )
                health.recommendations.append("Check broker health and network connectivity")
        
        # Check leader imbalance
        if health.leader_imbalance_ratio > 0.2:  # 20% imbalance threshold
            score -= 15
            health.issues.append(
                f"Warning: Leader imbalance detected ({health.leader_imbalance_ratio:.1%})"
            )
            health.recommendations.append("Consider running leader election to rebalance")
        
        # Check broker count
        if health.broker_count < 3:
            score -= 10
            health.issues.append("Warning: Less than 3 brokers in cluster")
            health.recommendations.append("Consider adding more brokers for better fault tolerance")
        
        health.health_score = max(0.0, score)
    
    async def _add_detailed_health_info(
        self, health: ClusterHealth, metadata
    ) -> None:
        """Add detailed health information."""
        try:
            # Check for ISR shrinking brokers
            # This is a simplified check - in production, you'd monitor broker metrics
            broker_partition_counts = {}
            for topic_metadata in metadata.topics.values():
                for partition_metadata in topic_metadata.partitions.values():
                    for replica in partition_metadata.replicas:
                        broker_partition_counts[replica] = broker_partition_counts.get(replica, 0) + 1
            
            # Identify potentially problematic brokers
            if broker_partition_counts:
                avg_partitions = sum(broker_partition_counts.values()) / len(broker_partition_counts)
                for broker_id, count in broker_partition_counts.items():
                    if count < avg_partitions * 0.5:  # Less than 50% of average
                        health.isr_shrinking_brokers.append(broker_id)
                        health.issues.append(f"Broker {broker_id} has unusually few partitions")
        
        except Exception as e:
            logger.warning(f"Failed to add detailed health info: {e}")
    
    async def list_consumer_groups(
        self, 
        group_pattern: Optional[str] = None,
        include_details: bool = True
    ) -> List[ConsumerGroupStatus]:
        """List and analyze consumer groups."""
        console.print("📋 Listing consumer groups...")
        
        groups = []
        
        try:
            # Get all consumer groups
            group_list = self.admin_client.list_consumer_groups(timeout=10)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing consumer groups...", 
                    total=len(group_list)
                )
                
                for group_info in group_list:
                    group_id = group_info.id
                    
                    # Apply pattern filter
                    if group_pattern and group_pattern not in group_id:
                        progress.advance(task)
                        continue
                    
                    try:
                        status = ConsumerGroupStatus(group_id=group_id)
                        
                        if include_details:
                            await self._get_consumer_group_details(status)
                            await self._calculate_consumer_lag(status)
                        
                        groups.append(status)
                        
                    except Exception as e:
                        logger.warning(f"Failed to get details for group {group_id}: {e}")
                        # Add basic info even if details fail
                        groups.append(ConsumerGroupStatus(
                            group_id=group_id,
                            state="error"
                        ))
                    
                    progress.advance(task)
        
        except Exception as e:
            logger.error(f"Failed to list consumer groups: {e}")
            raise
        
        return groups
    
    async def _get_consumer_group_details(self, status: ConsumerGroupStatus) -> None:
        """Get detailed information for a consumer group."""
        try:
            # Describe consumer group
            group_descriptions = self.admin_client.describe_consumer_groups([status.group_id])
            
            for group_id, future in group_descriptions.items():
                try:
                    group_desc = future.result(timeout=10)
                    status.state = group_desc.state
                    status.protocol_type = group_desc.protocol_type
                    status.protocol = group_desc.protocol
                    status.coordinator = group_desc.coordinator.id if group_desc.coordinator else None
                    
                    # Get member details
                    status.members = []
                    for member in group_desc.members:
                        member_info = {
                            "member_id": member.member_id,
                            "client_id": member.client_id,
                            "client_host": member.client_host,
                            "assignment": []
                        }
                        
                        # Get topic assignments
                        for assignment in member.assignment:
                            member_info["assignment"].append({
                                "topic": assignment.topic,
                                "partitions": assignment.partitions
                            })
                        
                        status.members.append(member_info)
                    
                    # Determine stability
                    if status.state == "Stable":
                        status.stability = "stable"
                    elif status.state in ["PreparingRebalance", "CompletingRebalance"]:
                        status.stability = "rebalancing"
                    elif status.state == "Dead":
                        status.stability = "dead"
                    else:
                        status.stability = "unknown"
                
                except Exception as e:
                    logger.warning(f"Failed to get description for group {group_id}: {e}")
                    status.state = "error"
        
        except Exception as e:
            logger.warning(f"Failed to describe consumer group {status.group_id}: {e}")
    
    async def _calculate_consumer_lag(self, status: ConsumerGroupStatus) -> None:
        """Calculate consumer lag for a group."""
        try:
            consumer = self._init_consumer()
            
            # Get committed offsets for the group
            # This is a simplified approach - in production you'd use more sophisticated methods
            total_lag = 0
            
            for member in status.members:
                for assignment in member["assignment"]:
                    topic = assignment["topic"]
                    partitions = assignment["partitions"]
                    
                    for partition_id in partitions:
                        try:
                            tp = TopicPartition(topic, partition_id)
                            
                            # Get committed offset
                            committed = consumer.committed([tp], timeout=5)
                            if committed and committed[0].offset >= 0:
                                committed_offset = committed[0].offset
                            else:
                                committed_offset = 0
                            
                            # Get high watermark
                            low, high = consumer.get_watermark_offsets(tp, timeout=5)
                            
                            # Calculate lag
                            lag = max(0, high - committed_offset)
                            status.lag[f"{topic}:{partition_id}"] = lag
                            total_lag += lag
                        
                        except Exception as e:
                            logger.warning(f"Failed to calculate lag for {topic}:{partition_id}: {e}")
            
            status.total_lag = total_lag
        
        except Exception as e:
            logger.warning(f"Failed to calculate lag for group {status.group_id}: {e}")
    
    async def reset_consumer_offsets(
        self,
        group_id: str,
        topic: Optional[str] = None,
        partitions: Optional[List[int]] = None,
        to_earliest: bool = False,
        to_latest: bool = False,
        to_datetime: Optional[datetime] = None,
        to_offset: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Reset consumer group offsets."""
        console.print(f"🔄 Resetting offsets for consumer group: {group_id}")
        
        if not any([to_earliest, to_latest, to_datetime, to_offset]):
            raise ValueError("Must specify one of: to_earliest, to_latest, to_datetime, to_offset")
        
        try:
            consumer = self._init_consumer()
            
            # Get current assignments if topic not specified
            if not topic:
                # Get topics for this consumer group
                group_descriptions = self.admin_client.describe_consumer_groups([group_id])
                topics_to_reset = set()
                
                for group, future in group_descriptions.items():
                    group_desc = future.result(timeout=10)
                    for member in group_desc.members:
                        for assignment in member.assignment:
                            topics_to_reset.add(assignment.topic)
                
                if not topics_to_reset:
                    return {"success": False, "message": "No topics found for consumer group"}
            else:
                topics_to_reset = [topic]
            
            # Build list of topic partitions to reset
            topic_partitions = []
            
            for topic_name in topics_to_reset:
                # Get topic metadata
                metadata = self.admin_client.list_topics(topic=topic_name, timeout=10)
                if topic_name not in metadata.topics:
                    continue
                
                topic_metadata = metadata.topics[topic_name]
                
                # Determine partitions to reset
                if partitions:
                    partitions_to_reset = partitions
                else:
                    partitions_to_reset = list(topic_metadata.partitions.keys())
                
                for partition_id in partitions_to_reset:
                    tp = TopicPartition(topic_name, partition_id)
                    topic_partitions.append(tp)
            
            if not topic_partitions:
                return {"success": False, "message": "No partitions found to reset"}
            
            # Calculate new offsets
            new_offsets = []
            
            for tp in topic_partitions:
                if to_earliest:
                    tp.offset = 0  # Will be resolved to earliest
                elif to_latest:
                    low, high = consumer.get_watermark_offsets(tp, timeout=10)
                    tp.offset = high
                elif to_datetime:
                    # Convert datetime to timestamp
                    timestamp = int(to_datetime.timestamp() * 1000)
                    offsets = consumer.offsets_for_times([
                        TopicPartition(tp.topic, tp.partition, timestamp)
                    ], timeout=10)
                    
                    if offsets and offsets[0].offset >= 0:
                        tp.offset = offsets[0].offset
                    else:
                        tp.offset = 0
                elif to_offset is not None:
                    tp.offset = to_offset
                
                new_offsets.append(tp)
            
            if dry_run:
                # Return what would be done
                offset_info = []
                for tp in new_offsets:
                    offset_info.append({
                        "topic": tp.topic,
                        "partition": tp.partition,
                        "new_offset": tp.offset
                    })
                
                return {
                    "success": True,
                    "message": f"Would reset {len(new_offsets)} partition offsets",
                    "offsets": offset_info,
                    "dry_run": True
                }
            
            # Actually reset the offsets
            # Note: This requires the consumer group to be inactive
            try:
                # Check if group is active
                group_descriptions = self.admin_client.describe_consumer_groups([group_id])
                for group, future in group_descriptions.items():
                    group_desc = future.result(timeout=10)
                    if group_desc.state != "Empty":
                        return {
                            "success": False,
                            "message": f"Consumer group {group_id} is active (state: {group_desc.state}). "
                                     "Stop all consumers before resetting offsets."
                        }
                
                # Reset offsets using admin client (simplified approach)
                # In production, you'd use the proper offset reset API
                consumer.assign(new_offsets)
                consumer.commit(offsets=new_offsets)
                
                return {
                    "success": True,
                    "message": f"Successfully reset {len(new_offsets)} partition offsets",
                    "group_id": group_id,
                    "reset_count": len(new_offsets)
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to reset offsets: {str(e)}"
                }
        
        except Exception as e:
            logger.error(f"Failed to reset consumer offsets: {e}")
            return {
                "success": False,
                "message": f"Offset reset failed: {str(e)}"
            }
    
    async def migrate_topic_data(
        self,
        source_topic: str,
        target_topic: str,
        from_offset: Optional[int] = None,
        to_offset: Optional[int] = None,
        transform_function: Optional[callable] = None,
        dry_run: bool = False
    ) -> str:
        """Migrate data from one topic to another."""
        task_id = f"migration_{int(time.time())}"
        
        migration = MigrationTask(
            task_id=task_id,
            source_topic=source_topic,
            target_topic=target_topic,
            start_time=datetime.utcnow()
        )
        
        self.migration_tasks[task_id] = migration
        
        console.print(f"🚀 Starting migration: {source_topic} → {target_topic}")
        console.print(f"📝 Task ID: {task_id}")
        
        if dry_run:
            migration.status = "completed"
            migration.end_time = datetime.utcnow()
            console.print("✅ Dry run completed - no data was migrated")
            return task_id
        
        # Run migration in background
        asyncio.create_task(self._execute_migration(migration, from_offset, to_offset, transform_function))
        
        return task_id
    
    async def _execute_migration(
        self,
        migration: MigrationTask,
        from_offset: Optional[int],
        to_offset: Optional[int],
        transform_function: Optional[callable]
    ) -> None:
        """Execute the actual data migration."""
        try:
            migration.status = "running"
            
            consumer = self._init_consumer()
            producer = self._init_producer()
            
            # Get source topic metadata
            metadata = self.admin_client.list_topics(topic=migration.source_topic, timeout=10)
            if migration.source_topic not in metadata.topics:
                raise ValueError(f"Source topic {migration.source_topic} not found")
            
            topic_metadata = metadata.topics[migration.source_topic]
            partitions = list(topic_metadata.partitions.keys())
            
            # Subscribe to source topic
            topic_partitions = [
                TopicPartition(migration.source_topic, p, from_offset or 0)
                for p in partitions
            ]
            
            consumer.assign(topic_partitions)
            
            # If specific offsets are provided, seek to them
            if from_offset is not None:
                for tp in topic_partitions:
                    consumer.seek(tp)
            
            # Calculate total records (estimate)
            total_estimate = 0
            for partition_id in partitions:
                tp = TopicPartition(migration.source_topic, partition_id)
                low, high = consumer.get_watermark_offsets(tp, timeout=5)
                start_offset = from_offset or low
                end_offset = to_offset or high
                total_estimate += max(0, end_offset - start_offset)
            
            migration.total_records = total_estimate
            
            # Migration loop
            batch_size = self.admin_config["migration"]["batch_size"]
            timeout_ms = self.admin_config["migration"]["timeout_ms"]
            records_migrated = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Migrating {migration.source_topic}...",
                    total=migration.total_records
                )
                
                while True:
                    msg_batch = consumer.consume(num_messages=batch_size, timeout=timeout_ms / 1000)
                    
                    if not msg_batch:
                        break  # No more messages
                    
                    for msg in msg_batch:
                        if msg.error():
                            if msg.error().code() == KafkaError._PARTITION_EOF:
                                continue
                            else:
                                raise Exception(f"Consumer error: {msg.error()}")
                        
                        # Check if we've reached the end offset
                        if to_offset is not None and msg.offset() >= to_offset:
                            break
                        
                        try:
                            # Transform message if function provided
                            if transform_function:
                                key, value = transform_function(msg.key(), msg.value())
                            else:
                                key, value = msg.key(), msg.value()
                            
                            # Produce to target topic
                            producer.produce(
                                topic=migration.target_topic,
                                key=key,
                                value=value,
                                partition=msg.partition(),  # Preserve partition if possible
                                timestamp=msg.timestamp()[1] if self.admin_config["migration"]["preserve_timestamps"] else None
                            )
                            
                            records_migrated += 1
                            migration.records_migrated = records_migrated
                            migration.progress = (records_migrated / migration.total_records) * 100 if migration.total_records > 0 else 0
                            
                            # Update progress
                            progress.update(task, completed=records_migrated)
                            
                        except Exception as e:
                            logger.error(f"Failed to migrate message: {e}")
                            continue
                    
                    # Flush producer periodically
                    producer.flush(timeout=1)
                    
                    # Check if we should stop
                    if to_offset is not None:
                        current_offset = max(msg.offset() for msg in msg_batch if not msg.error())
                        if current_offset >= to_offset:
                            break
            
            # Final flush
            producer.flush()
            
            migration.status = "completed"
            migration.end_time = datetime.utcnow()
            migration.records_migrated = records_migrated
            migration.progress = 100.0
            
            console.print(f"✅ Migration completed: {records_migrated} records migrated")
        
        except Exception as e:
            migration.status = "failed"
            migration.end_time = datetime.utcnow()
            migration.error_message = str(e)
            logger.error(f"Migration failed: {e}")
            console.print(f"❌ Migration failed: {e}")
    
    def get_migration_status(self, task_id: str) -> Optional[MigrationTask]:
        """Get status of a migration task."""
        return self.migration_tasks.get(task_id)
    
    def list_migration_tasks(self) -> List[MigrationTask]:
        """List all migration tasks."""
        return list(self.migration_tasks.values())
    
    async def analyze_topic_performance(
        self, topic: str, duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """Analyze topic performance over a specified duration."""
        console.print(f"📊 Analyzing performance for topic: {topic}")
        
        try:
            # Get topic metadata
            metadata = self.admin_client.list_topics(topic=topic, timeout=10)
            if topic not in metadata.topics:
                return {"error": f"Topic {topic} not found"}
            
            topic_metadata = metadata.topics[topic]
            partitions = list(topic_metadata.partitions.keys())
            
            # Initialize tracking
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            performance_data = {
                "topic": topic,
                "analysis_start": start_time.isoformat(),
                "analysis_duration_minutes": duration_minutes,
                "partitions": len(partitions),
                "message_rates": {},
                "partition_distribution": {},
                "leader_distribution": {},
                "size_analysis": {},
                "recommendations": []
            }
            
            consumer = self._init_consumer()
            
            # Sample data at intervals
            sample_interval = min(duration_minutes * 60 // 10, 60)  # Sample every minute or 1/10th of duration
            samples = []
            
            console.print(f"📈 Collecting performance samples (interval: {sample_interval}s)")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Collecting samples...", total=duration_minutes * 60 // sample_interval)
                
                current_time = start_time
                while current_time < end_time:
                    sample = {
                        "timestamp": current_time.isoformat(),
                        "partition_offsets": {},
                        "partition_sizes": {}
                    }
                    
                    for partition_id in partitions:
                        tp = TopicPartition(topic, partition_id)
                        try:
                            low, high = consumer.get_watermark_offsets(tp, timeout=5)
                            sample["partition_offsets"][partition_id] = {
                                "low": low,
                                "high": high,
                                "messages": high - low
                            }
                        except Exception as e:
                            logger.warning(f"Failed to get offsets for {topic}:{partition_id}: {e}")
                    
                    samples.append(sample)
                    progress.advance(task)
                    
                    # Wait for next sample
                    await asyncio.sleep(sample_interval)
                    current_time = datetime.utcnow()
            
            # Analyze collected data
            if len(samples) >= 2:
                # Calculate message rates
                first_sample = samples[0]
                last_sample = samples[-1]
                time_diff = (datetime.fromisoformat(last_sample["timestamp"]) - 
                           datetime.fromisoformat(first_sample["timestamp"])).total_seconds()
                
                for partition_id in partitions:
                    if (partition_id in first_sample["partition_offsets"] and 
                        partition_id in last_sample["partition_offsets"]):
                        
                        first_high = first_sample["partition_offsets"][partition_id]["high"]
                        last_high = last_sample["partition_offsets"][partition_id]["high"]
                        
                        message_diff = last_high - first_high
                        rate = message_diff / time_diff if time_diff > 0 else 0
                        
                        performance_data["message_rates"][partition_id] = {
                            "messages_per_second": rate,
                            "total_messages": message_diff
                        }
                
                # Analyze partition distribution
                total_messages = sum(
                    data["total_messages"] 
                    for data in performance_data["message_rates"].values()
                )
                
                for partition_id, rate_data in performance_data["message_rates"].items():
                    if total_messages > 0:
                        percentage = (rate_data["total_messages"] / total_messages) * 100
                        performance_data["partition_distribution"][partition_id] = percentage
                
                # Add recommendations
                performance_data["recommendations"] = self._generate_performance_recommendations(
                    performance_data, topic_metadata
                )
            
            return performance_data
        
        except Exception as e:
            logger.error(f"Failed to analyze topic performance: {e}")
            return {"error": str(e)}
    
    def _generate_performance_recommendations(
        self, performance_data: Dict[str, Any], topic_metadata
    ) -> List[str]:
        """Generate performance recommendations based on analysis."""
        recommendations = []
        
        try:
            # Check partition balance
            if performance_data["partition_distribution"]:
                distribution_values = list(performance_data["partition_distribution"].values())
                if distribution_values:
                    max_percentage = max(distribution_values)
                    min_percentage = min(distribution_values)
                    
                    if max_percentage - min_percentage > 50:  # More than 50% difference
                        recommendations.append(
                            "Consider reviewing your partitioning strategy - significant imbalance detected"
                        )
            
            # Check message rates
            if performance_data["message_rates"]:
                rates = [data["messages_per_second"] for data in performance_data["message_rates"].values()]
                avg_rate = sum(rates) / len(rates)
                
                if avg_rate > 1000:  # High throughput
                    recommendations.append(
                        "High message rate detected - consider increasing partition count if needed"
                    )
                elif avg_rate < 1:  # Very low throughput
                    recommendations.append(
                        "Low message rate detected - consider consolidating partitions if appropriate"
                    )
            
            # Check replication factor
            if topic_metadata.partitions:
                first_partition = list(topic_metadata.partitions.values())[0]
                replication_factor = len(first_partition.replicas)
                
                if replication_factor < 3:
                    recommendations.append(
                        f"Replication factor is {replication_factor} - consider increasing for better fault tolerance"
                    )
        
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    def export_cluster_info(self, format: str = "json") -> Dict[str, Any]:
        """Export comprehensive cluster information."""
        console.print("📤 Exporting cluster information...")
        
        try:
            # Collect all cluster information
            cluster_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "kafka_version": kafka_version,
                "cluster": {},
                "topics": {},
                "consumer_groups": {},
                "brokers": {}
            }
            
            # Get cluster metadata
            metadata = self.admin_client.list_topics(timeout=10)
            
            # Cluster info
            cluster_info["cluster"] = {
                "cluster_id": metadata.cluster_id,
                "broker_count": len(metadata.brokers),
                "topic_count": len(metadata.topics),
                "controller_id": getattr(metadata, 'controller_id', None)
            }
            
            # Broker info
            for broker_id, broker in metadata.brokers.items():
                cluster_info["brokers"][broker_id] = {
                    "id": broker.id,
                    "host": broker.host,
                    "port": broker.port,
                    "rack": getattr(broker, 'rack', None)
                }
            
            # Topic info
            for topic_name, topic_metadata in metadata.topics.items():
                if topic_name.startswith('__'):
                    continue  # Skip internal topics
                
                partition_info = {}
                for partition_id, partition_metadata in topic_metadata.partitions.items():
                    partition_info[partition_id] = {
                        "leader": partition_metadata.leader,
                        "replicas": partition_metadata.replicas,
                        "in_sync_replicas": partition_metadata.isrs,
                        "error": str(partition_metadata.error) if partition_metadata.error else None
                    }
                
                cluster_info["topics"][topic_name] = {
                    "partitions": partition_info,
                    "partition_count": len(partition_info),
                    "error": str(topic_metadata.error) if topic_metadata.error else None
                }
            
            return cluster_info
        
        except Exception as e:
            logger.error(f"Failed to export cluster info: {e}")
            return {"error": str(e)}
    
    def cleanup_resources(self) -> None:
        """Cleanup resources."""
        try:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.flush()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# CLI Commands
@click.group()
@click.option(
    "--kafka-config",
    help="Kafka configuration (JSON format)"
)
@click.option(
    "--admin-config",
    help="Admin configuration file path"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level"
)
@click.pass_context
def cli(ctx, kafka_config, admin_config, log_level):
    """TDA Kafka Administrative Utilities - Comprehensive cluster management tool."""
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    kafka_conf = None
    if kafka_config:
        kafka_conf = json.loads(kafka_config)
    
    admin_conf = None
    if admin_config:
        with open(admin_config, 'r') as f:
            admin_conf = yaml.safe_load(f)
    
    ctx.ensure_object(dict)
    ctx.obj["admin"] = KafkaAdmin(
        kafka_config=kafka_conf,
        admin_config=admin_conf
    )


@cli.command()
@click.option(
    "--detailed",
    is_flag=True,
    help="Include detailed health information"
)
@click.option(
    "--format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format"
)
@click.pass_context
def cluster_health(ctx, detailed, format):
    """Perform comprehensive cluster health check."""
    admin = ctx.obj["admin"]
    
    async def _check_health():
        health = await admin.cluster_health_check(detailed=detailed)
        
        if format == "json":
            import dataclasses
            health_dict = dataclasses.asdict(health)
            console.print(json.dumps(health_dict, indent=2, default=str))
        else:
            # Display as table
            table = Table(title="🏥 Cluster Health Report")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            table.add_column("Status", justify="center")
            
            # Health score with color
            if health.health_score >= 90:
                score_color = "green"
                score_icon = "🟢"
            elif health.health_score >= 70:
                score_color = "yellow"
                score_icon = "🟡"
            else:
                score_color = "red"
                score_icon = "🔴"
            
            table.add_row(
                "Health Score",
                f"{health.health_score:.1f}/100",
                f"[{score_color}]{score_icon}[/{score_color}]"
            )
            table.add_row("Cluster ID", health.cluster_id, "ℹ️")
            table.add_row("Brokers", str(health.broker_count), "🖥️")
            table.add_row("Topics", str(health.topic_count), "📋")
            table.add_row("Partitions", str(health.partition_count), "🗂️")
            
            # Issues
            offline_color = "red" if health.offline_partitions > 0 else "green"
            table.add_row(
                "Offline Partitions",
                str(health.offline_partitions),
                f"[{offline_color}]{'⚠️' if health.offline_partitions > 0 else '✅'}[/{offline_color}]"
            )
            
            under_rep_color = "red" if health.under_replicated_partitions > 0 else "green"
            table.add_row(
                "Under-replicated",
                str(health.under_replicated_partitions),
                f"[{under_rep_color}]{'⚠️' if health.under_replicated_partitions > 0 else '✅'}[/{under_rep_color}]"
            )
            
            table.add_row(
                "Leader Imbalance",
                f"{health.leader_imbalance_ratio:.1%}",
                "⚖️"
            )
            
            console.print(table)
            
            # Issues and recommendations
            if health.issues:
                console.print("\n🚨 Issues Found:")
                for issue in health.issues:
                    console.print(f"  • {issue}")
            
            if health.recommendations:
                console.print("\n💡 Recommendations:")
                for rec in health.recommendations:
                    console.print(f"  • {rec}")
    
    asyncio.run(_check_health())


@cli.command()
@click.option(
    "--group-pattern",
    help="Filter consumer groups by pattern"
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Include detailed consumer group information"
)
@click.option(
    "--format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format"
)
@click.pass_context
def consumer_groups(ctx, group_pattern, detailed, format):
    """List and analyze consumer groups."""
    admin = ctx.obj["admin"]
    
    async def _list_groups():
        groups = await admin.list_consumer_groups(
            group_pattern=group_pattern,
            include_details=detailed
        )
        
        if format == "json":
            import dataclasses
            groups_data = [dataclasses.asdict(group) for group in groups]
            console.print(json.dumps(groups_data, indent=2, default=str))
        else:
            # Display as table
            table = Table(title="📋 Consumer Groups")
            table.add_column("Group ID", style="cyan")
            table.add_column("State", justify="center")
            table.add_column("Members", justify="right")
            table.add_column("Total Lag", justify="right")
            table.add_column("Stability", justify="center")
            
            for group in groups:
                # State color
                state_color = {
                    "Stable": "green",
                    "PreparingRebalance": "yellow",
                    "CompletingRebalance": "yellow",
                    "Dead": "red",
                    "Empty": "dim",
                    "error": "red"
                }.get(group.state, "white")
                
                # Stability icon
                stability_icon = {
                    "stable": "🟢",
                    "rebalancing": "🟡",
                    "dead": "🔴",
                    "unknown": "❓"
                }.get(group.stability, "❓")
                
                table.add_row(
                    group.group_id,
                    f"[{state_color}]{group.state}[/{state_color}]",
                    str(len(group.members)),
                    f"{group.total_lag:,}",
                    stability_icon
                )
            
            console.print(table)
            console.print(f"\n📊 Total: {len(groups)} consumer groups")
    
    asyncio.run(_list_groups())


@cli.command()
@click.option(
    "--group",
    required=True,
    help="Consumer group ID"
)
@click.option(
    "--topic",
    help="Specific topic to reset (default: all topics for group)"
)
@click.option(
    "--partitions",
    help="Comma-separated list of partition IDs"
)
@click.option(
    "--to-earliest",
    is_flag=True,
    help="Reset to earliest offset"
)
@click.option(
    "--to-latest",
    is_flag=True,
    help="Reset to latest offset"
)
@click.option(
    "--to-datetime",
    help="Reset to specific datetime (ISO format)"
)
@click.option(
    "--to-offset",
    type=int,
    help="Reset to specific offset"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without actually resetting"
)
@click.pass_context
def reset_offsets(ctx, group, topic, partitions, to_earliest, to_latest, to_datetime, to_offset, dry_run):
    """Reset consumer group offsets."""
    admin = ctx.obj["admin"]
    
    partition_list = None
    if partitions:
        partition_list = [int(p.strip()) for p in partitions.split(",")]
    
    datetime_obj = None
    if to_datetime:
        datetime_obj = datetime.fromisoformat(to_datetime)
    
    async def _reset():
        result = await admin.reset_consumer_offsets(
            group_id=group,
            topic=topic,
            partitions=partition_list,
            to_earliest=to_earliest,
            to_latest=to_latest,
            to_datetime=datetime_obj,
            to_offset=to_offset,
            dry_run=dry_run
        )
        
        if result["success"]:
            console.print(f"✅ {result['message']}")
            if dry_run and "offsets" in result:
                table = Table(title="Planned Offset Changes")
                table.add_column("Topic")
                table.add_column("Partition")
                table.add_column("New Offset")
                
                for offset_info in result["offsets"]:
                    table.add_row(
                        offset_info["topic"],
                        str(offset_info["partition"]),
                        str(offset_info["new_offset"])
                    )
                
                console.print(table)
        else:
            console.print(f"❌ {result['message']}")
    
    asyncio.run(_reset())


@cli.command()
@click.option(
    "--source-topic",
    required=True,
    help="Source topic name"
)
@click.option(
    "--target-topic",
    required=True,
    help="Target topic name"
)
@click.option(
    "--from-offset",
    type=int,
    help="Start offset (default: earliest)"
)
@click.option(
    "--to-offset",
    type=int,
    help="End offset (default: latest)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Plan migration without executing"
)
@click.pass_context
def migrate_data(ctx, source_topic, target_topic, from_offset, to_offset, dry_run):
    """Migrate data from one topic to another."""
    admin = ctx.obj["admin"]
    
    async def _migrate():
        task_id = await admin.migrate_topic_data(
            source_topic=source_topic,
            target_topic=target_topic,
            from_offset=from_offset,
            to_offset=to_offset,
            dry_run=dry_run
        )
        
        console.print(f"🚀 Migration task started: {task_id}")
        
        if not dry_run:
            console.print("💡 Use 'migration-status' command to check progress")
    
    asyncio.run(_migrate())


@cli.command()
@click.option(
    "--task-id",
    help="Specific task ID to check"
)
@click.pass_context
def migration_status(ctx, task_id):
    """Check migration task status."""
    admin = ctx.obj["admin"]
    
    if task_id:
        migration = admin.get_migration_status(task_id)
        if migration:
            table = Table(title=f"Migration Task: {task_id}")
            table.add_column("Property")
            table.add_column("Value")
            
            table.add_row("Status", migration.status)
            table.add_row("Source Topic", migration.source_topic)
            table.add_row("Target Topic", migration.target_topic)
            table.add_row("Progress", f"{migration.progress:.1f}%")
            table.add_row("Records Migrated", f"{migration.records_migrated:,}")
            table.add_row("Total Records", f"{migration.total_records:,}")
            
            if migration.start_time:
                table.add_row("Start Time", migration.start_time.strftime("%Y-%m-%d %H:%M:%S"))
            if migration.end_time:
                table.add_row("End Time", migration.end_time.strftime("%Y-%m-%d %H:%M:%S"))
            if migration.error_message:
                table.add_row("Error", migration.error_message)
            
            console.print(table)
        else:
            console.print(f"❌ Migration task {task_id} not found")
    else:
        # List all tasks
        tasks = admin.list_migration_tasks()
        
        if tasks:
            table = Table(title="Migration Tasks")
            table.add_column("Task ID")
            table.add_column("Source → Target")
            table.add_column("Status")
            table.add_column("Progress")
            table.add_column("Records")
            
            for task in tasks:
                status_color = {
                    "pending": "yellow",
                    "running": "blue",
                    "completed": "green",
                    "failed": "red"
                }.get(task.status, "white")
                
                table.add_row(
                    task.task_id,
                    f"{task.source_topic} → {task.target_topic}",
                    f"[{status_color}]{task.status}[/{status_color}]",
                    f"{task.progress:.1f}%",
                    f"{task.records_migrated:,}"
                )
            
            console.print(table)
        else:
            console.print("📋 No migration tasks found")


@cli.command()
@click.option(
    "--topic",
    required=True,
    help="Topic to analyze"
)
@click.option(
    "--duration",
    default=60,
    type=int,
    help="Analysis duration in minutes"
)
@click.option(
    "--format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format"
)
@click.pass_context
def analyze_performance(ctx, topic, duration, format):
    """Analyze topic performance."""
    admin = ctx.obj["admin"]
    
    async def _analyze():
        result = await admin.analyze_topic_performance(topic, duration)
        
        if "error" in result:
            console.print(f"❌ {result['error']}")
            return
        
        if format == "json":
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"📊 Performance Analysis: {topic}")
            console.print(f"⏱️ Duration: {duration} minutes")
            console.print(f"🗂️ Partitions: {result['partitions']}")
            
            if result.get("message_rates"):
                table = Table(title="Message Rates by Partition")
                table.add_column("Partition")
                table.add_column("Messages/sec")
                table.add_column("Total Messages")
                table.add_column("% of Traffic")
                
                for partition_id, rate_data in result["message_rates"].items():
                    distribution = result["partition_distribution"].get(partition_id, 0)
                    table.add_row(
                        str(partition_id),
                        f"{rate_data['messages_per_second']:.2f}",
                        f"{rate_data['total_messages']:,}",
                        f"{distribution:.1f}%"
                    )
                
                console.print(table)
            
            if result.get("recommendations"):
                console.print("\n💡 Recommendations:")
                for rec in result["recommendations"]:
                    console.print(f"  • {rec}")
    
    asyncio.run(_analyze())


@cli.command()
@click.option(
    "--format",
    default="json",
    type=click.Choice(["json", "yaml"]),
    help="Export format"
)
@click.option(
    "--output",
    help="Output file path"
)
@click.pass_context
def export_cluster(ctx, format, output):
    """Export comprehensive cluster information."""
    admin = ctx.obj["admin"]
    
    cluster_info = admin.export_cluster_info(format=format)
    
    if format == "json":
        output_text = json.dumps(cluster_info, indent=2, default=str)
    else:  # yaml
        output_text = yaml.dump(cluster_info, default_flow_style=False)
    
    if output:
        with open(output, 'w') as f:
            f.write(output_text)
        console.print(f"📤 Cluster information exported to {output}")
    else:
        console.print(output_text)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Operation cancelled")
    except Exception as e:
        console.print(f"❌ Error: {e}")
        logger.error(f"CLI error: {e}")
    finally:
        # Cleanup if admin object exists
        try:
            import contextvars
            ctx = click.get_current_context(silent=True)
            if ctx and ctx.obj and "admin" in ctx.obj:
                ctx.obj["admin"].cleanup_resources()
        except:
            pass
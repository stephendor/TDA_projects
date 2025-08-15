#!/usr/bin/env python3
"""
TDA Kafka Topic Manager

Advanced topic management script for creating, updating, deleting, and validating
Kafka topics based on YAML configuration with schema registry integration.

Features:
- Create/update/delete topics from YAML config
- Validate topic configurations
- Perform topic health checks
- Handle schema registry integration
- Support for batch operations
- Environment-specific configurations
- Comprehensive error handling and logging

Usage:
    python topic-manager.py create --env development
    python topic-manager.py update --topic tda_jobs --env production
    python topic-manager.py validate --config topics.yml
    python topic-manager.py health-check --all
    python topic-manager.py delete --topic tda_test --confirm
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import click
import yaml
from confluent_kafka import KafkaError
from confluent_kafka.admin import (
    AdminClient,
    ConfigResource,
    NewTopic,
    TopicMetadata,
)
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("topic-manager.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
console = Console()


class TopicConfig(BaseModel):
    """Topic configuration model with validation."""
    
    description: str
    partitions: int = Field(ge=1, le=1000)
    config: Dict[str, Union[str, int]] = Field(default_factory=dict)
    schema: Optional[Dict[str, Any]] = None
    event_types: List[str] = Field(default_factory=list)
    environment_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class TopicsConfiguration(BaseModel):
    """Complete topics configuration model."""
    
    version: str
    metadata: Dict[str, Any]
    defaults: Dict[str, Union[str, int]]
    environments: Dict[str, Dict[str, Any]]
    topics: Dict[str, TopicConfig]
    schemas: Optional[Dict[str, Dict[str, Any]]] = None
    monitoring: Optional[Dict[str, Any]] = None
    retention_policies: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None


@dataclass
class TopicStatus:
    """Topic status information."""
    
    name: str
    exists: bool = False
    partitions: int = 0
    replication_factor: int = 0
    config: Dict[str, str] = field(default_factory=dict)
    health: str = "unknown"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class OperationResult:
    """Result of a topic operation."""
    
    success: bool
    message: str
    topic_name: Optional[str] = None
    operation: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class TopicManager:
    """Advanced Kafka topic manager with comprehensive functionality."""
    
    def __init__(
        self,
        config_path: str = "config/topics.yml",
        kafka_config: Optional[Dict[str, str]] = None,
        schema_registry_url: Optional[str] = None,
    ):
        """Initialize the topic manager."""
        self.config_path = Path(config_path)
        self.kafka_config = kafka_config or self._get_default_kafka_config()
        self.schema_registry_url = schema_registry_url or os.getenv(
            "SCHEMA_REGISTRY_URL", "http://localhost:8081"
        )
        
        # Initialize clients
        self.admin_client = AdminClient(self.kafka_config)
        self.schema_registry_client = None
        if self.schema_registry_url:
            try:
                self.schema_registry_client = SchemaRegistryClient({
                    "url": self.schema_registry_url
                })
            except Exception as e:
                logger.warning(f"Failed to connect to schema registry: {e}")
        
        # Load configuration
        self.topics_config = self._load_configuration()
        
    def _get_default_kafka_config(self) -> Dict[str, str]:
        """Get default Kafka configuration from environment."""
        return {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "client.id": "tda-topic-manager",
            "request.timeout.ms": "30000",
            "api.version.request": "true",
        }
    
    def _load_configuration(self) -> TopicsConfiguration:
        """Load and validate topics configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Replace environment variables
            config_data = self._substitute_env_vars(config_data)
            
            return TopicsConfiguration(**config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            # Extract variable name and default value
            var_expr = data[2:-1]
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
            else:
                var_name, default_value = var_expr, ""
            return os.getenv(var_name, default_value)
        else:
            return data
    
    def _get_topic_config_for_env(
        self, topic_name: str, environment: str
    ) -> Dict[str, Any]:
        """Get topic configuration for specific environment."""
        topic_config = self.topics_config.topics[topic_name]
        
        # Start with global defaults
        config = self.topics_config.defaults.copy()
        
        # Apply environment defaults
        if environment in self.topics_config.environments:
            env_defaults = self.topics_config.environments[environment].get("defaults", {})
            config.update(env_defaults)
        
        # Apply topic-specific config
        config.update(topic_config.config)
        
        # Apply environment-specific overrides
        if environment in topic_config.environment_overrides:
            env_overrides = topic_config.environment_overrides[environment]
            if "config" in env_overrides:
                config.update(env_overrides["config"])
        
        return config
    
    def _get_topic_partitions_for_env(
        self, topic_name: str, environment: str
    ) -> int:
        """Get topic partition count for specific environment."""
        topic_config = self.topics_config.topics[topic_name]
        
        # Check environment overrides first
        if environment in topic_config.environment_overrides:
            env_overrides = topic_config.environment_overrides[environment]
            if "partitions" in env_overrides:
                return env_overrides["partitions"]
        
        return topic_config.partitions
    
    async def create_topics(
        self, 
        environment: str = "development",
        topics: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> List[OperationResult]:
        """Create topics based on configuration."""
        results = []
        topics_to_create = topics or list(self.topics_config.topics.keys())
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Creating topics...", total=len(topics_to_create))
            
            for topic_name in topics_to_create:
                progress.update(task, description=f"Creating {topic_name}")
                
                try:
                    result = await self._create_single_topic(
                        topic_name, environment, dry_run
                    )
                    results.append(result)
                    
                    if result.success:
                        console.print(f"✅ {topic_name}: {result.message}")
                    else:
                        console.print(f"❌ {topic_name}: {result.message}")
                        
                except Exception as e:
                    error_result = OperationResult(
                        success=False,
                        message=f"Failed to create topic: {str(e)}",
                        topic_name=topic_name,
                        operation="create",
                        errors=[str(e)]
                    )
                    results.append(error_result)
                    console.print(f"❌ {topic_name}: {str(e)}")
                
                progress.advance(task)
        
        return results
    
    async def _create_single_topic(
        self, topic_name: str, environment: str, dry_run: bool = False
    ) -> OperationResult:
        """Create a single topic."""
        if topic_name not in self.topics_config.topics:
            return OperationResult(
                success=False,
                message=f"Topic {topic_name} not found in configuration",
                topic_name=topic_name,
                operation="create"
            )
        
        # Get topic configuration for environment
        config = self._get_topic_config_for_env(topic_name, environment)
        partitions = self._get_topic_partitions_for_env(topic_name, environment)
        replication_factor = config.get("replication_factor", 1)
        
        # Check if topic already exists
        existing_topics = await self._get_existing_topics()
        if topic_name in existing_topics:
            return OperationResult(
                success=False,
                message=f"Topic {topic_name} already exists",
                topic_name=topic_name,
                operation="create"
            )
        
        if dry_run:
            return OperationResult(
                success=True,
                message=f"Would create topic with {partitions} partitions, "
                       f"replication factor {replication_factor}",
                topic_name=topic_name,
                operation="create",
                details={
                    "partitions": partitions,
                    "replication_factor": replication_factor,
                    "config": config
                }
            )
        
        # Create new topic
        new_topic = NewTopic(
            topic=topic_name,
            num_partitions=partitions,
            replication_factor=replication_factor,
            config={str(k): str(v) for k, v in config.items()}
        )
        
        try:
            # Create topic
            futures = self.admin_client.create_topics([new_topic])
            
            # Wait for operation to complete
            for topic, future in futures.items():
                try:
                    future.result(timeout=30)
                    logger.info(f"Topic {topic} created successfully")
                except KafkaError as e:
                    if e.code() == KafkaError.TOPIC_ALREADY_EXISTS:
                        return OperationResult(
                            success=False,
                            message=f"Topic {topic_name} already exists",
                            topic_name=topic_name,
                            operation="create"
                        )
                    else:
                        raise e
            
            # Register schema if specified
            schema_result = await self._register_schema(topic_name)
            
            return OperationResult(
                success=True,
                message=f"Topic created with {partitions} partitions, "
                       f"replication factor {replication_factor}",
                topic_name=topic_name,
                operation="create",
                details={
                    "partitions": partitions,
                    "replication_factor": replication_factor,
                    "config": config,
                    "schema_registered": schema_result
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create topic {topic_name}: {e}")
            return OperationResult(
                success=False,
                message=f"Failed to create topic: {str(e)}",
                topic_name=topic_name,
                operation="create",
                errors=[str(e)]
            )
    
    async def _register_schema(self, topic_name: str) -> bool:
        """Register schema for topic if configured."""
        if not self.schema_registry_client:
            return False
        
        topic_config = self.topics_config.topics[topic_name]
        if not topic_config.schema or not self.topics_config.schemas:
            return False
        
        try:
            value_schema_name = topic_config.schema.get("value_schema")
            if value_schema_name and value_schema_name in self.topics_config.schemas:
                schema_def = self.topics_config.schemas[value_schema_name]
                schema_str = json.dumps(schema_def)
                
                # Register value schema
                schema_id = self.schema_registry_client.register_schema(
                    f"{topic_name}-value", schema_str
                )
                logger.info(f"Registered schema for {topic_name}-value: {schema_id}")
                
                # Register key schema if specified
                key_schema_name = topic_config.schema.get("key_schema")
                if key_schema_name:
                    key_schema_str = json.dumps({"type": "string"})
                    key_schema_id = self.schema_registry_client.register_schema(
                        f"{topic_name}-key", key_schema_str
                    )
                    logger.info(f"Registered key schema for {topic_name}-key: {key_schema_id}")
                
                return True
                
        except Exception as e:
            logger.warning(f"Failed to register schema for {topic_name}: {e}")
            return False
        
        return False
    
    async def update_topics(
        self,
        environment: str = "development",
        topics: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> List[OperationResult]:
        """Update existing topics with new configuration."""
        results = []
        topics_to_update = topics or list(self.topics_config.topics.keys())
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Updating topics...", total=len(topics_to_update))
            
            for topic_name in topics_to_update:
                progress.update(task, description=f"Updating {topic_name}")
                
                try:
                    result = await self._update_single_topic(
                        topic_name, environment, dry_run
                    )
                    results.append(result)
                    
                    if result.success:
                        console.print(f"✅ {topic_name}: {result.message}")
                    else:
                        console.print(f"❌ {topic_name}: {result.message}")
                        
                except Exception as e:
                    error_result = OperationResult(
                        success=False,
                        message=f"Failed to update topic: {str(e)}",
                        topic_name=topic_name,
                        operation="update",
                        errors=[str(e)]
                    )
                    results.append(error_result)
                    console.print(f"❌ {topic_name}: {str(e)}")
                
                progress.advance(task)
        
        return results
    
    async def _update_single_topic(
        self, topic_name: str, environment: str, dry_run: bool = False
    ) -> OperationResult:
        """Update a single topic configuration."""
        if topic_name not in self.topics_config.topics:
            return OperationResult(
                success=False,
                message=f"Topic {topic_name} not found in configuration",
                topic_name=topic_name,
                operation="update"
            )
        
        # Check if topic exists
        existing_topics = await self._get_existing_topics()
        if topic_name not in existing_topics:
            return OperationResult(
                success=False,
                message=f"Topic {topic_name} does not exist",
                topic_name=topic_name,
                operation="update"
            )
        
        # Get current and desired configuration
        current_config = await self._get_topic_config(topic_name)
        desired_config = self._get_topic_config_for_env(topic_name, environment)
        
        # Find differences
        config_changes = {}
        for key, value in desired_config.items():
            if key not in current_config or str(current_config[key]) != str(value):
                config_changes[key] = str(value)
        
        if not config_changes:
            return OperationResult(
                success=True,
                message="No configuration changes needed",
                topic_name=topic_name,
                operation="update"
            )
        
        if dry_run:
            return OperationResult(
                success=True,
                message=f"Would update configuration: {config_changes}",
                topic_name=topic_name,
                operation="update",
                details={"changes": config_changes}
            )
        
        try:
            # Update topic configuration
            resource = ConfigResource(
                restype=ConfigResource.Type.TOPIC,
                name=topic_name
            )
            
            futures = self.admin_client.alter_configs([resource], config_changes)
            
            # Wait for operation to complete
            for resource, future in futures.items():
                try:
                    future.result(timeout=30)
                    logger.info(f"Topic {topic_name} configuration updated")
                except Exception as e:
                    raise e
            
            return OperationResult(
                success=True,
                message=f"Configuration updated: {list(config_changes.keys())}",
                topic_name=topic_name,
                operation="update",
                details={"changes": config_changes}
            )
            
        except Exception as e:
            logger.error(f"Failed to update topic {topic_name}: {e}")
            return OperationResult(
                success=False,
                message=f"Failed to update topic: {str(e)}",
                topic_name=topic_name,
                operation="update",
                errors=[str(e)]
            )
    
    async def delete_topics(
        self,
        topics: List[str],
        confirm: bool = False,
        force: bool = False
    ) -> List[OperationResult]:
        """Delete topics with safety checks."""
        if not confirm and not force:
            console.print("❌ Topic deletion requires --confirm flag")
            return []
        
        results = []
        
        # Safety check for production topics
        protected_topics = {"tda_audit", "tda_errors"}
        dangerous_topics = [t for t in topics if t in protected_topics]
        
        if dangerous_topics and not force:
            console.print(f"❌ Cannot delete protected topics without --force: {dangerous_topics}")
            return [
                OperationResult(
                    success=False,
                    message=f"Protected topic, use --force to delete",
                    topic_name=topic,
                    operation="delete"
                )
                for topic in dangerous_topics
            ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Deleting topics...", total=len(topics))
            
            for topic_name in topics:
                progress.update(task, description=f"Deleting {topic_name}")
                
                try:
                    result = await self._delete_single_topic(topic_name)
                    results.append(result)
                    
                    if result.success:
                        console.print(f"✅ {topic_name}: {result.message}")
                    else:
                        console.print(f"❌ {topic_name}: {result.message}")
                        
                except Exception as e:
                    error_result = OperationResult(
                        success=False,
                        message=f"Failed to delete topic: {str(e)}",
                        topic_name=topic_name,
                        operation="delete",
                        errors=[str(e)]
                    )
                    results.append(error_result)
                    console.print(f"❌ {topic_name}: {str(e)}")
                
                progress.advance(task)
        
        return results
    
    async def _delete_single_topic(self, topic_name: str) -> OperationResult:
        """Delete a single topic."""
        try:
            # Check if topic exists
            existing_topics = await self._get_existing_topics()
            if topic_name not in existing_topics:
                return OperationResult(
                    success=False,
                    message=f"Topic {topic_name} does not exist",
                    topic_name=topic_name,
                    operation="delete"
                )
            
            # Delete topic
            futures = self.admin_client.delete_topics([topic_name])
            
            # Wait for operation to complete
            for topic, future in futures.items():
                try:
                    future.result(timeout=30)
                    logger.info(f"Topic {topic} deleted successfully")
                except Exception as e:
                    raise e
            
            return OperationResult(
                success=True,
                message="Topic deleted successfully",
                topic_name=topic_name,
                operation="delete"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete topic {topic_name}: {e}")
            return OperationResult(
                success=False,
                message=f"Failed to delete topic: {str(e)}",
                topic_name=topic_name,
                operation="delete",
                errors=[str(e)]
            )
    
    async def validate_configuration(
        self, environment: str = "development"
    ) -> List[OperationResult]:
        """Validate topic configuration."""
        results = []
        
        console.print("🔍 Validating topic configuration...")
        
        for topic_name, topic_config in self.topics_config.topics.items():
            try:
                # Validate basic configuration
                partitions = self._get_topic_partitions_for_env(topic_name, environment)
                config = self._get_topic_config_for_env(topic_name, environment)
                
                errors = []
                warnings = []
                
                # Validate partition count
                if partitions < 1:
                    errors.append("Partition count must be at least 1")
                elif partitions > 100:
                    warnings.append("High partition count may impact performance")
                
                # Validate replication factor
                replication_factor = config.get("replication_factor", 1)
                if replication_factor < 1:
                    errors.append("Replication factor must be at least 1")
                elif replication_factor > 5:
                    warnings.append("High replication factor may impact performance")
                
                # Validate retention settings
                retention_ms = config.get("retention_ms")
                if retention_ms and retention_ms < 3600000:  # 1 hour
                    warnings.append("Very short retention period")
                
                # Validate schema if present
                schema_errors = await self._validate_schema(topic_name)
                errors.extend(schema_errors)
                
                # Create result
                success = len(errors) == 0
                message = "Configuration valid"
                if errors:
                    message = f"Validation failed: {', '.join(errors)}"
                elif warnings:
                    message = f"Configuration valid with warnings: {', '.join(warnings)}"
                
                result = OperationResult(
                    success=success,
                    message=message,
                    topic_name=topic_name,
                    operation="validate",
                    errors=errors,
                    details={
                        "warnings": warnings,
                        "partitions": partitions,
                        "config": config
                    }
                )
                results.append(result)
                
            except Exception as e:
                error_result = OperationResult(
                    success=False,
                    message=f"Validation error: {str(e)}",
                    topic_name=topic_name,
                    operation="validate",
                    errors=[str(e)]
                )
                results.append(error_result)
        
        # Print validation summary
        self._print_validation_summary(results)
        
        return results
    
    async def _validate_schema(self, topic_name: str) -> List[str]:
        """Validate schema configuration for topic."""
        errors = []
        
        if not self.topics_config.topics[topic_name].schema:
            return errors
        
        topic_schema = self.topics_config.topics[topic_name].schema
        
        # Check if schema exists in definitions
        value_schema_name = topic_schema.get("value_schema")
        if value_schema_name:
            if not self.topics_config.schemas:
                errors.append("Schema definitions missing")
            elif value_schema_name not in self.topics_config.schemas:
                errors.append(f"Schema {value_schema_name} not found in definitions")
            else:
                # Validate schema structure
                schema_def = self.topics_config.schemas[value_schema_name]
                if not isinstance(schema_def, dict):
                    errors.append("Invalid schema definition format")
                elif "type" not in schema_def:
                    errors.append("Schema missing 'type' field")
        
        return errors
    
    def _print_validation_summary(self, results: List[OperationResult]) -> None:
        """Print validation summary table."""
        table = Table(title="Configuration Validation Summary")
        table.add_column("Topic", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Issues", style="yellow")
        
        for result in results:
            status = "✅ Valid" if result.success else "❌ Invalid"
            issues = ", ".join(result.errors + result.details.get("warnings", []))
            if not issues:
                issues = "None"
            
            table.add_row(result.topic_name, status, issues)
        
        console.print(table)
    
    async def health_check(
        self, topics: Optional[List[str]] = None
    ) -> List[TopicStatus]:
        """Perform comprehensive health check on topics."""
        topics_to_check = topics or list(self.topics_config.topics.keys())
        statuses = []
        
        console.print("🏥 Performing topic health checks...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Health checking...", total=len(topics_to_check))
            
            for topic_name in topics_to_check:
                progress.update(task, description=f"Checking {topic_name}")
                
                try:
                    status = await self._check_single_topic_health(topic_name)
                    statuses.append(status)
                except Exception as e:
                    error_status = TopicStatus(
                        name=topic_name,
                        health="error",
                        errors=[str(e)]
                    )
                    statuses.append(error_status)
                
                progress.advance(task)
        
        # Print health summary
        self._print_health_summary(statuses)
        
        return statuses
    
    async def _check_single_topic_health(self, topic_name: str) -> TopicStatus:
        """Check health of a single topic."""
        status = TopicStatus(name=topic_name)
        
        try:
            # Check if topic exists
            metadata = await self._get_topic_metadata(topic_name)
            if not metadata:
                status.health = "missing"
                status.errors.append("Topic does not exist")
                return status
            
            status.exists = True
            status.partitions = len(metadata.partitions)
            
            # Check partition health
            for partition_id, partition in metadata.partitions.items():
                if partition.error:
                    status.errors.append(f"Partition {partition_id}: {partition.error}")
                
                # Check replica health
                if len(partition.replicas) < len(partition.isrs):
                    status.warnings.append(
                        f"Partition {partition_id}: Some replicas out of sync"
                    )
            
            # Get topic configuration
            config = await self._get_topic_config(topic_name)
            status.config = config
            
            # Determine overall health
            if status.errors:
                status.health = "unhealthy"
            elif status.warnings:
                status.health = "warning"
            else:
                status.health = "healthy"
            
        except Exception as e:
            status.health = "error"
            status.errors.append(str(e))
        
        return status
    
    def _print_health_summary(self, statuses: List[TopicStatus]) -> None:
        """Print health check summary table."""
        table = Table(title="Topic Health Summary")
        table.add_column("Topic", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Partitions", justify="right")
        table.add_column("Health", style="bold")
        table.add_column("Issues", style="yellow")
        
        for status in statuses:
            # Status icon
            if status.exists:
                exists_icon = "✅"
            else:
                exists_icon = "❌"
            
            # Health color
            health_color = {
                "healthy": "green",
                "warning": "yellow",
                "unhealthy": "red",
                "error": "red",
                "missing": "red",
                "unknown": "white"
            }.get(status.health, "white")
            
            issues = status.errors + status.warnings
            issues_text = ", ".join(issues) if issues else "None"
            
            table.add_row(
                status.name,
                exists_icon,
                str(status.partitions),
                f"[{health_color}]{status.health.upper()}[/{health_color}]",
                issues_text
            )
        
        console.print(table)
    
    async def list_topics(self, pattern: Optional[str] = None) -> List[str]:
        """List existing topics, optionally filtered by pattern."""
        try:
            metadata = self.admin_client.list_topics(timeout=10)
            topics = list(metadata.topics.keys())
            
            if pattern:
                import fnmatch
                topics = [t for t in topics if fnmatch.fnmatch(t, pattern)]
            
            return sorted(topics)
        except Exception as e:
            logger.error(f"Failed to list topics: {e}")
            return []
    
    async def _get_existing_topics(self) -> Set[str]:
        """Get set of existing topic names."""
        try:
            metadata = self.admin_client.list_topics(timeout=10)
            return set(metadata.topics.keys())
        except Exception as e:
            logger.error(f"Failed to get existing topics: {e}")
            return set()
    
    async def _get_topic_metadata(self, topic_name: str) -> Optional[TopicMetadata]:
        """Get metadata for a specific topic."""
        try:
            metadata = self.admin_client.list_topics(topic=topic_name, timeout=10)
            return metadata.topics.get(topic_name)
        except Exception as e:
            logger.error(f"Failed to get metadata for topic {topic_name}: {e}")
            return None
    
    async def _get_topic_config(self, topic_name: str) -> Dict[str, str]:
        """Get current configuration for a topic."""
        try:
            resource = ConfigResource(
                restype=ConfigResource.Type.TOPIC,
                name=topic_name
            )
            
            configs = self.admin_client.describe_configs([resource])
            
            for resource, future in configs.items():
                config_dict = future.result(timeout=10)
                return {entry.name: entry.value for entry in config_dict.values()}
            
            return {}
        except Exception as e:
            logger.error(f"Failed to get config for topic {topic_name}: {e}")
            return {}
    
    def print_configuration_tree(self, environment: str = "development") -> None:
        """Print configuration tree for visualization."""
        tree = Tree(f"🌳 TDA Kafka Topics Configuration ({environment})")
        
        for topic_name, topic_config in self.topics_config.topics.items():
            partitions = self._get_topic_partitions_for_env(topic_name, environment)
            config = self._get_topic_config_for_env(topic_name, environment)
            
            topic_branch = tree.add(f"📋 {topic_name}")
            topic_branch.add(f"📝 {topic_config.description}")
            topic_branch.add(f"🔢 Partitions: {partitions}")
            topic_branch.add(f"🔄 Replication: {config.get('replication_factor', 1)}")
            topic_branch.add(f"⏰ Retention: {config.get('retention_ms', 'N/A')}ms")
            
            if topic_config.event_types:
                events_branch = topic_branch.add("📨 Event Types")
                for event_type in topic_config.event_types:
                    events_branch.add(f"• {event_type}")
        
        console.print(tree)


# CLI Commands
@click.group()
@click.option(
    "--config",
    default="config/topics.yml",
    help="Path to topics configuration file"
)
@click.option(
    "--kafka-config",
    help="Kafka configuration (JSON format)"
)
@click.option(
    "--schema-registry-url",
    help="Schema Registry URL"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level"
)
@click.pass_context
def cli(ctx, config, kafka_config, schema_registry_url, log_level):
    """TDA Kafka Topic Manager - Advanced topic management tool."""
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    kafka_conf = None
    if kafka_config:
        kafka_conf = json.loads(kafka_config)
    
    ctx.ensure_object(dict)
    ctx.obj["manager"] = TopicManager(
        config_path=config,
        kafka_config=kafka_conf,
        schema_registry_url=schema_registry_url
    )


@cli.command()
@click.option(
    "--env",
    default="development",
    help="Environment to use for topic creation"
)
@click.option(
    "--topics",
    help="Comma-separated list of topics to create (default: all)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be created without actually creating"
)
@click.pass_context
def create(ctx, env, topics, dry_run):
    """Create topics based on configuration."""
    manager = ctx.obj["manager"]
    
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    async def _create():
        return await manager.create_topics(
            environment=env,
            topics=topic_list,
            dry_run=dry_run
        )
    
    results = asyncio.run(_create())
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    total = len(results)
    console.print(f"\n📊 Summary: {successful}/{total} topics created successfully")


@cli.command()
@click.option(
    "--env",
    default="development",
    help="Environment to use for topic updates"
)
@click.option(
    "--topics",
    help="Comma-separated list of topics to update (default: all)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be updated without actually updating"
)
@click.pass_context
def update(ctx, env, topics, dry_run):
    """Update existing topics with new configuration."""
    manager = ctx.obj["manager"]
    
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    async def _update():
        return await manager.update_topics(
            environment=env,
            topics=topic_list,
            dry_run=dry_run
        )
    
    results = asyncio.run(_update())
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    total = len(results)
    console.print(f"\n📊 Summary: {successful}/{total} topics updated successfully")


@cli.command()
@click.option(
    "--topics",
    required=True,
    help="Comma-separated list of topics to delete"
)
@click.option(
    "--confirm",
    is_flag=True,
    help="Confirm topic deletion"
)
@click.option(
    "--force",
    is_flag=True,
    help="Force deletion of protected topics"
)
@click.pass_context
def delete(ctx, topics, confirm, force):
    """Delete topics (use with caution)."""
    manager = ctx.obj["manager"]
    
    topic_list = [t.strip() for t in topics.split(",")]
    
    if not confirm and not force:
        console.print("❌ Topic deletion requires --confirm flag")
        return
    
    if not force:
        console.print(f"⚠️  You are about to delete topics: {', '.join(topic_list)}")
        if not click.confirm("Are you sure?"):
            console.print("❌ Operation cancelled")
            return
    
    async def _delete():
        return await manager.delete_topics(
            topics=topic_list,
            confirm=confirm,
            force=force
        )
    
    results = asyncio.run(_delete())
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    total = len(results)
    console.print(f"\n📊 Summary: {successful}/{total} topics deleted successfully")


@cli.command()
@click.option(
    "--env",
    default="development",
    help="Environment to validate configuration for"
)
@click.pass_context
def validate(ctx, env):
    """Validate topic configuration."""
    manager = ctx.obj["manager"]
    
    async def _validate():
        return await manager.validate_configuration(environment=env)
    
    results = asyncio.run(_validate())
    
    # Print summary
    valid = sum(1 for r in results if r.success)
    total = len(results)
    console.print(f"\n📊 Validation Summary: {valid}/{total} topics have valid configuration")


@cli.command()
@click.option(
    "--topics",
    help="Comma-separated list of topics to check (default: all configured)"
)
@click.pass_context
def health_check(ctx, topics):
    """Perform health check on topics."""
    manager = ctx.obj["manager"]
    
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    async def _health_check():
        return await manager.health_check(topics=topic_list)
    
    statuses = asyncio.run(_health_check())
    
    # Print summary
    healthy = sum(1 for s in statuses if s.health == "healthy")
    total = len(statuses)
    console.print(f"\n📊 Health Summary: {healthy}/{total} topics are healthy")


@cli.command()
@click.option(
    "--pattern",
    help="Pattern to filter topics (supports wildcards)"
)
@click.pass_context
def list(ctx, pattern):
    """List existing topics."""
    manager = ctx.obj["manager"]
    
    async def _list():
        return await manager.list_topics(pattern=pattern)
    
    topics = asyncio.run(_list())
    
    if topics:
        console.print("📋 Existing Topics:")
        for topic in topics:
            console.print(f"  • {topic}")
    else:
        console.print("📋 No topics found")


@cli.command()
@click.option(
    "--env",
    default="development",
    help="Environment to show configuration for"
)
@click.pass_context
def show_config(ctx, env):
    """Show configuration tree."""
    manager = ctx.obj["manager"]
    manager.print_configuration_tree(environment=env)


if __name__ == "__main__":
    cli()
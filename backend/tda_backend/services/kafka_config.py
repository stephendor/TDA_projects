"""
Kafka Configuration Management for TDA Backend.

This module provides comprehensive configuration management for Kafka operations,
including environment-specific settings, producer/consumer configurations,
security settings, and performance tuning parameters.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..config import settings


class CompressionType(str, Enum):
    """Kafka compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class SecurityProtocol(str, Enum):
    """Kafka security protocols."""
    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


class SASLMechanism(str, Enum):
    """SASL authentication mechanisms."""
    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"
    GSSAPI = "GSSAPI"


class KafkaProducerConfig(BaseModel):
    """Configuration for Kafka producer."""
    
    # Core settings
    bootstrap_servers: List[str] = Field(
        default_factory=lambda: settings.kafka_bootstrap_servers,
        description="Kafka bootstrap servers"
    )
    client_id: str = Field(
        default="tda-backend-producer",
        description="Producer client ID"
    )
    
    # Reliability settings
    acks: str = Field(
        default="all",
        description="Acknowledgment level (0, 1, all)"
    )
    retries: int = Field(
        default=2147483647,  # Max retries
        description="Number of retries"
    )
    retry_backoff_ms: int = Field(
        default=100,
        description="Retry backoff time in milliseconds"
    )
    delivery_timeout_ms: int = Field(
        default=120000,
        description="Total time to spend on delivery attempts"
    )
    request_timeout_ms: int = Field(
        default=30000,
        description="Request timeout in milliseconds"
    )
    
    # Performance settings
    batch_size: int = Field(
        default=16384,
        description="Batch size in bytes"
    )
    linger_ms: int = Field(
        default=0,
        description="Time to wait for additional messages"
    )
    buffer_memory: int = Field(
        default=33554432,  # 32MB
        description="Total memory for buffering"
    )
    compression_type: CompressionType = Field(
        default=CompressionType.SNAPPY,
        description="Compression algorithm"
    )
    max_in_flight_requests_per_connection: int = Field(
        default=5,
        description="Max unacknowledged requests per connection"
    )
    
    # Idempotence
    enable_idempotence: bool = Field(
        default=True,
        description="Enable idempotent producer"
    )
    
    # Partitioning
    partitioner_class: str = Field(
        default="org.apache.kafka.clients.producer.internals.DefaultPartitioner",
        description="Partitioner class"
    )
    
    @validator('acks')
    def validate_acks(cls, v):
        valid_acks = ['0', '1', 'all', '-1']
        if v not in valid_acks:
            raise ValueError(f"acks must be one of {valid_acks}")
        return v


class KafkaConsumerConfig(BaseModel):
    """Configuration for Kafka consumer."""
    
    # Core settings
    bootstrap_servers: List[str] = Field(
        default_factory=lambda: settings.kafka_bootstrap_servers,
        description="Kafka bootstrap servers"
    )
    group_id: str = Field(
        default_factory=lambda: settings.kafka_consumer_group,
        description="Consumer group ID"
    )
    client_id: str = Field(
        default="tda-backend-consumer",
        description="Consumer client ID"
    )
    
    # Offset management
    auto_offset_reset: str = Field(
        default_factory=lambda: settings.kafka_auto_offset_reset,
        description="Auto offset reset strategy"
    )
    enable_auto_commit: bool = Field(
        default_factory=lambda: settings.kafka_enable_auto_commit,
        description="Enable auto commit"
    )
    auto_commit_interval_ms: int = Field(
        default=5000,
        description="Auto commit interval in milliseconds"
    )
    
    # Session management
    session_timeout_ms: int = Field(
        default_factory=lambda: settings.kafka_session_timeout_ms,
        description="Session timeout in milliseconds"
    )
    heartbeat_interval_ms: int = Field(
        default_factory=lambda: settings.kafka_heartbeat_interval_ms,
        description="Heartbeat interval in milliseconds"
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        description="Maximum poll interval in milliseconds"
    )
    
    # Fetch settings
    fetch_min_bytes: int = Field(
        default=1,
        description="Minimum bytes to fetch"
    )
    fetch_max_wait_ms: int = Field(
        default=500,
        description="Maximum wait time for fetch"
    )
    max_partition_fetch_bytes: int = Field(
        default=1048576,  # 1MB
        description="Maximum bytes per partition fetch"
    )
    
    # Processing settings
    max_poll_records: int = Field(
        default=500,
        description="Maximum records per poll"
    )
    isolation_level: str = Field(
        default="read_uncommitted",
        description="Isolation level (read_uncommitted, read_committed)"
    )
    
    @validator('auto_offset_reset')
    def validate_auto_offset_reset(cls, v):
        valid_values = ['earliest', 'latest', 'none']
        if v not in valid_values:
            raise ValueError(f"auto_offset_reset must be one of {valid_values}")
        return v
    
    @validator('isolation_level')
    def validate_isolation_level(cls, v):
        valid_levels = ['read_uncommitted', 'read_committed']
        if v not in valid_levels:
            raise ValueError(f"isolation_level must be one of {valid_levels}")
        return v


class KafkaSecurityConfig(BaseModel):
    """Security configuration for Kafka."""
    
    security_protocol: SecurityProtocol = Field(
        default=SecurityProtocol.PLAINTEXT,
        description="Security protocol"
    )
    
    # SSL Configuration
    ssl_cafile: Optional[str] = Field(
        default=None,
        description="Path to CA certificate file"
    )
    ssl_certfile: Optional[str] = Field(
        default=None,
        description="Path to client certificate file"
    )
    ssl_keyfile: Optional[str] = Field(
        default=None,
        description="Path to client private key file"
    )
    ssl_password: Optional[str] = Field(
        default=None,
        description="Password for private key"
    )
    ssl_check_hostname: bool = Field(
        default=True,
        description="Check SSL hostname"
    )
    
    # SASL Configuration
    sasl_mechanism: Optional[SASLMechanism] = Field(
        default=None,
        description="SASL mechanism"
    )
    sasl_username: Optional[str] = Field(
        default=None,
        description="SASL username"
    )
    sasl_password: Optional[str] = Field(
        default=None,
        description="SASL password"
    )


class KafkaTopicConfig(BaseModel):
    """Topic configuration and management."""
    
    # Topic names with prefix
    jobs_topic: str = Field(
        default_factory=lambda: settings.get_kafka_topic(settings.kafka_topic_tda_jobs),
        description="TDA jobs topic name"
    )
    results_topic: str = Field(
        default_factory=lambda: settings.get_kafka_topic(settings.kafka_topic_tda_results),
        description="TDA results topic name"
    )
    events_topic: str = Field(
        default_factory=lambda: settings.get_kafka_topic(settings.kafka_topic_tda_events),
        description="TDA events topic name"
    )
    stream_data_topic: str = Field(
        default_factory=lambda: settings.get_kafka_topic(settings.kafka_topic_stream_data),
        description="Stream data topic name"
    )
    
    # Topic creation settings
    num_partitions: int = Field(
        default=3,
        description="Default number of partitions"
    )
    replication_factor: int = Field(
        default=1,
        description="Replication factor"
    )
    retention_ms: int = Field(
        default=604800000,  # 7 days
        description="Message retention time in milliseconds"
    )
    cleanup_policy: str = Field(
        default="delete",
        description="Topic cleanup policy"
    )
    compression_type: CompressionType = Field(
        default=CompressionType.SNAPPY,
        description="Topic compression type"
    )
    
    def get_all_topics(self) -> List[str]:
        """Get list of all managed topics."""
        return [
            self.jobs_topic,
            self.results_topic,
            self.events_topic,
            self.stream_data_topic
        ]
    
    def get_topic_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get topic configuration for creation."""
        base_config = {
            'num_partitions': self.num_partitions,
            'replication_factor': self.replication_factor,
            'config': {
                'retention.ms': str(self.retention_ms),
                'cleanup.policy': self.cleanup_policy,
                'compression.type': self.compression_type.value,
                'min.insync.replicas': '1'
            }
        }
        
        return {topic: base_config.copy() for topic in self.get_all_topics()}


class KafkaHealthConfig(BaseModel):
    """Health check configuration for Kafka."""
    
    health_check_timeout: int = Field(
        default=10,
        description="Health check timeout in seconds"
    )
    metadata_timeout: int = Field(
        default=5,
        description="Metadata request timeout in seconds"
    )
    connection_retry_delay: int = Field(
        default=5,
        description="Delay between connection retries in seconds"
    )
    max_connection_retries: int = Field(
        default=3,
        description="Maximum connection retry attempts"
    )


class KafkaMetricsConfig(BaseModel):
    """Metrics and monitoring configuration."""
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable Kafka metrics collection"
    )
    metrics_recording_level: str = Field(
        default="INFO",
        description="Metrics recording level"
    )
    jmx_port: Optional[int] = Field(
        default=None,
        description="JMX port for metrics"
    )


class KafkaConfig(BaseModel):
    """Complete Kafka configuration."""
    
    producer: KafkaProducerConfig = Field(
        default_factory=KafkaProducerConfig,
        description="Producer configuration"
    )
    consumer: KafkaConsumerConfig = Field(
        default_factory=KafkaConsumerConfig,
        description="Consumer configuration"
    )
    security: KafkaSecurityConfig = Field(
        default_factory=KafkaSecurityConfig,
        description="Security configuration"
    )
    topics: KafkaTopicConfig = Field(
        default_factory=KafkaTopicConfig,
        description="Topic configuration"
    )
    health: KafkaHealthConfig = Field(
        default_factory=KafkaHealthConfig,
        description="Health check configuration"
    )
    metrics: KafkaMetricsConfig = Field(
        default_factory=KafkaMetricsConfig,
        description="Metrics configuration"
    )
    
    # Global settings
    debug: bool = Field(
        default_factory=lambda: settings.debug_kafka,
        description="Enable Kafka debugging"
    )
    
    def get_producer_config(self) -> Dict[str, Any]:
        """Get producer configuration as dict for aiokafka."""
        config = {
            'bootstrap_servers': self.producer.bootstrap_servers,
            'client_id': self.producer.client_id,
            'acks': self.producer.acks,
            'retries': self.producer.retries,
            'retry_backoff_ms': self.producer.retry_backoff_ms,
            'delivery_timeout_ms': self.producer.delivery_timeout_ms,
            'request_timeout_ms': self.producer.request_timeout_ms,
            'batch_size': self.producer.batch_size,
            'linger_ms': self.producer.linger_ms,
            'buffer_memory': self.producer.buffer_memory,
            'compression_type': self.producer.compression_type.value,
            'max_in_flight_requests_per_connection': self.producer.max_in_flight_requests_per_connection,
            'enable_idempotence': self.producer.enable_idempotence,
        }
        
        # Add security configuration
        if self.security.security_protocol != SecurityProtocol.PLAINTEXT:
            config.update(self._get_security_config())
        
        return config
    
    def get_consumer_config(self) -> Dict[str, Any]:
        """Get consumer configuration as dict for aiokafka."""
        config = {
            'bootstrap_servers': self.consumer.bootstrap_servers,
            'group_id': self.consumer.group_id,
            'client_id': self.consumer.client_id,
            'auto_offset_reset': self.consumer.auto_offset_reset,
            'enable_auto_commit': self.consumer.enable_auto_commit,
            'auto_commit_interval_ms': self.consumer.auto_commit_interval_ms,
            'session_timeout_ms': self.consumer.session_timeout_ms,
            'heartbeat_interval_ms': self.consumer.heartbeat_interval_ms,
            'max_poll_interval_ms': self.consumer.max_poll_interval_ms,
            'fetch_min_bytes': self.consumer.fetch_min_bytes,
            'fetch_max_wait_ms': self.consumer.fetch_max_wait_ms,
            'max_partition_fetch_bytes': self.consumer.max_partition_fetch_bytes,
            'max_poll_records': self.consumer.max_poll_records,
            'isolation_level': self.consumer.isolation_level,
        }
        
        # Add security configuration
        if self.security.security_protocol != SecurityProtocol.PLAINTEXT:
            config.update(self._get_security_config())
        
        return config
    
    def _get_security_config(self) -> Dict[str, Any]:
        """Get security configuration for producer/consumer."""
        config = {
            'security_protocol': self.security.security_protocol.value
        }
        
        # SSL configuration
        if self.security.security_protocol in [SecurityProtocol.SSL, SecurityProtocol.SASL_SSL]:
            if self.security.ssl_cafile:
                config['ssl_cafile'] = self.security.ssl_cafile
            if self.security.ssl_certfile:
                config['ssl_certfile'] = self.security.ssl_certfile
            if self.security.ssl_keyfile:
                config['ssl_keyfile'] = self.security.ssl_keyfile
            if self.security.ssl_password:
                config['ssl_password'] = self.security.ssl_password
            config['ssl_check_hostname'] = self.security.ssl_check_hostname
        
        # SASL configuration
        if self.security.security_protocol in [SecurityProtocol.SASL_PLAINTEXT, SecurityProtocol.SASL_SSL]:
            if self.security.sasl_mechanism:
                config['sasl_mechanism'] = self.security.sasl_mechanism.value
            if self.security.sasl_username:
                config['sasl_username'] = self.security.sasl_username
            if self.security.sasl_password:
                config['sasl_password'] = self.security.sasl_password
        
        return config


# Global Kafka configuration instance
kafka_config = KafkaConfig()


def get_kafka_config() -> KafkaConfig:
    """Get the global Kafka configuration instance."""
    return kafka_config


def update_kafka_config(**kwargs) -> None:
    """Update Kafka configuration with new values."""
    global kafka_config
    
    # Create new config with updated values
    config_dict = kafka_config.dict()
    config_dict.update(kwargs)
    kafka_config = KafkaConfig(**config_dict)
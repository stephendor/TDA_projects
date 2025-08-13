"""
Configuration management for TDA Backend.

Handles environment variables and application settings using Pydantic.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # =============================================================================
    # API Configuration
    # =============================================================================
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_reload: bool = Field(default=False, description="Enable auto-reload in development")
    api_debug: bool = Field(default=False, description="Enable debug mode")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    # =============================================================================
    # Database Configuration
    # =============================================================================
    database_url: str = Field(
        default="postgresql+asyncpg://tda_user:tda_password@localhost:5432/tda_platform",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_pool_overflow: int = Field(default=30, description="Database pool overflow size")
    
    # =============================================================================
    # Redis Configuration
    # =============================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0", 
        description="Redis connection URL"
    )
    redis_max_connections: int = Field(default=50, description="Redis max connections")
    redis_retry_on_timeout: bool = Field(default=True, description="Retry Redis on timeout")
    
    # =============================================================================
    # Kafka Configuration
    # =============================================================================
    kafka_bootstrap_servers: List[str] = Field(
        default=["localhost:9092"],
        description="Kafka bootstrap servers"
    )
    kafka_topic_prefix: str = Field(default="tda_", description="Kafka topic prefix")
    kafka_consumer_group: str = Field(default="tda_backend", description="Kafka consumer group")
    kafka_auto_offset_reset: str = Field(default="earliest", description="Kafka offset reset strategy")
    kafka_enable_auto_commit: bool = Field(default=False, description="Enable Kafka auto commit")
    kafka_session_timeout_ms: int = Field(default=30000, description="Kafka session timeout")
    kafka_heartbeat_interval_ms: int = Field(default=10000, description="Kafka heartbeat interval")
    
    @validator("kafka_bootstrap_servers", pre=True)
    def parse_kafka_servers(cls, v):
        if isinstance(v, str):
            return [server.strip() for server in v.split(",") if server.strip()]
        return v
    
    # Kafka Topics
    kafka_topic_tda_jobs: str = Field(default="tda_jobs", description="TDA jobs topic")
    kafka_topic_tda_results: str = Field(default="tda_results", description="TDA results topic")
    kafka_topic_tda_events: str = Field(default="tda_events", description="TDA events topic")
    kafka_topic_stream_data: str = Field(default="stream_data", description="Stream data topic")
    
    # =============================================================================
    # Apache Flink Configuration
    # =============================================================================
    flink_rest_url: str = Field(default="http://localhost:8081", description="Flink REST API URL")
    flink_parallelism: int = Field(default=2, description="Flink job parallelism")
    flink_checkpoint_interval: int = Field(default=30000, description="Flink checkpoint interval")
    flink_state_backend: str = Field(default="filesystem", description="Flink state backend")
    flink_state_checkpoint_dir: str = Field(
        default="file:///tmp/flink-checkpoints",
        description="Flink checkpoint directory"
    )
    
    # =============================================================================
    # Security Configuration
    # =============================================================================
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        description="Application secret key"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    jwt_secret_key: str = Field(
        default="your-jwt-secret-key-change-this",
        description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time in minutes")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per period")
    rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")
    
    # =============================================================================
    # Logging Configuration
    # =============================================================================
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_max_size: str = Field(default="10MB", description="Max log file size")
    log_backup_count: int = Field(default=5, description="Log file backup count")
    
    # =============================================================================
    # Monitoring Configuration
    # =============================================================================
    prometheus_multiproc_dir: str = Field(
        default="/tmp/prometheus_multiproc",
        description="Prometheus multiprocess directory"
    )
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    tracing_enabled: bool = Field(default=False, description="Enable distributed tracing")
    jaeger_agent_host: str = Field(default="localhost", description="Jaeger agent host")
    jaeger_agent_port: int = Field(default=14268, description="Jaeger agent port")
    
    # =============================================================================
    # TDA Core Configuration
    # =============================================================================
    tda_max_points: int = Field(default=1000000, description="Maximum points in point cloud")
    tda_max_dimension: int = Field(default=3, description="Maximum homology dimension")
    tda_timeout_seconds: int = Field(default=300, description="TDA computation timeout")
    tda_result_cache_ttl: int = Field(default=3600, description="Result cache TTL in seconds")
    
    # Computation limits
    max_concurrent_jobs: int = Field(default=10, description="Maximum concurrent TDA jobs")
    job_cleanup_interval: int = Field(default=300, description="Job cleanup interval in seconds")
    result_retention_days: int = Field(default=7, description="Result retention period in days")
    
    # =============================================================================
    # C++ Integration Configuration
    # =============================================================================
    cpp_library_path: str = Field(default="../build/lib", description="Path to C++ libraries")
    cpp_thread_count: int = Field(default=0, description="C++ thread count (0=auto)")
    cpp_memory_limit_gb: int = Field(default=8, description="C++ memory limit in GB")
    
    # =============================================================================
    # File Upload Configuration
    # =============================================================================
    upload_max_size: str = Field(default="100MB", description="Maximum upload size")
    upload_allowed_extensions: List[str] = Field(
        default=[".json", ".csv", ".npz", ".h5"],
        description="Allowed file extensions"
    )
    upload_dir: str = Field(default="/tmp/tda-uploads", description="Upload directory")
    upload_cleanup_hours: int = Field(default=24, description="Upload cleanup period in hours")
    
    @validator("upload_allowed_extensions", pre=True)
    def parse_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",") if ext.strip()]
        return v
    
    # =============================================================================
    # Environment Configuration
    # =============================================================================
    environment: str = Field(default="development", description="Environment name")
    debug_sql: bool = Field(default=False, description="Enable SQL debugging")
    debug_kafka: bool = Field(default=False, description="Enable Kafka debugging")
    testing: bool = Field(default=False, description="Testing mode")
    
    # Mock services for development
    mock_flink: bool = Field(default=False, description="Mock Flink service")
    mock_kafka: bool = Field(default=False, description="Mock Kafka service")
    
    @validator("upload_max_size")
    def parse_upload_size(cls, v):
        """Convert size string to bytes."""
        if isinstance(v, str):
            v = v.upper()
            multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
            for suffix, multiplier in multipliers.items():
                if v.endswith(suffix):
                    return int(v[:-len(suffix)]) * multiplier
            return int(v)
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.testing or self.environment.lower() == "testing"
    
    def get_kafka_topic(self, topic: str) -> str:
        """Get full Kafka topic name with prefix."""
        return f"{self.kafka_topic_prefix}{topic}"
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.upload_dir,
            self.prometheus_multiproc_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    settings = Settings()
    
    # Ensure directories exist
    settings.ensure_directories()
    
    return settings


# Global settings instance
settings = get_settings()
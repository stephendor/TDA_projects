"""
Kafka Background Service for TDA Backend.

This module provides comprehensive background service management for Kafka consumers,
including lifecycle management, health monitoring, error recovery, and graceful shutdown.
Integrates with FastAPI's lifespan management for production-ready operation.
"""

import asyncio
import logging
import signal
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field

from ..config import settings
from .kafka_consumer import KafkaConsumerService, create_tda_consumer, MessageHandler
from .kafka_config import KafkaConfig
from .kafka_producer import MessageType
from .job_orchestration import get_kafka_message_handlers, get_job_service


logger = logging.getLogger(__name__)


class ServiceState(str, Enum):
    """Background service states."""
    STOPPED = "stopped"
    STARTING = "starting" 
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    RECOVERING = "recovering"


@dataclass
class ServiceMetrics:
    """Metrics for the background service."""
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    restart_count: int = 0
    last_restart_time: Optional[datetime] = None
    health_check_count: int = 0
    health_check_failures: int = 0
    consumer_errors: int = 0
    recovery_attempts: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    
    def update_uptime(self) -> None:
        """Update uptime calculation."""
        if self.start_time:
            self.uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()


@dataclass
class ServiceConfig:
    """Configuration for the background service."""
    consumer_group: str = "tda-backend-group"
    topics: List[str] = field(default_factory=list)
    health_check_interval: float = 30.0  # seconds
    restart_delay: float = 5.0  # seconds
    max_restart_attempts: int = 5
    restart_backoff_multiplier: float = 2.0
    max_restart_delay: float = 300.0  # 5 minutes
    graceful_shutdown_timeout: float = 30.0  # seconds
    enable_auto_recovery: bool = True
    enable_health_monitoring: bool = True


class KafkaBackgroundService:
    """
    Background service manager for Kafka consumer operations.
    
    Provides:
    - Consumer lifecycle management
    - Health monitoring and recovery
    - Graceful shutdown handling
    - Error tracking and metrics
    - Integration with FastAPI lifespan
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        kafka_config: Optional[KafkaConfig] = None,
        message_handlers: Optional[Dict[MessageType, MessageHandler]] = None
    ):
        """Initialize the background service."""
        self.config = config or ServiceConfig()
        self.kafka_config = kafka_config or KafkaConfig()
        self.message_handlers = message_handlers or {}
        
        # Service state
        self._state = ServiceState.STOPPED
        self._metrics = ServiceMetrics()
        self._shutdown_event = asyncio.Event()
        self._restart_lock = asyncio.Lock()
        
        # Consumer instance
        self.consumer: Optional[KafkaConsumerService] = None
        
        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            # Only set up signal handlers if running in main thread
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, self._signal_handler)
        except ValueError:
            # Not in main thread, signal handlers not available
            logger.debug("Signal handlers not available (not in main thread)")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())
    
    @property
    def state(self) -> ServiceState:
        """Get current service state."""
        return self._state
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is in a healthy state."""
        return self._state == ServiceState.RUNNING and self.consumer is not None
    
    async def start(self) -> None:
        """Start the background service."""
        if self._state in [ServiceState.STARTING, ServiceState.RUNNING]:
            logger.warning(f"Service already starting/running, current state: {self._state}")
            return
        
        logger.info("🚀 Starting Kafka background service...")
        self._state = ServiceState.STARTING
        
        try:
            # Record start time
            self._metrics.start_time = datetime.now(timezone.utc)
            
            # Create and start consumer
            await self._start_consumer()
            
            # Start background monitoring
            if self.config.enable_health_monitoring:
                await self._start_health_monitoring()
            
            self._state = ServiceState.RUNNING
            logger.info("✅ Kafka background service started successfully")
            
        except Exception as e:
            self._state = ServiceState.ERROR
            self._update_error_metrics("startup_error")
            logger.error(f"❌ Failed to start background service: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the background service gracefully."""
        if self._state in [ServiceState.STOPPED, ServiceState.STOPPING]:
            logger.warning(f"Service already stopped/stopping, current state: {self._state}")
            return
        
        logger.info("🛑 Stopping Kafka background service...")
        self._state = ServiceState.STOPPING
        
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop health monitoring
            if self._health_monitor_task and not self._health_monitor_task.done():
                self._health_monitor_task.cancel()
                try:
                    await asyncio.wait_for(
                        self._health_monitor_task,
                        timeout=5.0
                    )
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Stop consumer
            await self._stop_consumer()
            
            self._state = ServiceState.STOPPED
            logger.info("✅ Kafka background service stopped successfully")
            
        except Exception as e:
            self._state = ServiceState.ERROR
            logger.error(f"❌ Error during service shutdown: {e}")
            raise
    
    async def _start_consumer(self) -> None:
        """Start the Kafka consumer."""
        logger.info("📨 Starting Kafka consumer...")
        
        try:
            # Determine topics
            topics = self.config.topics or [
                self.kafka_config.topic.jobs_topic,
                self.kafka_config.topic.results_topic,
                self.kafka_config.topic.events_topic,
                self.kafka_config.topic.uploads_topic,
                self.kafka_config.topic.errors_topic
            ]
            
            # Create consumer with registered handlers
            if self.message_handlers:
                self.consumer = KafkaConsumerService(
                    consumer_group=self.config.consumer_group,
                    topics=topics,
                    config=self.kafka_config,
                    message_handlers=self.message_handlers
                )
            else:
                # Use default TDA consumer with standard handlers
                self.consumer = create_tda_consumer(
                    consumer_group=self.config.consumer_group,
                    topics=topics
                )
            
            # Start consumer
            await self.consumer.start()
            
            # Start consuming messages
            await self.consumer.start_consuming()
            
            logger.info(f"✅ Consumer started for topics: {topics}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start consumer: {e}")
            raise
    
    async def _stop_consumer(self) -> None:
        """Stop the Kafka consumer gracefully."""
        if not self.consumer:
            return
        
        logger.info("📪 Stopping Kafka consumer...")
        
        try:
            # Stop consuming
            await self.consumer.stop_consuming()
            
            # Stop consumer
            await asyncio.wait_for(
                self.consumer.stop(),
                timeout=self.config.graceful_shutdown_timeout
            )
            
            self.consumer = None
            logger.info("✅ Consumer stopped successfully")
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ Consumer shutdown timed out")
            self.consumer = None
        except Exception as e:
            logger.error(f"❌ Error stopping consumer: {e}")
            self.consumer = None
            raise
    
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring task."""
        logger.info("💊 Starting health monitoring...")
        
        self._health_monitor_task = asyncio.create_task(
            self._health_monitor_loop()
        )
    
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        logger.info("🔍 Health monitoring started")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Perform health check
                    await self._perform_health_check()
                    
                    # Wait for next check
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.health_check_interval
                    )
                    
                except asyncio.TimeoutError:
                    # Expected timeout, continue monitoring
                    continue
                    
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
                    self._metrics.health_check_failures += 1
                    
                    # Brief pause before retrying
                    await asyncio.sleep(5.0)
        
        except asyncio.CancelledError:
            logger.info("Health monitoring cancelled")
            raise
        
        finally:
            logger.info("Health monitoring stopped")
    
    async def _perform_health_check(self) -> None:
        """Perform health check on consumer."""
        self._metrics.health_check_count += 1
        
        if not self.consumer:
            logger.warning("🚨 Consumer not available during health check")
            if self.config.enable_auto_recovery:
                await self._attempt_recovery()
            return
        
        try:
            # Check consumer health
            health_status = await self.consumer.health_check()
            
            if health_status.get("status") != "healthy":
                logger.warning(f"🚨 Consumer unhealthy: {health_status}")
                self._metrics.health_check_failures += 1
                
                if self.config.enable_auto_recovery:
                    await self._attempt_recovery()
            else:
                # Update metrics on successful check
                self._metrics.update_uptime()
                logger.debug("✅ Health check passed")
                
        except Exception as e:
            logger.error(f"🚨 Health check failed: {e}")
            self._metrics.health_check_failures += 1
            self._update_error_metrics("health_check_error")
            
            if self.config.enable_auto_recovery:
                await self._attempt_recovery()
    
    async def _attempt_recovery(self) -> None:
        """Attempt to recover the consumer service."""
        async with self._restart_lock:
            if self._state != ServiceState.RUNNING:
                return
            
            logger.info("🔄 Attempting consumer recovery...")
            self._state = ServiceState.RECOVERING
            self._metrics.recovery_attempts += 1
            
            try:
                # Stop current consumer
                await self._stop_consumer()
                
                # Wait before restart
                delay = min(
                    self.config.restart_delay * (self.config.restart_backoff_multiplier ** self._metrics.restart_count),
                    self.config.max_restart_delay
                )
                
                logger.info(f"⏳ Waiting {delay:.1f}s before restart...")
                await asyncio.sleep(delay)
                
                # Restart consumer
                await self._start_consumer()
                
                # Update restart metrics
                self._metrics.restart_count += 1
                self._metrics.last_restart_time = datetime.now(timezone.utc)
                
                self._state = ServiceState.RUNNING
                logger.info("✅ Consumer recovery successful")
                
            except Exception as e:
                self._state = ServiceState.ERROR
                self._update_error_metrics("recovery_error")
                logger.error(f"❌ Consumer recovery failed: {e}")
                
                # Check restart limits
                if self._metrics.restart_count >= self.config.max_restart_attempts:
                    logger.error("🚨 Maximum restart attempts reached, stopping recovery")
                    await self.stop()
                    raise
    
    def _update_error_metrics(self, error_type: str) -> None:
        """Update error metrics."""
        self._metrics.consumer_errors += 1
        if error_type not in self._metrics.errors_by_type:
            self._metrics.errors_by_type[error_type] = 0
        self._metrics.errors_by_type[error_type] += 1
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: MessageHandler
    ) -> None:
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
        if self.consumer:
            self.consumer.register_handler(message_type, handler)
        logger.info(f"Registered handler for message type: {message_type}")
    
    def register_default_handler(self, handler: MessageHandler) -> None:
        """Register a default handler for unhandled message types."""
        if self.consumer:
            self.consumer.register_default_handler(handler)
        logger.info("Registered default message handler")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        self._metrics.update_uptime()
        
        consumer_health = None
        if self.consumer:
            try:
                consumer_health = await self.consumer.health_check()
            except Exception as e:
                consumer_health = {"status": "unhealthy", "error": str(e)}
        
        return {
            "service_state": self._state.value,
            "is_healthy": self.is_healthy,
            "uptime_seconds": self._metrics.uptime_seconds,
            "restart_count": self._metrics.restart_count,
            "last_restart_time": (
                self._metrics.last_restart_time.isoformat()
                if self._metrics.last_restart_time else None
            ),
            "health_checks": {
                "total": self._metrics.health_check_count,
                "failures": self._metrics.health_check_failures,
                "success_rate": (
                    (self._metrics.health_check_count - self._metrics.health_check_failures) / self._metrics.health_check_count
                    if self._metrics.health_check_count > 0 else 0
                )
            },
            "errors": {
                "total": self._metrics.consumer_errors,
                "by_type": self._metrics.errors_by_type
            },
            "recovery_attempts": self._metrics.recovery_attempts,
            "consumer_health": consumer_health,
            "config": {
                "consumer_group": self.config.consumer_group,
                "topics": self.config.topics,
                "auto_recovery_enabled": self.config.enable_auto_recovery,
                "health_monitoring_enabled": self.config.enable_health_monitoring
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        consumer_metrics = None
        if self.consumer:
            try:
                consumer_metrics = self.consumer.get_metrics()
            except Exception as e:
                logger.error(f"Failed to get consumer metrics: {e}")
        
        return {
            "service_metrics": {
                "state": self._state.value,
                "uptime_seconds": self._metrics.uptime_seconds,
                "restart_count": self._metrics.restart_count,
                "recovery_attempts": self._metrics.recovery_attempts,
                "consumer_errors": self._metrics.consumer_errors,
                "health_check_count": self._metrics.health_check_count,
                "health_check_failures": self._metrics.health_check_failures,
                "errors_by_type": self._metrics.errors_by_type
            },
            "consumer_metrics": consumer_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global service instance
_kafka_background_service: Optional[KafkaBackgroundService] = None


async def initialize_kafka_background_service(
    config: Optional[ServiceConfig] = None,
    kafka_config: Optional[KafkaConfig] = None,
    message_handlers: Optional[Dict[MessageType, MessageHandler]] = None
) -> KafkaBackgroundService:
    """Initialize the global Kafka background service."""
    global _kafka_background_service
    
    if _kafka_background_service is not None:
        logger.warning("Kafka background service already initialized")
        return _kafka_background_service
    
    logger.info("🚀 Initializing Kafka background service...")
    
    # Create service instance
    _kafka_background_service = KafkaBackgroundService(
        config=config,
        kafka_config=kafka_config,
        message_handlers=message_handlers
    )
    
    # Start the service
    await _kafka_background_service.start()
    
    logger.info("✅ Kafka background service initialized successfully")
    return _kafka_background_service


async def shutdown_kafka_background_service() -> None:
    """Shutdown the global Kafka background service."""
    global _kafka_background_service
    
    if _kafka_background_service is None:
        logger.warning("No Kafka background service to shutdown")
        return
    
    logger.info("🛑 Shutting down Kafka background service...")
    
    try:
        await _kafka_background_service.stop()
        _kafka_background_service = None
        logger.info("✅ Kafka background service shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error shutting down Kafka background service: {e}")
        raise


def get_kafka_background_service() -> Optional[KafkaBackgroundService]:
    """Get the global Kafka background service instance."""
    return _kafka_background_service


# FastAPI Lifespan Integration
@asynccontextmanager
async def kafka_background_lifespan():
    """
    FastAPI lifespan context manager for Kafka background service.
    
    Usage in main.py:
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with kafka_background_lifespan():
            yield
    """
    service = None
    try:
        # Startup
        if not settings.mock_kafka and not settings.testing:
            logger.info("🚀 Starting Kafka background service...")
            
            # Create service configuration
            config = ServiceConfig(
                consumer_group=settings.kafka_consumer_group,
                health_check_interval=30.0,
                enable_auto_recovery=True,
                enable_health_monitoring=True
            )
            
            # Get job orchestration handlers
            job_handlers = get_kafka_message_handlers()
            
            # Initialize service with job orchestration handlers
            service = await initialize_kafka_background_service(
                config=config, 
                message_handlers=job_handlers
            )
            
            logger.info("✅ Kafka background service started successfully")
        else:
            logger.info("📪 Kafka background service skipped (mocked or testing)")
        
        yield
        
    finally:
        # Shutdown
        if service:
            logger.info("🛑 Shutting down Kafka background service...")
            await shutdown_kafka_background_service()
            logger.info("✅ Kafka background service shutdown complete")


# Example custom message handlers
async def custom_job_handler(message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    """Custom handler for job messages with enhanced processing."""
    try:
        payload = message.get('payload', {})
        job_id = payload.get('job_id')
        
        logger.info(f"Processing job message: {job_id}")
        
        # Add custom job processing logic here
        # For example: database updates, notifications, downstream triggers
        
        return True
        
    except Exception as e:
        logger.error(f"Error in custom job handler: {e}")
        return False


async def custom_error_handler(message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    """Custom handler for error messages with alerting."""
    try:
        payload = message.get('payload', {})
        error_type = payload.get('error_type')
        error_message = payload.get('error_message')
        
        logger.error(f"Received error message - Type: {error_type}, Message: {error_message}")
        
        # Add custom error handling logic here
        # For example: alert generation, incident tracking, auto-remediation
        
        return True
        
    except Exception as e:
        logger.error(f"Error in custom error handler: {e}")
        return False


def create_custom_kafka_service(
    additional_handlers: Optional[Dict[MessageType, MessageHandler]] = None
) -> KafkaBackgroundService:
    """
    Factory function to create a Kafka background service with custom handlers.
    
    Args:
        additional_handlers: Additional message handlers to register
        
    Returns:
        Configured KafkaBackgroundService instance
    """
    # Base configuration
    config = ServiceConfig(
        consumer_group=settings.kafka_consumer_group,
        health_check_interval=30.0,
        restart_delay=5.0,
        max_restart_attempts=5,
        enable_auto_recovery=True,
        enable_health_monitoring=True
    )
    
    # Default handlers
    handlers = {
        MessageType.JOB_SUBMITTED: custom_job_handler,
        MessageType.JOB_STARTED: custom_job_handler,
        MessageType.JOB_COMPLETED: custom_job_handler,
        MessageType.JOB_FAILED: custom_job_handler,
        MessageType.ERROR_OCCURRED: custom_error_handler,
        MessageType.WARNING_ISSUED: custom_error_handler,
    }
    
    # Add additional handlers
    if additional_handlers:
        handlers.update(additional_handlers)
    
    return KafkaBackgroundService(
        config=config,
        kafka_config=KafkaConfig(),
        message_handlers=handlers
    )


# Wrapper functions for main.py integration
async def start_kafka_background_service() -> None:
    """Start the Kafka background service with job orchestration handlers."""
    if not settings.mock_kafka and not settings.testing:
        logger.info("🚀 Starting Kafka background service with job orchestration...")
        
        # Create service configuration
        config = ServiceConfig(
            consumer_group=settings.kafka_consumer_group,
            health_check_interval=30.0,
            enable_auto_recovery=True,
            enable_health_monitoring=True
        )
        
        # Initialize job orchestration service first
        job_service = get_job_service()
        
        # Get job orchestration handlers
        job_handlers = get_kafka_message_handlers()
        
        # Initialize service with job orchestration handlers
        await initialize_kafka_background_service(
            config=config, 
            message_handlers=job_handlers
        )
        
        logger.info("✅ Kafka background service started successfully")
    else:
        logger.info("📪 Kafka background service skipped (mocked or testing)")


async def stop_kafka_background_service() -> None:
    """Stop the Kafka background service."""
    await shutdown_kafka_background_service()
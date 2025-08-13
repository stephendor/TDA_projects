"""
Kafka Consumer Service for TDA Backend.

This module provides asynchronous Kafka consumer functionality with message
processing handlers, error handling, and monitoring capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime, timezone
from dataclasses import dataclass

import aiokafka
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError, CommitFailedError

from ..config import settings
from .kafka_config import KafkaConfig
from .kafka_producer import MessageType, TDAMessage


logger = logging.getLogger(__name__)


@dataclass
class ConsumerMetrics:
    """Metrics for Kafka consumer."""
    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_consumed: int = 0
    processing_time_total: float = 0.0
    last_consume_time: Optional[datetime] = None
    errors_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.errors_by_type is None:
            self.errors_by_type = {}


MessageHandler = Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[bool]]


class KafkaConsumerService:
    """Asynchronous Kafka Consumer Service for TDA Backend."""
    
    def __init__(
        self,
        consumer_group: str,
        topics: List[str],
        config: Optional[KafkaConfig] = None,
        message_handlers: Optional[Dict[MessageType, MessageHandler]] = None
    ):
        """Initialize the Kafka consumer service."""
        self.config = config or KafkaConfig()
        self.consumer_group = consumer_group
        self.topics = topics
        self.consumer: Optional[AIOKafkaConsumer] = None
        self._is_started = False
        self._is_consuming = False
        self._consume_task: Optional[asyncio.Task] = None
        self._metrics = ConsumerMetrics()
        
        # Message handlers
        self._message_handlers: Dict[MessageType, MessageHandler] = message_handlers or {}
        self._default_handler: Optional[MessageHandler] = None
        
        # Processing configuration
        self.max_batch_size = 100
        self.max_wait_time = 1.0
        self.auto_commit = True
        
    async def start(self) -> None:
        """Start the Kafka consumer."""
        if self._is_started:
            logger.warning("Kafka consumer is already started")
            return
        
        try:
            logger.info(f"Starting Kafka consumer for group '{self.consumer_group}'...")
            
            # Create consumer with configuration
            consumer_config = self.config.consumer
            
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=consumer_config.bootstrap_servers,
                group_id=self.consumer_group,
                client_id=f"{consumer_config.client_id}-{self.consumer_group}",
                
                # Offset management
                auto_offset_reset=consumer_config.auto_offset_reset,
                enable_auto_commit=self.auto_commit,
                auto_commit_interval_ms=consumer_config.auto_commit_interval_ms,
                
                # Session management
                session_timeout_ms=consumer_config.session_timeout_ms,
                heartbeat_interval_ms=consumer_config.heartbeat_interval_ms,
                max_poll_interval_ms=consumer_config.max_poll_interval_ms,
                
                # Fetching
                fetch_max_wait_ms=consumer_config.fetch_max_wait_ms,
                fetch_min_bytes=consumer_config.fetch_min_bytes,
                fetch_max_bytes=consumer_config.fetch_max_bytes,
                max_partition_fetch_bytes=consumer_config.max_partition_fetch_bytes,
                
                # Processing
                max_poll_records=min(self.max_batch_size, consumer_config.max_poll_records),
                
                # Deserialization
                value_deserializer=self._deserialize_message,
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
            )
            
            await self.consumer.start()
            self._is_started = True
            
            logger.info(
                f"Kafka consumer started successfully. "
                f"Group: {self.consumer_group}, Topics: {self.topics}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Kafka consumer."""
        if not self._is_started:
            return
        
        try:
            logger.info("Stopping Kafka consumer...")
            
            # Stop consuming task
            if self._consume_task and not self._consume_task.done():
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass
            
            # Stop consumer
            if self.consumer:
                await self.consumer.stop()
            
            self._is_started = False
            self._is_consuming = False
            
            logger.info("Kafka consumer stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Kafka consumer: {e}")
            raise
    
    def _deserialize_message(self, data: bytes) -> Dict[str, Any]:
        """Deserialize message from JSON bytes."""
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to deserialize message: {e}")
            return {"error": "Failed to deserialize message", "raw_data": data.hex()}
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: MessageHandler
    ) -> None:
        """Register a message handler for a specific message type."""
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def register_default_handler(self, handler: MessageHandler) -> None:
        """Register a default handler for unhandled message types."""
        self._default_handler = handler
        logger.info("Registered default message handler")
    
    async def _process_message(
        self,
        message: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """Process a single message using registered handlers."""
        try:
            # Extract message type from metadata
            message_type_str = None
            if "metadata" in message and "message_type" in message["metadata"]:
                message_type_str = message["metadata"]["message_type"]
            elif "message_type" in message:
                message_type_str = message["message_type"]
            
            # Find appropriate handler
            handler = None
            if message_type_str:
                try:
                    message_type = MessageType(message_type_str)
                    handler = self._message_handlers.get(message_type)
                except ValueError:
                    logger.warning(f"Unknown message type: {message_type_str}")
            
            if handler is None:
                handler = self._default_handler
            
            if handler is None:
                logger.warning(f"No handler found for message type: {message_type_str}")
                return True  # Consider unhandled messages as "processed"
            
            # Process message
            start_time = asyncio.get_event_loop().time()
            success = await handler(message, metadata)
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update metrics
            if success:
                self._metrics.messages_processed += 1
            else:
                self._metrics.messages_failed += 1
                self._update_error_metrics("processing_failed")
            
            self._metrics.processing_time_total += processing_time
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._metrics.messages_failed += 1
            self._update_error_metrics("processing_error")
            return False
    
    def _update_error_metrics(self, error_type: str) -> None:
        """Update error metrics."""
        if error_type not in self._metrics.errors_by_type:
            self._metrics.errors_by_type[error_type] = 0
        self._metrics.errors_by_type[error_type] += 1
    
    async def start_consuming(self) -> None:
        """Start consuming messages."""
        if not self._is_started:
            raise RuntimeError("Consumer must be started before consuming")
        
        if self._is_consuming:
            logger.warning("Consumer is already consuming messages")
            return
        
        self._is_consuming = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        
        logger.info("Started consuming messages")
    
    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        if not self._is_consuming:
            return
        
        self._is_consuming = False
        
        if self._consume_task and not self._consume_task.done():
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped consuming messages")
    
    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        logger.info("Starting message consumption loop")
        
        try:
            while self._is_consuming:
                try:
                    # Fetch messages
                    msg_pack = await asyncio.wait_for(
                        self.consumer.getmany(
                            max_records=self.max_batch_size,
                            timeout_ms=int(self.max_wait_time * 1000)
                        ),
                        timeout=self.max_wait_time + 1.0
                    )
                    
                    if not msg_pack:
                        continue
                    
                    # Process messages by topic-partition
                    for topic_partition, messages in msg_pack.items():
                        await self._process_message_batch(topic_partition, messages)
                    
                    # Commit offsets if auto-commit is disabled
                    if not self.auto_commit:
                        try:
                            await self.consumer.commit()
                        except CommitFailedError as e:
                            logger.error(f"Failed to commit offsets: {e}")
                            self._update_error_metrics("commit_failed")
                
                except asyncio.TimeoutError:
                    # Timeout is expected, continue consuming
                    continue
                    
                except KafkaError as e:
                    logger.error(f"Kafka error in consume loop: {e}")
                    self._update_error_metrics("kafka_error")
                    await asyncio.sleep(1)  # Brief pause before retrying
                    
                except Exception as e:
                    logger.error(f"Unexpected error in consume loop: {e}")
                    self._update_error_metrics("unexpected_error")
                    await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            logger.info("Consume loop cancelled")
            raise
        
        finally:
            logger.info("Message consumption loop ended")
    
    async def _process_message_batch(
        self,
        topic_partition,
        messages: List[aiokafka.ConsumerRecord]
    ) -> None:
        """Process a batch of messages from a single partition."""
        for message in messages:
            try:
                # Update consumption metrics
                self._metrics.messages_consumed += 1
                self._metrics.bytes_consumed += len(message.value)
                self._metrics.last_consume_time = datetime.now(timezone.utc)
                
                # Extract message metadata
                metadata = {
                    "topic": message.topic,
                    "partition": message.partition,
                    "offset": message.offset,
                    "timestamp": message.timestamp,
                    "headers": dict(message.headers) if message.headers else {},
                    "key": message.key
                }
                
                # Process the message
                await self._process_message(message.value, metadata)
                
                logger.debug(
                    f"Processed message from {message.topic}:{message.partition}@{message.offset}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error processing message from "
                    f"{message.topic}:{message.partition}@{message.offset}: {e}"
                )
                self._update_error_metrics("message_processing_error")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        avg_processing_time = (
            self._metrics.processing_time_total / self._metrics.messages_processed
            if self._metrics.messages_processed > 0 else 0
        )
        
        success_rate = (
            self._metrics.messages_processed / self._metrics.messages_consumed
            if self._metrics.messages_consumed > 0 else 0
        )
        
        return {
            "messages_consumed": self._metrics.messages_consumed,
            "messages_processed": self._metrics.messages_processed,
            "messages_failed": self._metrics.messages_failed,
            "bytes_consumed": self._metrics.bytes_consumed,
            "processing_time_total": self._metrics.processing_time_total,
            "average_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "last_consume_time": (
                self._metrics.last_consume_time.isoformat()
                if self._metrics.last_consume_time else None
            ),
            "errors_by_type": self._metrics.errors_by_type,
            "is_consuming": self._is_consuming,
            "is_healthy": self._is_started and self.consumer is not None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the consumer."""
        if not self._is_started or not self.consumer:
            return {
                "status": "unhealthy",
                "reason": "Consumer not started",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        try:
            # Get consumer group information
            cluster_metadata = await self.consumer.client.fetch_metadata()
            
            return {
                "status": "healthy",
                "consumer_group": self.consumer_group,
                "topics": self.topics,
                "is_consuming": self._is_consuming,
                "cluster_size": len(cluster_metadata.brokers),
                "metrics": self.get_metrics(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Example message handlers
async def job_message_handler(message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    """Example handler for job messages."""
    try:
        logger.info(f"Processing job message: {message.get('payload', {}).get('job_id')}")
        
        # Add your job processing logic here
        # For example: update job status in database, trigger notifications, etc.
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing job message: {e}")
        return False


async def result_message_handler(message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    """Example handler for result messages."""
    try:
        logger.info(f"Processing result message: {message.get('payload', {}).get('result_id')}")
        
        # Add your result processing logic here
        # For example: store results, update indexes, trigger downstream processes, etc.
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing result message: {e}")
        return False


async def error_message_handler(message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    """Example handler for error messages."""
    try:
        payload = message.get('payload', {})
        error_type = payload.get('error_type')
        error_message = payload.get('error_message')
        
        logger.error(f"Received error message - Type: {error_type}, Message: {error_message}")
        
        # Add your error handling logic here
        # For example: send alerts, update monitoring systems, etc.
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing error message: {e}")
        return False


async def default_message_handler(message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    """Default handler for unhandled message types."""
    try:
        logger.info(f"Processing unhandled message from {metadata['topic']}")
        
        # Add your default processing logic here
        # For example: log the message, forward to a dead letter queue, etc.
        
        return True
        
    except Exception as e:
        logger.error(f"Error in default message handler: {e}")
        return False


def create_tda_consumer(
    consumer_group: str = "tda-backend-group",
    topics: Optional[List[str]] = None
) -> KafkaConsumerService:
    """Create a TDA consumer with default message handlers."""
    config = KafkaConfig()
    
    if topics is None:
        topics = [
            config.topic.jobs_topic,
            config.topic.results_topic,
            config.topic.events_topic,
            config.topic.uploads_topic,
            config.topic.errors_topic
        ]
    
    # Create consumer
    consumer = KafkaConsumerService(
        consumer_group=consumer_group,
        topics=topics,
        config=config
    )
    
    # Register message handlers
    consumer.register_handler(MessageType.JOB_SUBMITTED, job_message_handler)
    consumer.register_handler(MessageType.JOB_STARTED, job_message_handler)
    consumer.register_handler(MessageType.JOB_COMPLETED, job_message_handler)
    consumer.register_handler(MessageType.JOB_FAILED, job_message_handler)
    consumer.register_handler(MessageType.JOB_CANCELLED, job_message_handler)
    
    consumer.register_handler(MessageType.RESULT_GENERATED, result_message_handler)
    consumer.register_handler(MessageType.RESULT_CACHED, result_message_handler)
    
    consumer.register_handler(MessageType.ERROR_OCCURRED, error_message_handler)
    consumer.register_handler(MessageType.WARNING_ISSUED, error_message_handler)
    
    consumer.register_default_handler(default_message_handler)
    
    return consumer

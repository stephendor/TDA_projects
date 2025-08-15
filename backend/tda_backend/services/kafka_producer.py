"""
Kafka Producer Service for TDA Backend.

This module provides asynchronous Kafka producer functionality with comprehensive
error handling, retry logic, monitoring, and message schema validation.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

import aiokafka
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError
from pydantic import BaseModel, Field

from ..config import settings
from .kafka_config import KafkaConfig


logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """TDA message types for different topics."""
    JOB_SUBMITTED = "job.submitted"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    
    RESULT_GENERATED = "result.generated"
    RESULT_CACHED = "result.cached"
    
    FILE_UPLOADED = "file.uploaded"
    FILE_VALIDATED = "file.validated"
    FILE_PROCESSED = "file.processed"
    
    ERROR_OCCURRED = "error.occurred"
    WARNING_ISSUED = "warning.issued"
    
    SYSTEM_HEALTH = "system.health"
    METRICS_UPDATE = "metrics.update"


@dataclass
class MessageMetadata:
    """Metadata for all TDA messages."""
    message_id: str
    message_type: MessageType
    timestamp: datetime
    source_service: str
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    version: str = "1.0"


class TDAMessage(BaseModel):
    """Base class for all TDA Kafka messages."""
    metadata: MessageMetadata = Field(..., description="Message metadata")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            MessageType: lambda v: v.value
        }


class JobMessage(TDAMessage):
    """Message for job lifecycle events."""
    
    @classmethod
    def create(
        cls,
        job_id: str,
        status: str,
        message_type: MessageType,
        **kwargs
    ) -> 'JobMessage':
        """Create a job message."""
        metadata = MessageMetadata(
            message_id=f"job-{job_id}-{int(time.time() * 1000)}",
            message_type=message_type,
            timestamp=datetime.now(timezone.utc),
            source_service="tda-backend",
            correlation_id=job_id
        )
        
        payload = {
            "job_id": job_id,
            "status": status,
            **kwargs
        }
        
        return cls(metadata=metadata, payload=payload)


class ResultMessage(TDAMessage):
    """Message for computation result events."""
    
    @classmethod
    def create(
        cls,
        job_id: str,
        result_id: str,
        result_type: str,
        message_type: MessageType,
        **kwargs
    ) -> 'ResultMessage':
        """Create a result message."""
        metadata = MessageMetadata(
            message_id=f"result-{result_id}-{int(time.time() * 1000)}",
            message_type=message_type,
            timestamp=datetime.now(timezone.utc),
            source_service="tda-backend",
            correlation_id=job_id
        )
        
        payload = {
            "job_id": job_id,
            "result_id": result_id,
            "result_type": result_type,
            **kwargs
        }
        
        return cls(metadata=metadata, payload=payload)


class FileMessage(TDAMessage):
    """Message for file processing events."""
    
    @classmethod
    def create(
        cls,
        file_id: str,
        filename: str,
        message_type: MessageType,
        **kwargs
    ) -> 'FileMessage':
        """Create a file message."""
        metadata = MessageMetadata(
            message_id=f"file-{file_id}-{int(time.time() * 1000)}",
            message_type=message_type,
            timestamp=datetime.now(timezone.utc),
            source_service="tda-backend"
        )
        
        payload = {
            "file_id": file_id,
            "filename": filename,
            **kwargs
        }
        
        return cls(metadata=metadata, payload=payload)


class ErrorMessage(TDAMessage):
    """Message for error and warning events."""
    
    @classmethod
    def create(
        cls,
        error_type: str,
        error_message: str,
        message_type: MessageType,
        **kwargs
    ) -> 'ErrorMessage':
        """Create an error message."""
        metadata = MessageMetadata(
            message_id=f"error-{int(time.time() * 1000)}",
            message_type=message_type,
            timestamp=datetime.now(timezone.utc),
            source_service="tda-backend"
        )
        
        payload = {
            "error_type": error_type,
            "error_message": error_message,
            **kwargs
        }
        
        return cls(metadata=metadata, payload=payload)


class KafkaProducerService:
    """Asynchronous Kafka Producer Service for TDA Backend."""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """Initialize the Kafka producer service."""
        self.config = config or KafkaConfig()
        self.producer: Optional[AIOKafkaProducer] = None
        self._is_started = False
        self._metrics = {
            "messages_sent": 0,
            "messages_failed": 0,
            "bytes_sent": 0,
            "send_duration_total": 0.0,
            "last_send_time": None,
            "errors_by_type": {}
        }
        
        # Topic mapping
        self.topic_mapping = {
            MessageType.JOB_SUBMITTED: self.config.topics.jobs_topic,
            MessageType.JOB_STARTED: self.config.topics.jobs_topic,
            MessageType.JOB_COMPLETED: self.config.topics.jobs_topic,
            MessageType.JOB_FAILED: self.config.topics.jobs_topic,
            MessageType.JOB_CANCELLED: self.config.topics.jobs_topic,
            
            MessageType.RESULT_GENERATED: self.config.topics.results_topic,
            MessageType.RESULT_CACHED: self.config.topics.results_topic,
            
            MessageType.FILE_UPLOADED: self.config.topics.events_topic,
            MessageType.FILE_VALIDATED: self.config.topics.events_topic,
            MessageType.FILE_PROCESSED: self.config.topics.events_topic,
            
            MessageType.ERROR_OCCURRED: self.config.topics.events_topic,
            MessageType.WARNING_ISSUED: self.config.topics.events_topic,
            
            MessageType.SYSTEM_HEALTH: self.config.topics.events_topic,
            MessageType.METRICS_UPDATE: self.config.topics.events_topic,
        }
    
    async def start(self) -> None:
        """Start the Kafka producer."""
        if self._is_started:
            logger.warning("Kafka producer is already started")
            return
        
        try:
            logger.info("Starting Kafka producer...")
            
            # Create producer with configuration
            producer_config = self.config.producer
            
            self.producer = AIOKafkaProducer(
                bootstrap_servers=producer_config.bootstrap_servers,
                client_id=producer_config.client_id,
                
                # Serialization
                value_serializer=self._serialize_message,
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                
                # Reliability settings
                acks=producer_config.acks,
                retries=producer_config.retries,
                retry_backoff_ms=producer_config.retry_backoff_ms,
                request_timeout_ms=producer_config.request_timeout_ms,
                delivery_timeout_ms=producer_config.delivery_timeout_ms,
                
                # Performance settings
                batch_size=producer_config.batch_size,
                linger_ms=producer_config.linger_ms,
                buffer_memory=producer_config.buffer_memory,
                max_request_size=producer_config.max_request_size,
                
                # Compression
                compression_type=producer_config.compression_type,
                
                # Monitoring
                enable_idempotence=producer_config.enable_idempotence,
            )
            
            await self.producer.start()
            self._is_started = True
            
            logger.info(
                f"Kafka producer started successfully. "
                f"Bootstrap servers: {producer_config.bootstrap_servers}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if not self._is_started or not self.producer:
            return
        
        try:
            logger.info("Stopping Kafka producer...")
            await self.producer.stop()
            self._is_started = False
            logger.info("Kafka producer stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Kafka producer: {e}")
            raise
    
    def _serialize_message(self, message: Union[Dict[str, Any], TDAMessage]) -> bytes:
        """Serialize message to JSON bytes."""
        if isinstance(message, TDAMessage):
            # Convert TDAMessage to dict with proper JSON encoding
            message_dict = message.dict()
            
            # Convert datetime objects to ISO format strings
            if "metadata" in message_dict and "timestamp" in message_dict["metadata"]:
                timestamp = message_dict["metadata"]["timestamp"]
                if isinstance(timestamp, datetime):
                    message_dict["metadata"]["timestamp"] = timestamp.isoformat()
                    
            # Convert enum values to strings
            if "metadata" in message_dict and "message_type" in message_dict["metadata"]:
                message_type = message_dict["metadata"]["message_type"]
                if hasattr(message_type, 'value'):
                    message_dict["metadata"]["message_type"] = message_type.value
                    
            return json.dumps(message_dict, default=str, ensure_ascii=False).encode('utf-8')
        else:
            return json.dumps(message, default=str, ensure_ascii=False).encode('utf-8')
    
    def _get_topic_for_message_type(self, message_type: MessageType) -> str:
        """Get the appropriate topic for a message type."""
        return self.topic_mapping.get(message_type, self.config.topic.events_topic)
    
    async def send_message(
        self,
        message: Union[Dict[str, Any], TDAMessage],
        topic: Optional[str] = None,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, bytes]] = None,
        timeout: Optional[float] = None
    ) -> bool:
        """Send a message to Kafka."""
        if not self._is_started or not self.producer:
            logger.error("Kafka producer is not started")
            return False
        
        start_time = time.time()
        
        try:
            # Determine topic
            if topic is None:
                if isinstance(message, TDAMessage):
                    topic = self._get_topic_for_message_type(message.metadata.message_type)
                else:
                    topic = self.config.topic.events_topic
            
            # Add default headers
            if headers is None:
                headers = {}
            
            headers.update({
                b'source': b'tda-backend',
                b'timestamp': str(int(time.time() * 1000)).encode('utf-8'),
                b'content-type': b'application/json'
            })
            
            # Send message
            future = await self.producer.send(
                topic=topic,
                value=message,
                key=key,
                partition=partition,
                headers=list(headers.items()) if headers else None
            )
            
            # Wait for delivery confirmation
            record_metadata = await future
            
            # Update metrics
            duration = time.time() - start_time
            message_size = len(self._serialize_message(message))
            
            self._metrics["messages_sent"] += 1
            self._metrics["bytes_sent"] += message_size
            self._metrics["send_duration_total"] += duration
            self._metrics["last_send_time"] = datetime.now(timezone.utc)
            
            logger.debug(
                f"Message sent successfully to {topic} "
                f"(partition: {record_metadata.partition}, "
                f"offset: {record_metadata.offset}, "
                f"duration: {duration:.3f}s)"
            )
            
            return True
            
        except KafkaTimeoutError:
            self._update_error_metrics("timeout")
            logger.error(f"Timeout sending message to {topic}")
            return False
            
        except KafkaError as e:
            self._update_error_metrics("kafka_error")
            logger.error(f"Kafka error sending message to {topic}: {e}")
            return False
            
        except Exception as e:
            self._update_error_metrics("unknown_error")
            logger.error(f"Unexpected error sending message to {topic}: {e}")
            return False
    
    def _update_error_metrics(self, error_type: str) -> None:
        """Update error metrics."""
        self._metrics["messages_failed"] += 1
        if error_type not in self._metrics["errors_by_type"]:
            self._metrics["errors_by_type"][error_type] = 0
        self._metrics["errors_by_type"][error_type] += 1
    
    async def send_job_message(
        self,
        job_id: str,
        status: str,
        message_type: MessageType,
        **kwargs
    ) -> bool:
        """Send a job lifecycle message."""
        message = JobMessage.create(
            job_id=job_id,
            status=status,
            message_type=message_type,
            **kwargs
        )
        
        return await self.send_message(
            message=message,
            key=job_id
        )
    
    async def send_result_message(
        self,
        job_id: str,
        result_id: str,
        result_type: str,
        message_type: MessageType,
        **kwargs
    ) -> bool:
        """Send a computation result message."""
        message = ResultMessage.create(
            job_id=job_id,
            result_id=result_id,
            result_type=result_type,
            message_type=message_type,
            **kwargs
        )
        
        return await self.send_message(
            message=message,
            key=job_id
        )
    
    async def send_file_message(
        self,
        file_id: str,
        filename: str,
        message_type: MessageType,
        **kwargs
    ) -> bool:
        """Send a file processing message."""
        message = FileMessage.create(
            file_id=file_id,
            filename=filename,
            message_type=message_type,
            **kwargs
        )
        
        return await self.send_message(
            message=message,
            key=file_id
        )
    
    async def send_error_message(
        self,
        error_type: str,
        error_message: str,
        message_type: MessageType = MessageType.ERROR_OCCURRED,
        **kwargs
    ) -> bool:
        """Send an error or warning message."""
        message = ErrorMessage.create(
            error_type=error_type,
            error_message=error_message,
            message_type=message_type,
            **kwargs
        )
        
        return await self.send_message(message=message)
    
    async def send_batch_messages(
        self,
        messages: List[Union[Dict[str, Any], TDAMessage]],
        topic: Optional[str] = None
    ) -> int:
        """Send multiple messages as a batch."""
        successful_sends = 0
        
        # Use asyncio.gather for concurrent sending
        tasks = []
        for message in messages:
            task = self.send_message(message=message, topic=topic)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, bool) and result:
                successful_sends += 1
            elif isinstance(result, Exception):
                logger.error(f"Batch message failed: {result}")
        
        logger.info(
            f"Batch send completed: {successful_sends}/{len(messages)} successful"
        )
        
        return successful_sends
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        avg_duration = (
            self._metrics["send_duration_total"] / self._metrics["messages_sent"]
            if self._metrics["messages_sent"] > 0 else 0
        )
        
        return {
            **self._metrics,
            "average_send_duration": avg_duration,
            "success_rate": (
                (self._metrics["messages_sent"] / 
                 (self._metrics["messages_sent"] + self._metrics["messages_failed"]))
                if (self._metrics["messages_sent"] + self._metrics["messages_failed"]) > 0
                else 0
            ),
            "is_healthy": self._is_started and self.producer is not None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the producer."""
        if not self._is_started or not self.producer:
            return {
                "status": "unhealthy",
                "reason": "Producer not started",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        try:
            # Try to get cluster metadata as a health check
            metadata = await self.producer.client.fetch_metadata()
            
            return {
                "status": "healthy",
                "cluster_size": len(metadata.brokers),
                "available_topics": len(metadata.topics),
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


# Singleton instance for global use
_kafka_producer: Optional[KafkaProducerService] = None


def get_kafka_producer() -> KafkaProducerService:
    """Get or create the global Kafka producer instance."""
    global _kafka_producer
    if _kafka_producer is None:
        _kafka_producer = KafkaProducerService()
    return _kafka_producer


async def initialize_kafka_producer() -> None:
    """Initialize the global Kafka producer."""
    producer = get_kafka_producer()
    if not producer._is_started:
        await producer.start()


async def shutdown_kafka_producer() -> None:
    """Shutdown the global Kafka producer."""
    global _kafka_producer
    if _kafka_producer and _kafka_producer._is_started:
        await _kafka_producer.stop()
        _kafka_producer = None

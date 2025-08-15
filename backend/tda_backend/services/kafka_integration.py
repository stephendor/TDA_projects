"""
Kafka Integration utilities for TDA Backend API endpoints.

This module provides helper functions and decorators to integrate Kafka
producer functionality seamlessly into FastAPI endpoints.
"""

import asyncio
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone

from .kafka_producer import (
    get_kafka_producer, 
    MessageType, 
    JobMessage, 
    ResultMessage, 
    FileMessage, 
    ErrorMessage
)
from ..config import settings


logger = logging.getLogger(__name__)


class KafkaIntegrationService:
    """Service for integrating Kafka messaging with TDA API operations."""
    
    def __init__(self):
        self.producer = get_kafka_producer()
        self.enabled = not settings.mock_kafka and not settings.testing
    
    async def send_job_submitted(
        self, 
        job_id: str, 
        user_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send job submitted notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_job_message(
            job_id=job_id,
            status="submitted",
            message_type=MessageType.JOB_SUBMITTED,
            user_id=user_id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_job_started(
        self, 
        job_id: str, 
        algorithm: str,
        **kwargs
    ) -> bool:
        """Send job started notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_job_message(
            job_id=job_id,
            status="running",
            message_type=MessageType.JOB_STARTED,
            algorithm=algorithm,
            started_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_job_completed(
        self, 
        job_id: str, 
        result_id: str,
        execution_time: float,
        **kwargs
    ) -> bool:
        """Send job completed notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_job_message(
            job_id=job_id,
            status="completed",
            message_type=MessageType.JOB_COMPLETED,
            result_id=result_id,
            execution_time=execution_time,
            completed_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_job_failed(
        self, 
        job_id: str, 
        error_message: str,
        error_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send job failed notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_job_message(
            job_id=job_id,
            status="failed",
            message_type=MessageType.JOB_FAILED,
            error_message=error_message,
            error_type=error_type,
            failed_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_result_generated(
        self, 
        job_id: str, 
        result_id: str,
        result_type: str,
        result_size: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Send result generated notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_result_message(
            job_id=job_id,
            result_id=result_id,
            result_type=result_type,
            message_type=MessageType.RESULT_GENERATED,
            result_size=result_size,
            generated_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_file_uploaded(
        self, 
        file_id: str, 
        filename: str,
        file_size: int,
        content_type: str,
        **kwargs
    ) -> bool:
        """Send file uploaded notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_file_message(
            file_id=file_id,
            filename=filename,
            message_type=MessageType.FILE_UPLOADED,
            file_size=file_size,
            content_type=content_type,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_file_validated(
        self, 
        file_id: str, 
        filename: str,
        validation_result: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Send file validated notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_file_message(
            file_id=file_id,
            filename=filename,
            message_type=MessageType.FILE_VALIDATED,
            validation_result=validation_result,
            validated_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_error_occurred(
        self, 
        error_type: str, 
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """Send error occurred notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_error_message(
            error_type=error_type,
            error_message=error_message,
            message_type=MessageType.ERROR_OCCURRED,
            context=context or {},
            occurred_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    async def send_warning(
        self, 
        warning_type: str, 
        warning_message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """Send warning notification."""
        if not self.enabled:
            return True
        
        return await self.producer.send_error_message(
            error_type=warning_type,
            error_message=warning_message,
            message_type=MessageType.WARNING_ISSUED,
            context=context or {},
            occurred_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )


# Singleton instance for global use
_kafka_integration: Optional[KafkaIntegrationService] = None


def get_kafka_integration() -> KafkaIntegrationService:
    """Get or create the global Kafka integration service."""
    global _kafka_integration
    if _kafka_integration is None:
        _kafka_integration = KafkaIntegrationService()
    return _kafka_integration


def kafka_notify(message_type: str):
    """
    Decorator for automatically sending Kafka notifications from API endpoints.
    
    Usage:
        @kafka_notify("job_submitted")
        async def create_job(job_data: JobCreate):
            # Your endpoint logic here
            return result
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            kafka = get_kafka_integration()
            
            try:
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Send notification based on message type
                if message_type == "job_submitted" and hasattr(result, 'job_id'):
                    asyncio.create_task(kafka.send_job_submitted(result.job_id))
                elif message_type == "job_completed" and hasattr(result, 'job_id'):
                    asyncio.create_task(kafka.send_job_completed(
                        result.job_id, 
                        getattr(result, 'result_id', ''),
                        getattr(result, 'execution_time', 0.0)
                    ))
                elif message_type == "file_uploaded" and hasattr(result, 'file_id'):
                    asyncio.create_task(kafka.send_file_uploaded(
                        result.file_id,
                        getattr(result, 'filename', ''),
                        getattr(result, 'file_size', 0),
                        getattr(result, 'content_type', '')
                    ))
                
                return result
                
            except Exception as e:
                # Send error notification
                asyncio.create_task(kafka.send_error_occurred(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"endpoint": func.__name__}
                ))
                raise
        
        return wrapper
    return decorator


async def send_background_notification(
    notification_func: Callable,
    *args,
    **kwargs
):
    """
    Send a Kafka notification in the background without blocking the response.
    
    Usage:
        await send_background_notification(
            kafka.send_job_submitted,
            job_id="123",
            user_id="user456"
        )
    """
    try:
        await notification_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Failed to send background notification: {e}")


class KafkaHealthChecker:
    """Health checker for Kafka integration."""
    
    def __init__(self):
        self.producer = get_kafka_producer()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check Kafka producer health."""
        try:
            health_info = await self.producer.health_check()
            return {
                "kafka_producer": {
                    "status": health_info["status"],
                    "details": health_info
                }
            }
        except Exception as e:
            return {
                "kafka_producer": {
                    "status": "unhealthy",
                    "error": str(e)
                }
            }
    
    async def check_metrics(self) -> Dict[str, Any]:
        """Get Kafka producer metrics."""
        try:
            return self.producer.get_metrics()
        except Exception as e:
            logger.error(f"Failed to get Kafka metrics: {e}")
            return {"error": str(e)}


# FastAPI Dependency for Kafka integration
def get_kafka_service() -> KafkaIntegrationService:
    """FastAPI dependency for Kafka integration service."""
    return get_kafka_integration()


def get_kafka_health_checker() -> KafkaHealthChecker:
    """FastAPI dependency for Kafka health checker."""
    return KafkaHealthChecker()

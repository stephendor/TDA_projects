"""
Kafka Message Schemas and Schema Registry Integration for TDA Backend.

This module provides message schema definitions, validation, evolution support,
and integration with Confluent Schema Registry for the TDA platform.
"""

import json
import logging
from typing import Dict, Any, Optional, Type, Union, List
from datetime import datetime
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..config import settings
from .kafka_producer import TDAMessage, MessageType


logger = logging.getLogger(__name__)


class SchemaFormat(str, Enum):
    """Supported schema formats."""
    AVRO = "AVRO"
    JSON_SCHEMA = "JSON"
    PROTOBUF = "PROTOBUF"


class SchemaCompatibility(str, Enum):
    """Schema compatibility levels."""
    BACKWARD = "BACKWARD"
    BACKWARD_TRANSITIVE = "BACKWARD_TRANSITIVE" 
    FORWARD = "FORWARD"
    FORWARD_TRANSITIVE = "FORWARD_TRANSITIVE"
    FULL = "FULL"
    FULL_TRANSITIVE = "FULL_TRANSITIVE"
    NONE = "NONE"


class TDAMessageSchema(BaseModel):
    """Base schema for all TDA Kafka messages."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            MessageType: lambda v: v.value
        }


class JobMessageSchema(TDAMessageSchema):
    """Schema for job lifecycle messages."""
    metadata: Dict[str, Any] = Field(..., description="Message metadata")
    payload: Dict[str, Any] = Field(..., description="Job payload")
    
    # Required payload fields
    job_id: Optional[str] = Field(None, alias="payload.job_id")
    status: Optional[str] = Field(None, alias="payload.status")
    algorithm: Optional[str] = Field(None, alias="payload.algorithm")
    priority: Optional[str] = Field(None, alias="payload.priority")
    
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for job messages."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "TDA Job Message",
            "description": "Schema for TDA job lifecycle messages",
            "required": ["metadata", "payload"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["message_id", "message_type", "timestamp", "source_service"],
                    "properties": {
                        "message_id": {"type": "string"},
                        "message_type": {
                            "type": "string",
                            "enum": [
                                "job.submitted", "job.started", "job.completed", 
                                "job.failed", "job.cancelled"
                            ]
                        },
                        "timestamp": {"type": "string", "format": "date-time"},
                        "source_service": {"type": "string"},
                        "correlation_id": {"type": ["string", "null"]},
                        "user_id": {"type": ["string", "null"]},
                        "session_id": {"type": ["string", "null"]},
                        "trace_id": {"type": ["string", "null"]},
                        "version": {"type": "string", "default": "1.0"}
                    }
                },
                "payload": {
                    "type": "object",
                    "required": ["job_id", "status"],
                    "properties": {
                        "job_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["submitted", "running", "completed", "failed", "cancelled"]
                        },
                        "algorithm": {"type": ["string", "null"]},
                        "priority": {"type": ["string", "null"]},
                        "user_id": {"type": ["string", "null"]},
                        "execution_time": {"type": ["number", "null"]},
                        "result_id": {"type": ["string", "null"]},
                        "error_message": {"type": ["string", "null"]},
                        "error_type": {"type": ["string", "null"]},
                        "submitted_at": {"type": ["string", "null"]},
                        "started_at": {"type": ["string", "null"]},
                        "completed_at": {"type": ["string", "null"]},
                        "failed_at": {"type": ["string", "null"]}
                    }
                }
            }
        }


class ResultMessageSchema(TDAMessageSchema):
    """Schema for result messages."""
    metadata: Dict[str, Any] = Field(..., description="Message metadata")
    payload: Dict[str, Any] = Field(..., description="Result payload")
    
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for result messages."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "TDA Result Message",
            "description": "Schema for TDA computation result messages",
            "required": ["metadata", "payload"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["message_id", "message_type", "timestamp", "source_service"],
                    "properties": {
                        "message_id": {"type": "string"},
                        "message_type": {
                            "type": "string",
                            "enum": ["result.generated", "result.cached"]
                        },
                        "timestamp": {"type": "string", "format": "date-time"},
                        "source_service": {"type": "string"},
                        "correlation_id": {"type": ["string", "null"]},
                        "version": {"type": "string", "default": "1.0"}
                    }
                },
                "payload": {
                    "type": "object",
                    "required": ["job_id", "result_id", "result_type"],
                    "properties": {
                        "job_id": {"type": "string"},
                        "result_id": {"type": "string"},
                        "result_type": {"type": "string"},
                        "result_size": {"type": ["number", "null"]},
                        "storage_path": {"type": ["string", "null"]},
                        "generated_at": {"type": ["string", "null"]},
                        "metadata": {"type": ["object", "null"]}
                    }
                }
            }
        }


class FileMessageSchema(TDAMessageSchema):
    """Schema for file processing messages."""
    metadata: Dict[str, Any] = Field(..., description="Message metadata")
    payload: Dict[str, Any] = Field(..., description="File payload")
    
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for file messages."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "TDA File Message",
            "description": "Schema for TDA file processing messages",
            "required": ["metadata", "payload"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["message_id", "message_type", "timestamp", "source_service"],
                    "properties": {
                        "message_id": {"type": "string"},
                        "message_type": {
                            "type": "string",
                            "enum": ["file.uploaded", "file.validated", "file.processed"]
                        },
                        "timestamp": {"type": "string", "format": "date-time"},
                        "source_service": {"type": "string"},
                        "version": {"type": "string", "default": "1.0"}
                    }
                },
                "payload": {
                    "type": "object",
                    "required": ["file_id", "filename"],
                    "properties": {
                        "file_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "file_size": {"type": ["number", "null"]},
                        "content_type": {"type": ["string", "null"]},
                        "checksum": {"type": ["string", "null"]},
                        "uploaded_at": {"type": ["string", "null"]},
                        "validated_at": {"type": ["string", "null"]},
                        "processed_at": {"type": ["string", "null"]},
                        "validation_result": {"type": ["object", "null"]},
                        "processing_result": {"type": ["object", "null"]}
                    }
                }
            }
        }


class ErrorMessageSchema(TDAMessageSchema):
    """Schema for error and warning messages."""
    metadata: Dict[str, Any] = Field(..., description="Message metadata")
    payload: Dict[str, Any] = Field(..., description="Error payload")
    
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for error messages."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "TDA Error Message",
            "description": "Schema for TDA error and warning messages",
            "required": ["metadata", "payload"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["message_id", "message_type", "timestamp", "source_service"],
                    "properties": {
                        "message_id": {"type": "string"},
                        "message_type": {
                            "type": "string",
                            "enum": ["error.occurred", "warning.issued"]
                        },
                        "timestamp": {"type": "string", "format": "date-time"},
                        "source_service": {"type": "string"},
                        "version": {"type": "string", "default": "1.0"}
                    }
                },
                "payload": {
                    "type": "object",
                    "required": ["error_type", "error_message"],
                    "properties": {
                        "error_type": {"type": "string"},
                        "error_message": {"type": "string"},
                        "context": {"type": ["object", "null"]},
                        "stack_trace": {"type": ["string", "null"]},
                        "occurred_at": {"type": ["string", "null"]},
                        "severity": {"type": ["string", "null"]},
                        "component": {"type": ["string", "null"]}
                    }
                }
            }
        }


class SchemaRegistry:
    """Client for interacting with Confluent Schema Registry."""
    
    def __init__(self, base_url: str = None):
        """Initialize the schema registry client."""
        self.base_url = base_url or f"http://localhost:{settings.schema_registry_port}"
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def register_schema(
        self,
        subject: str,
        schema: Dict[str, Any],
        schema_type: SchemaFormat = SchemaFormat.JSON_SCHEMA
    ) -> int:
        """Register a new schema version."""
        try:
            response = await self.client.post(
                f"{self.base_url}/subjects/{subject}/versions",
                json={
                    "schema": json.dumps(schema),
                    "schemaType": schema_type.value
                }
            )
            response.raise_for_status()
            result = response.json()
            schema_id = result.get("id")
            
            logger.info(f"Registered schema for subject {subject} with ID {schema_id}")
            return schema_id
            
        except Exception as e:
            logger.error(f"Failed to register schema for {subject}: {e}")
            raise
    
    async def get_schema(self, subject: str, version: str = "latest") -> Dict[str, Any]:
        """Get schema by subject and version."""
        try:
            response = await self.client.get(
                f"{self.base_url}/subjects/{subject}/versions/{version}"
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Schema not found for {subject}:{version}")
                return None
            raise
            
        except Exception as e:
            logger.error(f"Failed to get schema for {subject}:{version}: {e}")
            raise
    
    async def check_compatibility(
        self,
        subject: str,
        schema: Dict[str, Any],
        version: str = "latest"
    ) -> bool:
        """Check if schema is compatible with existing version."""
        try:
            response = await self.client.post(
                f"{self.base_url}/compatibility/subjects/{subject}/versions/{version}",
                json={"schema": json.dumps(schema)}
            )
            response.raise_for_status()
            result = response.json()
            return result.get("is_compatible", False)
            
        except Exception as e:
            logger.error(f"Failed to check compatibility for {subject}: {e}")
            return False
    
    async def set_compatibility(
        self,
        subject: str,
        compatibility: SchemaCompatibility
    ) -> bool:
        """Set compatibility level for a subject."""
        try:
            response = await self.client.put(
                f"{self.base_url}/config/{subject}",
                json={"compatibility": compatibility.value}
            )
            response.raise_for_status()
            logger.info(f"Set compatibility for {subject} to {compatibility.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set compatibility for {subject}: {e}")
            return False
    
    async def list_subjects(self) -> List[str]:
        """List all registered subjects."""
        try:
            response = await self.client.get(f"{self.base_url}/subjects")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to list subjects: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MessageValidator:
    """Validator for TDA Kafka messages using schemas."""
    
    def __init__(self, schema_registry: Optional[SchemaRegistry] = None):
        """Initialize the message validator."""
        self.schema_registry = schema_registry or SchemaRegistry()
        self.schemas = {
            MessageType.JOB_SUBMITTED: JobMessageSchema,
            MessageType.JOB_STARTED: JobMessageSchema,
            MessageType.JOB_COMPLETED: JobMessageSchema,
            MessageType.JOB_FAILED: JobMessageSchema,
            MessageType.JOB_CANCELLED: JobMessageSchema,
            MessageType.RESULT_GENERATED: ResultMessageSchema,
            MessageType.RESULT_CACHED: ResultMessageSchema,
            MessageType.FILE_UPLOADED: FileMessageSchema,
            MessageType.FILE_VALIDATED: FileMessageSchema,
            MessageType.FILE_PROCESSED: FileMessageSchema,
            MessageType.ERROR_OCCURRED: ErrorMessageSchema,
            MessageType.WARNING_ISSUED: ErrorMessageSchema,
        }
    
    async def validate_message(
        self,
        message: Union[TDAMessage, Dict[str, Any]],
        message_type: Optional[MessageType] = None
    ) -> bool:
        """Validate a message against its schema."""
        try:
            # Extract message type if not provided
            if message_type is None:
                if hasattr(message, 'metadata') and hasattr(message.metadata, 'message_type'):
                    message_type = message.metadata.message_type
                elif isinstance(message, dict) and 'metadata' in message:
                    message_type_str = message['metadata'].get('message_type')
                    if message_type_str:
                        message_type = MessageType(message_type_str)
            
            if message_type is None:
                logger.warning("Cannot validate message without message type")
                return False
            
            # Get schema class
            schema_class = self.schemas.get(message_type)
            if schema_class is None:
                logger.warning(f"No schema found for message type: {message_type}")
                return False
            
            # Convert to dict for validation
            if isinstance(message, TDAMessage):
                message_dict = message.dict()
            else:
                message_dict = message
            
            # Validate using Pydantic
            schema_class(**message_dict)
            return True
            
        except ValidationError as e:
            logger.error(f"Message validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False
    
    async def register_all_schemas(self) -> bool:
        """Register all TDA schemas in the Schema Registry."""
        try:
            schema_mappings = {
                "tda-job-messages": JobMessageSchema.get_json_schema(),
                "tda-result-messages": ResultMessageSchema.get_json_schema(),
                "tda-file-messages": FileMessageSchema.get_json_schema(),
                "tda-error-messages": ErrorMessageSchema.get_json_schema(),
            }
            
            success_count = 0
            for subject, schema in schema_mappings.items():
                try:
                    # Set compatibility level
                    await self.schema_registry.set_compatibility(
                        subject, 
                        SchemaCompatibility.BACKWARD
                    )
                    
                    # Register schema
                    schema_id = await self.schema_registry.register_schema(
                        subject,
                        schema,
                        SchemaFormat.JSON_SCHEMA
                    )
                    
                    if schema_id:
                        success_count += 1
                        logger.info(f"Registered schema {subject} with ID {schema_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to register schema {subject}: {e}")
            
            logger.info(f"Registered {success_count}/{len(schema_mappings)} schemas")
            return success_count == len(schema_mappings)
            
        except Exception as e:
            logger.error(f"Failed to register schemas: {e}")
            return False


# Global instances
_schema_registry: Optional[SchemaRegistry] = None
_message_validator: Optional[MessageValidator] = None


def get_schema_registry() -> SchemaRegistry:
    """Get or create the global schema registry client."""
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = SchemaRegistry()
    return _schema_registry


def get_message_validator() -> MessageValidator:
    """Get or create the global message validator."""
    global _message_validator
    if _message_validator is None:
        _message_validator = MessageValidator(get_schema_registry())
    return _message_validator


async def initialize_schemas() -> bool:
    """Initialize and register all schemas."""
    try:
        validator = get_message_validator()
        success = await validator.register_all_schemas()
        
        if success:
            logger.info("✅ All schemas registered successfully")
        else:
            logger.warning("⚠️ Some schemas failed to register")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize schemas: {e}")
        return False


async def cleanup_schemas():
    """Cleanup schema registry resources."""
    global _schema_registry, _message_validator
    
    if _schema_registry:
        await _schema_registry.close()
        _schema_registry = None
        
    _message_validator = None
    logger.info("Schema registry resources cleaned up")

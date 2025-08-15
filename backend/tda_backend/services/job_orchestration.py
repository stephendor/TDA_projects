"""
Job Orchestration Service for TDA Backend - Comprehensive Event-Driven Job Management.

This service provides complete job lifecycle management with event-driven orchestration,
state persistence, retry logic, timeout handling, and seamless integration with the
existing TDA backend architecture.

Features:
    - Complete job state management (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
    - Event-driven job orchestration via Kafka integration
    - Job retry logic with exponential backoff
    - Timeout handling and resource cleanup
    - Job dependency management framework
    - Comprehensive metrics and monitoring
    - Thread-safe operations with asyncio
    - Production-ready error handling and recovery
    - Abstract storage interface for database integration
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from collections import defaultdict
import json

from pydantic import BaseModel, Field, field_validator
from ..models import JobStatus, TDAAlgorithm, TDAComputationRequest, TDAResults
from ..config import get_settings
from .kafka_integration import KafkaIntegrationService, get_kafka_integration
from .kafka_producer import MessageType
from .tda_service import TDAService, get_tda_service


logger = logging.getLogger(__name__)


class JobPriority(str, Enum):
    """Job priority levels for scheduling."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobFailureReason(str, Enum):
    """Enumeration of job failure reasons."""
    VALIDATION_ERROR = "validation_error"
    COMPUTATION_ERROR = "computation_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    DEPENDENCY_ERROR = "dependency_error"
    RESOURCE_ERROR = "resource_error"
    UNKNOWN_ERROR = "unknown_error"


class RetryPolicy(BaseModel):
    """Configuration for job retry behavior."""
    max_attempts: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
    initial_delay: float = Field(1.0, gt=0, description="Initial retry delay in seconds")
    backoff_multiplier: float = Field(2.0, gt=1, description="Exponential backoff multiplier")
    max_delay: float = Field(300.0, gt=0, description="Maximum retry delay in seconds")
    jitter: bool = Field(True, description="Add random jitter to retry delays")


@dataclass
class JobMetrics:
    """Job execution metrics."""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    attempt_count: int = 0
    retry_count: int = 0
    
    execution_time: Optional[float] = None
    queue_time: Optional[float] = None
    total_time: Optional[float] = None
    
    memory_peak_mb: Optional[float] = None
    cpu_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "execution_time": self.execution_time,
            "queue_time": self.queue_time,
            "total_time": self.total_time,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_time": self.cpu_time
        }


@dataclass 
class JobContext:
    """Complete job context and state."""
    job_id: str
    user_id: Optional[str]
    status: JobStatus
    algorithm: TDAAlgorithm
    priority: JobPriority
    
    # Job configuration
    computation_request: TDAComputationRequest
    timeout_seconds: Optional[float] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    
    # Execution state
    current_attempt: int = 0
    last_error: Optional[str] = None
    last_error_type: Optional[JobFailureReason] = None
    stack_trace: Optional[str] = None
    
    # Results and progress
    result: Optional[TDAResults] = None
    progress: float = 0.0
    progress_message: Optional[str] = None
    
    # Dependencies and metadata
    depends_on: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Metrics
    metrics: JobMetrics = field(default_factory=JobMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job context to dictionary for storage/serialization."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "algorithm": self.algorithm.value,
            "priority": self.priority.value,
            "computation_request": self.computation_request.dict(),
            "timeout_seconds": self.timeout_seconds,
            "retry_policy": self.retry_policy.dict(),
            "current_attempt": self.current_attempt,
            "last_error": self.last_error,
            "last_error_type": self.last_error_type.value if self.last_error_type else None,
            "stack_trace": self.stack_trace,
            "result": self.result.dict() if self.result else None,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "depends_on": list(self.depends_on),
            "metadata": self.metadata,
            "tags": list(self.tags),
            "metrics": self.metrics.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobContext":
        """Create job context from dictionary."""
        # TODO: Implement proper deserialization when database integration is added
        raise NotImplementedError("JobContext.from_dict will be implemented with database integration")


class JobStorage(ABC):
    """Abstract interface for job state persistence."""
    
    @abstractmethod
    async def save_job(self, job: JobContext) -> bool:
        """Save job state to storage."""
        pass
    
    @abstractmethod
    async def load_job(self, job_id: str) -> Optional[JobContext]:
        """Load job state from storage."""
        pass
    
    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """Delete job from storage."""
        pass
    
    @abstractmethod
    async def list_jobs(self, status: Optional[JobStatus] = None, 
                       user_id: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> List[JobContext]:
        """List jobs with optional filtering."""
        pass
    
    @abstractmethod
    async def get_jobs_by_status(self, status: JobStatus) -> List[JobContext]:
        """Get all jobs with specific status."""
        pass


class InMemoryJobStorage(JobStorage):
    """In-memory job storage implementation for development/testing."""
    
    def __init__(self):
        self.jobs: Dict[str, JobContext] = {}
        self._lock = asyncio.Lock()
    
    async def save_job(self, job: JobContext) -> bool:
        """Save job to in-memory storage."""
        async with self._lock:
            self.jobs[job.job_id] = job
            return True
    
    async def load_job(self, job_id: str) -> Optional[JobContext]:
        """Load job from in-memory storage."""
        async with self._lock:
            return self.jobs.get(job_id)
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete job from in-memory storage."""
        async with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                return True
            return False
    
    async def list_jobs(self, status: Optional[JobStatus] = None, 
                       user_id: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> List[JobContext]:
        """List jobs with filtering."""
        async with self._lock:
            jobs = list(self.jobs.values())
            
            # Apply filters
            if status:
                jobs = [j for j in jobs if j.status == status]
            if user_id:
                jobs = [j for j in jobs if j.user_id == user_id]
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.metrics.created_at, reverse=True)
            
            # Apply pagination
            return jobs[offset:offset + limit]
    
    async def get_jobs_by_status(self, status: JobStatus) -> List[JobContext]:
        """Get all jobs with specific status."""
        async with self._lock:
            return [job for job in self.jobs.values() if job.status == status]


class JobOrchestrationMetrics:
    """Comprehensive metrics for job orchestration."""
    
    def __init__(self):
        self.job_counts = defaultdict(int)
        self.completion_times = []
        self.failure_counts = defaultdict(int)
        self.retry_counts = defaultdict(int)
        self.queue_lengths = defaultdict(list)
        self.algorithm_stats = defaultdict(lambda: defaultdict(int))
        
        # Performance metrics
        self.total_jobs_processed = 0
        self.total_execution_time = 0.0
        self.peak_concurrent_jobs = 0
        self.current_concurrent_jobs = 0
        
        self._lock = asyncio.Lock()
    
    async def record_job_created(self, job: JobContext):
        """Record job creation."""
        async with self._lock:
            self.job_counts[job.status.value] += 1
            self.algorithm_stats[job.algorithm.value]["created"] += 1
    
    async def record_job_started(self, job: JobContext):
        """Record job start."""
        async with self._lock:
            self.job_counts[JobStatus.PENDING.value] -= 1
            self.job_counts[JobStatus.RUNNING.value] += 1
            self.current_concurrent_jobs += 1
            self.peak_concurrent_jobs = max(self.peak_concurrent_jobs, self.current_concurrent_jobs)
            
            if job.metrics.queue_time:
                self.queue_lengths[job.algorithm.value].append(job.metrics.queue_time)
    
    async def record_job_completed(self, job: JobContext):
        """Record job completion."""
        async with self._lock:
            self.job_counts[JobStatus.RUNNING.value] -= 1
            self.job_counts[JobStatus.COMPLETED.value] += 1
            self.current_concurrent_jobs -= 1
            self.total_jobs_processed += 1
            
            if job.metrics.execution_time:
                self.completion_times.append(job.metrics.execution_time)
                self.total_execution_time += job.metrics.execution_time
            
            self.algorithm_stats[job.algorithm.value]["completed"] += 1
    
    async def record_job_failed(self, job: JobContext):
        """Record job failure."""
        async with self._lock:
            self.job_counts[JobStatus.RUNNING.value] -= 1
            self.job_counts[JobStatus.FAILED.value] += 1
            self.current_concurrent_jobs -= 1
            
            if job.last_error_type:
                self.failure_counts[job.last_error_type.value] += 1
            
            self.retry_counts[job.current_attempt] += 1
            self.algorithm_stats[job.algorithm.value]["failed"] += 1
    
    async def record_job_cancelled(self, job: JobContext):
        """Record job cancellation."""
        async with self._lock:
            old_status = job.status.value
            if old_status in self.job_counts:
                self.job_counts[old_status] -= 1
            self.job_counts[JobStatus.CANCELLED.value] += 1
            
            if old_status == JobStatus.RUNNING.value:
                self.current_concurrent_jobs -= 1
            
            self.algorithm_stats[job.algorithm.value]["cancelled"] += 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        async with self._lock:
            avg_completion_time = (
                sum(self.completion_times) / len(self.completion_times)
                if self.completion_times else 0.0
            )
            
            avg_queue_times = {}
            for alg, times in self.queue_lengths.items():
                avg_queue_times[alg] = sum(times) / len(times) if times else 0.0
            
            return {
                "job_counts": dict(self.job_counts),
                "performance": {
                    "total_jobs_processed": self.total_jobs_processed,
                    "total_execution_time": self.total_execution_time,
                    "avg_completion_time": avg_completion_time,
                    "current_concurrent_jobs": self.current_concurrent_jobs,
                    "peak_concurrent_jobs": self.peak_concurrent_jobs
                },
                "failures": dict(self.failure_counts),
                "retries": dict(self.retry_counts),
                "algorithms": dict(self.algorithm_stats),
                "queue_times": avg_queue_times
            }


class JobOrchestrationService:
    """
    Comprehensive job orchestration service with event-driven architecture.
    
    Provides complete job lifecycle management with state persistence, retry logic,
    timeout handling, and seamless integration with Kafka messaging and TDA computation.
    """
    
    def __init__(self, 
                 storage: Optional[JobStorage] = None,
                 kafka_service: Optional[KafkaIntegrationService] = None,
                 tda_service: Optional[TDAService] = None,
                 settings = None):
        """Initialize the job orchestration service."""
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Service dependencies
        self.storage = storage or InMemoryJobStorage()
        self.kafka = kafka_service or get_kafka_integration()
        self.tda_service = tda_service or get_tda_service()
        
        # Metrics and monitoring
        self.metrics = JobOrchestrationMetrics()
        self.health_status = {"status": "healthy", "last_check": datetime.now(timezone.utc)}
        
        # Job execution tracking
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.job_timeouts: Dict[str, asyncio.Task] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.max_concurrent_jobs = getattr(self.settings, 'max_concurrent_jobs', 10)
        self.default_timeout = getattr(self.settings, 'job_default_timeout', 3600)  # 1 hour
        self.cleanup_interval = getattr(self.settings, 'job_cleanup_interval', 300)  # 5 minutes
        
        # Control flags
        self._shutdown = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._state_lock = asyncio.Lock()
        
        self.logger.info("JobOrchestrationService initialized")
    
    async def start(self):
        """Start the job orchestration service."""
        self.logger.info("Starting JobOrchestrationService...")
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        # Recover any running jobs from storage
        await self._recover_jobs()
        
        self.logger.info("JobOrchestrationService started successfully")
    
    async def stop(self):
        """Stop the job orchestration service."""
        self.logger.info("Stopping JobOrchestrationService...")
        
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active jobs
        for job_id, task in list(self.active_jobs.items()):
            await self._cancel_job_execution(job_id, "Service shutdown")
        
        # Cancel all timeout tasks
        for timeout_task in self.job_timeouts.values():
            timeout_task.cancel()
        
        self.logger.info("JobOrchestrationService stopped")
    
    async def create_job(self,
                        computation_request: TDAComputationRequest,
                        user_id: Optional[str] = None,
                        priority: JobPriority = JobPriority.NORMAL,
                        timeout_seconds: Optional[float] = None,
                        retry_policy: Optional[RetryPolicy] = None,
                        depends_on: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        tags: Optional[List[str]] = None) -> str:
        """Create a new job for TDA computation."""
        
        job_id = str(uuid4())
        
        try:
            # Create job context
            job = JobContext(
                job_id=job_id,
                user_id=user_id,
                status=JobStatus.PENDING,
                algorithm=computation_request.config.algorithm,
                priority=priority,
                computation_request=computation_request,
                timeout_seconds=timeout_seconds or self.default_timeout,
                retry_policy=retry_policy or RetryPolicy(),
                depends_on=set(depends_on or []),
                metadata=metadata or {},
                tags=set(tags or [])
            )
            
            # Validate dependencies
            await self._validate_dependencies(job)
            
            # Save to storage
            await self.storage.save_job(job)
            
            # Update dependency graph
            async with self._state_lock:
                for dep_id in job.depends_on:
                    self.dependency_graph[dep_id].add(job_id)
            
            # Record metrics
            await self.metrics.record_job_created(job)
            
            # Send Kafka notification
            asyncio.create_task(self.kafka.send_job_submitted(
                job_id=job_id,
                user_id=user_id,
                algorithm=job.algorithm.value,
                priority=priority.value,
                metadata=metadata
            ))
            
            self.logger.info(f"Job {job_id} created successfully with algorithm {job.algorithm.value}")
            
            # Try to start the job if dependencies are satisfied
            await self._try_start_job(job_id)
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to create job: {str(e)}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[JobContext]:
        """Get job by ID."""
        return await self.storage.load_job(job_id)
    
    async def cancel_job(self, job_id: str, reason: str = "User cancelled") -> bool:
        """Cancel a job."""
        try:
            job = await self.storage.load_job(job_id)
            if not job:
                self.logger.warning(f"Job {job_id} not found for cancellation")
                return False
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                self.logger.warning(f"Job {job_id} already in terminal state: {job.status}")
                return False
            
            # Update job status
            old_status = job.status
            job.status = JobStatus.CANCELLED
            job.metrics.cancelled_at = datetime.now(timezone.utc)
            job.last_error = reason
            
            # Cancel active execution
            if job_id in self.active_jobs:
                await self._cancel_job_execution(job_id, reason)
            
            # Save updated state
            await self.storage.save_job(job)
            
            # Record metrics
            await self.metrics.record_job_cancelled(job)
            
            # Send Kafka notification
            asyncio.create_task(self.kafka.send_job_failed(
                job_id=job_id,
                error_message=f"Job cancelled: {reason}",
                error_type="cancellation"
            ))
            
            self.logger.info(f"Job {job_id} cancelled: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            return False
    
    async def update_job_progress(self, job_id: str, progress: float, message: Optional[str] = None):
        """Update job progress."""
        try:
            job = await self.storage.load_job(job_id)
            if job and job.status == JobStatus.RUNNING:
                job.progress = max(0.0, min(1.0, progress))
                if message:
                    job.progress_message = message
                
                await self.storage.save_job(job)
                
                # Could send progress updates via Kafka if needed
                self.logger.debug(f"Job {job_id} progress: {progress:.1%} - {message}")
        
        except Exception as e:
            self.logger.error(f"Failed to update job progress {job_id}: {str(e)}")
    
    async def list_jobs(self, status: Optional[JobStatus] = None,
                       user_id: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> List[JobContext]:
        """List jobs with optional filtering."""
        return await self.storage.list_jobs(status, user_id, limit, offset)
    
    async def get_job_metrics(self) -> Dict[str, Any]:
        """Get comprehensive job metrics."""
        return await self.metrics.get_metrics()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Check storage
            storage_healthy = True
            try:
                # Try to perform a simple storage operation
                test_jobs = await self.storage.list_jobs(limit=1)
                storage_healthy = True
            except Exception as e:
                storage_healthy = False
                self.logger.error(f"Storage health check failed: {e}")
            
            # Check active jobs
            active_job_count = len(self.active_jobs)
            
            # Check TDA service
            tda_healthy = True
            try:
                tda_health = await self.tda_service.health_check()
                tda_healthy = tda_health.get("status") == "healthy"
            except Exception as e:
                tda_healthy = False
                self.logger.error(f"TDA service health check failed: {e}")
            
            # Overall health
            overall_status = "healthy"
            if not storage_healthy or not tda_healthy:
                overall_status = "unhealthy"
            elif active_job_count > self.max_concurrent_jobs * 0.9:
                overall_status = "degraded"
            
            self.health_status = {
                "status": overall_status,
                "last_check": datetime.now(timezone.utc),
                "components": {
                    "storage": "healthy" if storage_healthy else "unhealthy",
                    "tda_service": "healthy" if tda_healthy else "unhealthy",
                    "kafka": "healthy",  # Assume healthy if no errors
                },
                "metrics": {
                    "active_jobs": active_job_count,
                    "max_concurrent_jobs": self.max_concurrent_jobs,
                    "job_capacity_usage": active_job_count / self.max_concurrent_jobs
                }
            }
            
            return self.health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "last_check": datetime.now(timezone.utc),
                "error": str(e)
            }
    
    # Event handlers for Kafka integration
    async def handle_job_submitted(self, message: Dict[str, Any]):
        """Handle JOB_SUBMITTED Kafka event."""
        try:
            job_id = message.get("payload", {}).get("job_id")
            if job_id:
                await self._try_start_job(job_id)
        except Exception as e:
            self.logger.error(f"Error handling job_submitted event: {e}")
    
    async def handle_job_started(self, message: Dict[str, Any]):
        """Handle JOB_STARTED Kafka event."""
        try:
            job_id = message.get("payload", {}).get("job_id")
            if job_id:
                job = await self.storage.load_job(job_id)
                if job:
                    await self.metrics.record_job_started(job)
        except Exception as e:
            self.logger.error(f"Error handling job_started event: {e}")
    
    async def handle_job_completed(self, message: Dict[str, Any]):
        """Handle JOB_COMPLETED Kafka event."""
        try:
            payload = message.get("payload", {})
            job_id = payload.get("job_id")
            result_id = payload.get("result_id")
            
            if job_id:
                await self._complete_job_dependencies(job_id)
        except Exception as e:
            self.logger.error(f"Error handling job_completed event: {e}")
    
    async def handle_job_failed(self, message: Dict[str, Any]):
        """Handle JOB_FAILED Kafka event."""
        try:
            payload = message.get("payload", {})
            job_id = payload.get("job_id")
            error_message = payload.get("error_message")
            
            if job_id:
                await self._handle_job_failure_dependencies(job_id, error_message)
        except Exception as e:
            self.logger.error(f"Error handling job_failed event: {e}")
    
    # Private methods
    async def _validate_dependencies(self, job: JobContext):
        """Validate job dependencies exist and are valid."""
        for dep_id in job.depends_on:
            dep_job = await self.storage.load_job(dep_id)
            if not dep_job:
                raise ValueError(f"Dependency job {dep_id} does not exist")
            
            # Check for circular dependencies (simple check)
            if job.job_id in dep_job.depends_on:
                raise ValueError(f"Circular dependency detected between {job.job_id} and {dep_id}")
    
    async def _try_start_job(self, job_id: str):
        """Try to start a job if dependencies are satisfied."""
        try:
            job = await self.storage.load_job(job_id)
            if not job or job.status != JobStatus.PENDING:
                return
            
            # Check if dependencies are satisfied
            if not await self._dependencies_satisfied(job):
                self.logger.debug(f"Job {job_id} dependencies not satisfied, waiting...")
                return
            
            # Check concurrent job limit
            async with self._state_lock:
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    self.logger.debug(f"Job {job_id} waiting for available slot...")
                    return
            
            # Start the job
            await self._start_job_execution(job)
            
        except Exception as e:
            self.logger.error(f"Failed to start job {job_id}: {str(e)}")
    
    async def _dependencies_satisfied(self, job: JobContext) -> bool:
        """Check if all job dependencies are satisfied."""
        for dep_id in job.depends_on:
            dep_job = await self.storage.load_job(dep_id)
            if not dep_job or dep_job.status != JobStatus.COMPLETED:
                return False
        return True
    
    async def _start_job_execution(self, job: JobContext):
        """Start executing a job."""
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.metrics.started_at = datetime.now(timezone.utc)
            job.metrics.attempt_count += 1
            
            if job.metrics.started_at and job.metrics.created_at:
                job.metrics.queue_time = (
                    job.metrics.started_at - job.metrics.created_at
                ).total_seconds()
            
            await self.storage.save_job(job)
            
            # Start execution task
            task = asyncio.create_task(self._execute_job(job))
            
            async with self._state_lock:
                self.active_jobs[job.job_id] = task
            
            # Start timeout task if configured
            if job.timeout_seconds:
                timeout_task = asyncio.create_task(
                    self._handle_job_timeout(job.job_id, job.timeout_seconds)
                )
                self.job_timeouts[job.job_id] = timeout_task
            
            # Send Kafka notification
            asyncio.create_task(self.kafka.send_job_started(
                job_id=job.job_id,
                algorithm=job.algorithm.value,
                user_id=job.user_id,
                priority=job.priority.value
            ))
            
            self.logger.info(f"Job {job.job_id} execution started")
            
        except Exception as e:
            self.logger.error(f"Failed to start job execution {job.job_id}: {str(e)}")
            await self._handle_job_failure(job, JobFailureReason.UNKNOWN_ERROR, str(e))
    
    async def _execute_job(self, job: JobContext):
        """Execute the actual job computation."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing job {job.job_id} with algorithm {job.algorithm.value}")
            
            # Update progress
            await self.update_job_progress(job.job_id, 0.1, "Starting computation")
            
            # Execute TDA computation
            result = await self.tda_service.compute_tda(job.computation_request)
            
            # Update progress
            await self.update_job_progress(job.job_id, 0.9, "Computation completed, finalizing")
            
            # Complete the job
            await self._complete_job(job, result, time.time() - start_time)
            
        except asyncio.CancelledError:
            self.logger.info(f"Job {job.job_id} execution cancelled")
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Job {job.job_id} execution failed after {execution_time:.2f}s: {str(e)}")
            
            # Determine failure reason
            failure_reason = JobFailureReason.COMPUTATION_ERROR
            if "validation" in str(e).lower():
                failure_reason = JobFailureReason.VALIDATION_ERROR
            elif "memory" in str(e).lower():
                failure_reason = JobFailureReason.MEMORY_ERROR
            
            await self._handle_job_failure(job, failure_reason, str(e), traceback.format_exc())
        finally:
            # Cleanup
            async with self._state_lock:
                self.active_jobs.pop(job.job_id, None)
            
            timeout_task = self.job_timeouts.pop(job.job_id, None)
            if timeout_task:
                timeout_task.cancel()
    
    async def _complete_job(self, job: JobContext, result: TDAResults, execution_time: float):
        """Mark job as completed."""
        try:
            # Update job state
            job.status = JobStatus.COMPLETED
            job.result = result
            job.progress = 1.0
            job.progress_message = "Completed successfully"
            job.metrics.completed_at = datetime.now(timezone.utc)
            job.metrics.execution_time = execution_time
            
            if job.metrics.completed_at and job.metrics.created_at:
                job.metrics.total_time = (
                    job.metrics.completed_at - job.metrics.created_at
                ).total_seconds()
            
            await self.storage.save_job(job)
            
            # Record metrics
            await self.metrics.record_job_completed(job)
            
            # Send Kafka notification
            asyncio.create_task(self.kafka.send_job_completed(
                job_id=job.job_id,
                result_id=str(uuid4()),  # Would be actual result ID in production
                execution_time=execution_time,
                user_id=job.user_id
            ))
            
            self.logger.info(f"Job {job.job_id} completed successfully in {execution_time:.2f}s")
            
            # Try to start dependent jobs
            await self._complete_job_dependencies(job.job_id)
            
        except Exception as e:
            self.logger.error(f"Failed to complete job {job.job_id}: {str(e)}")
    
    async def _handle_job_failure(self, job: JobContext, failure_reason: JobFailureReason, 
                                 error_message: str, stack_trace: Optional[str] = None):
        """Handle job failure with retry logic."""
        try:
            # Update job state
            job.last_error = error_message
            job.last_error_type = failure_reason
            job.stack_trace = stack_trace
            job.current_attempt += 1
            
            # Check if we should retry
            should_retry = (
                job.current_attempt < job.retry_policy.max_attempts and
                failure_reason not in [JobFailureReason.VALIDATION_ERROR, JobFailureReason.DEPENDENCY_ERROR]
            )
            
            if should_retry:
                # Schedule retry
                job.status = JobStatus.PENDING  # Back to pending for retry
                job.metrics.retry_count += 1
                
                # Calculate retry delay
                delay = min(
                    job.retry_policy.initial_delay * (job.retry_policy.backoff_multiplier ** (job.current_attempt - 1)),
                    job.retry_policy.max_delay
                )
                
                # Add jitter if configured
                if job.retry_policy.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                await self.storage.save_job(job)
                
                self.logger.info(f"Job {job.job_id} will retry in {delay:.1f}s (attempt {job.current_attempt}/{job.retry_policy.max_attempts})")
                
                # Schedule retry
                asyncio.create_task(self._schedule_retry(job.job_id, delay))
                
            else:
                # Final failure
                job.status = JobStatus.FAILED
                job.metrics.failed_at = datetime.now(timezone.utc)
                
                if job.metrics.failed_at and job.metrics.created_at:
                    job.metrics.total_time = (
                        job.metrics.failed_at - job.metrics.created_at
                    ).total_seconds()
                
                await self.storage.save_job(job)
                
                # Record metrics
                await self.metrics.record_job_failed(job)
                
                # Send Kafka notification
                asyncio.create_task(self.kafka.send_job_failed(
                    job_id=job.job_id,
                    error_message=error_message,
                    error_type=failure_reason.value,
                    user_id=job.user_id
                ))
                
                self.logger.error(f"Job {job.job_id} failed permanently after {job.current_attempt} attempts: {error_message}")
                
                # Handle dependent jobs
                await self._handle_job_failure_dependencies(job.job_id, error_message)
            
        except Exception as e:
            self.logger.error(f"Failed to handle job failure for {job.job_id}: {str(e)}")
    
    async def _schedule_retry(self, job_id: str, delay: float):
        """Schedule a job retry after delay."""
        try:
            await asyncio.sleep(delay)
            if not self._shutdown:
                await self._try_start_job(job_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Failed to schedule retry for job {job_id}: {str(e)}")
    
    async def _handle_job_timeout(self, job_id: str, timeout_seconds: float):
        """Handle job timeout."""
        try:
            await asyncio.sleep(timeout_seconds)
            
            # Check if job is still running
            if job_id in self.active_jobs:
                job = await self.storage.load_job(job_id)
                if job and job.status == JobStatus.RUNNING:
                    await self._cancel_job_execution(job_id, f"Job timed out after {timeout_seconds}s")
                    await self._handle_job_failure(job, JobFailureReason.TIMEOUT_ERROR, 
                                                 f"Job timed out after {timeout_seconds}s")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Failed to handle timeout for job {job_id}: {str(e)}")
    
    async def _cancel_job_execution(self, job_id: str, reason: str):
        """Cancel active job execution."""
        try:
            async with self._state_lock:
                task = self.active_jobs.pop(job_id, None)
            
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Cancel timeout task
            timeout_task = self.job_timeouts.pop(job_id, None)
            if timeout_task:
                timeout_task.cancel()
            
            self.logger.info(f"Job {job_id} execution cancelled: {reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job execution {job_id}: {str(e)}")
    
    async def _complete_job_dependencies(self, completed_job_id: str):
        """Try to start jobs that depend on the completed job."""
        try:
            async with self._state_lock:
                dependent_job_ids = self.dependency_graph.pop(completed_job_id, set())
            
            for dep_job_id in dependent_job_ids:
                await self._try_start_job(dep_job_id)
                
        except Exception as e:
            self.logger.error(f"Failed to handle dependencies for completed job {completed_job_id}: {str(e)}")
    
    async def _handle_job_failure_dependencies(self, failed_job_id: str, error_message: str):
        """Handle dependent jobs when a job fails."""
        try:
            async with self._state_lock:
                dependent_job_ids = self.dependency_graph.pop(failed_job_id, set())
            
            # Cancel or fail dependent jobs
            for dep_job_id in dependent_job_ids:
                dep_job = await self.storage.load_job(dep_job_id)
                if dep_job and dep_job.status == JobStatus.PENDING:
                    await self._handle_job_failure(
                        dep_job, 
                        JobFailureReason.DEPENDENCY_ERROR,
                        f"Dependency job {failed_job_id} failed: {error_message}"
                    )
                
        except Exception as e:
            self.logger.error(f"Failed to handle dependencies for failed job {failed_job_id}: {str(e)}")
    
    async def _recover_jobs(self):
        """Recover jobs from storage on service startup."""
        try:
            # Get all running jobs
            running_jobs = await self.storage.get_jobs_by_status(JobStatus.RUNNING)
            
            for job in running_jobs:
                # Reset to pending to restart
                job.status = JobStatus.PENDING
                await self.storage.save_job(job)
                
                # Try to start if dependencies are satisfied
                await self._try_start_job(job.job_id)
            
            self.logger.info(f"Recovered {len(running_jobs)} jobs from storage")
            
        except Exception as e:
            self.logger.error(f"Failed to recover jobs: {str(e)}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if self._shutdown:
                    break
                
                # Cleanup completed jobs older than retention period
                # TODO: Implement job retention cleanup
                
                # Health check
                await self.health_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {str(e)}")
                await asyncio.sleep(60)  # Back off on error


# Global service instance and management
_job_orchestration_service: Optional[JobOrchestrationService] = None


def get_job_orchestration_service() -> JobOrchestrationService:
    """Get or create the global job orchestration service."""
    global _job_orchestration_service
    if _job_orchestration_service is None:
        _job_orchestration_service = JobOrchestrationService()
    return _job_orchestration_service


@asynccontextmanager
async def job_orchestration_context():
    """Async context manager for job orchestration service."""
    service = get_job_orchestration_service()
    try:
        await service.start()
        yield service
    finally:
        await service.stop()


# Kafka event handler registration
def get_kafka_message_handlers() -> Dict[MessageType, Callable]:
    """Get Kafka message handlers for job orchestration."""
    service = get_job_orchestration_service()
    
    return {
        MessageType.JOB_SUBMITTED: service.handle_job_submitted,
        MessageType.JOB_STARTED: service.handle_job_started,
        MessageType.JOB_COMPLETED: service.handle_job_completed,
        MessageType.JOB_FAILED: service.handle_job_failed,
    }


# FastAPI dependencies
def get_job_service() -> JobOrchestrationService:
    """FastAPI dependency for job orchestration service."""
    return get_job_orchestration_service()
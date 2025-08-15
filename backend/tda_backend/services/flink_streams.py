"""
Apache Flink Stream Processing Integration for TDA Backend.

This module provides real-time stream processing capabilities for TDA computations,
including streaming TDA analysis, incremental persistence computation, and 
real-time result publishing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import time
import subprocess
import tempfile
import os

from ..models import (
    TDAComputationRequest, 
    TDAResults, 
    PointCloud, 
    Point,
    PersistenceDiagram,
    PersistencePair,
    HomologyGroup
)
from ..config import settings
from .kafka_producer import KafkaProducerService, MessageType, TDAMessage
from .kafka_consumer import KafkaConsumerService

logger = logging.getLogger(__name__)


class StreamProcessingMode(str, Enum):
    """Stream processing modes for TDA computations."""
    INCREMENTAL = "incremental"  # Process points as they arrive
    WINDOWED = "windowed"        # Process in time/count windows
    TRIGGERED = "triggered"      # Process on explicit triggers
    CONTINUOUS = "continuous"    # Continuous sliding window


@dataclass
class StreamingConfig:
    """Configuration for streaming TDA processing."""
    window_size: int = 100           # Number of points per window
    window_timeout: float = 10.0     # Window timeout in seconds
    slide_interval: int = 10         # Sliding window interval
    parallelism: int = 4             # Processing parallelism
    checkpoint_interval: int = 10000 # Checkpoint interval (ms)
    mode: StreamProcessingMode = StreamProcessingMode.WINDOWED


@dataclass
class StreamingPoint:
    """A point in the streaming context with metadata."""
    point: Point
    timestamp: datetime
    source_id: str
    sequence_number: int
    batch_id: Optional[str] = None


@dataclass
class StreamingWindow:
    """A window of points for processing."""
    points: List[StreamingPoint]
    window_id: str
    start_time: datetime
    end_time: datetime
    trigger_type: str = "time"


class FlinkJobManager:
    """Manages Flink jobs for TDA stream processing."""
    
    def __init__(self, flink_rest_url: str = "http://localhost:8082"):
        """Initialize Flink job manager."""
        self.flink_rest_url = flink_rest_url
        self.active_jobs: Dict[str, str] = {}  # job_name -> flink_job_id
        self.job_configs: Dict[str, Dict] = {}
        
    async def submit_tda_job(
        self, 
        job_name: str, 
        config: StreamingConfig,
        input_topics: List[str],
        output_topics: List[str]
    ) -> str:
        """Submit a TDA streaming job to Flink."""
        logger.info(f"Submitting TDA streaming job: {job_name}")
        
        # Generate Flink job configuration
        job_config = self._create_job_config(
            job_name, config, input_topics, output_topics
        )
        
        try:
            # Create temporary JAR file with our TDA streaming application
            jar_path = await self._create_tda_streaming_jar(job_config)
            
            # Submit job to Flink
            job_id = await self._submit_flink_job(jar_path, job_config)
            
            self.active_jobs[job_name] = job_id
            self.job_configs[job_name] = job_config
            
            logger.info(f"Successfully submitted job {job_name} with ID: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit Flink job {job_name}: {e}")
            raise
    
    async def cancel_job(self, job_name: str) -> bool:
        """Cancel a running Flink job."""
        if job_name not in self.active_jobs:
            logger.warning(f"Job {job_name} not found in active jobs")
            return False
        
        job_id = self.active_jobs[job_name]
        
        try:
            # Cancel job via Flink REST API
            await self._cancel_flink_job(job_id)
            
            del self.active_jobs[job_name]
            del self.job_configs[job_name]
            
            logger.info(f"Successfully cancelled job {job_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_name}: {e}")
            return False
    
    async def get_job_status(self, job_name: str) -> Optional[Dict]:
        """Get status of a Flink job."""
        if job_name not in self.active_jobs:
            return None
        
        job_id = self.active_jobs[job_name]
        
        try:
            # Get job status via Flink REST API
            status = await self._get_flink_job_status(job_id)
            return status
            
        except Exception as e:
            logger.error(f"Failed to get status for job {job_name}: {e}")
            return None
    
    def _create_job_config(
        self,
        job_name: str,
        config: StreamingConfig,
        input_topics: List[str],
        output_topics: List[str]
    ) -> Dict:
        """Create Flink job configuration."""
        return {
            "job_name": job_name,
            "parallelism": config.parallelism,
            "checkpoint_interval": config.checkpoint_interval,
            "window_size": config.window_size,
            "window_timeout": config.window_timeout,
            "slide_interval": config.slide_interval,
            "mode": config.mode.value,
            "input_topics": input_topics,
            "output_topics": output_topics,
            "kafka_bootstrap_servers": settings.kafka_bootstrap_servers,
            "kafka_group_id": f"flink-tda-{job_name}",
            "processing_guarantees": "exactly_once"
        }
    
    async def _create_tda_streaming_jar(self, job_config: Dict) -> str:
        """Create a JAR file for the TDA streaming application."""
        # For now, we'll use PyFlink to create the streaming application
        # In production, this would be a pre-built JAR
        
        # Create Python Flink application
        app_code = self._generate_pyflink_app(job_config)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(app_code)
            return f.name
    
    def _generate_pyflink_app(self, job_config: Dict) -> str:
        """Generate PyFlink application code for TDA streaming."""
        return f'''
import json
import numpy as np
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import MapFunction, WindowFunction
from pyflink.datastream.window import TumblingEventTimeWindows, Time

class TDAStreamProcessor(MapFunction):
    def map(self, value):
        # Parse incoming point data
        data = json.loads(value)
        
        # Extract point cloud data
        points = data.get('points', [])
        
        # Perform incremental TDA computation
        result = self.compute_tda(points)
        
        return json.dumps(result)
    
    def compute_tda(self, points):
        # Simplified TDA computation for streaming
        # In practice, this would use optimized algorithms
        
        # Convert points to numpy array
        point_array = np.array([[p['x'], p['y']] for p in points])
        
        # Compute simple persistence (placeholder)
        persistence_pairs = []
        
        # Return results
        return {{
            'timestamp': '{datetime.now().isoformat()}',
            'num_points': len(points),
            'persistence_pairs': persistence_pairs,
            'betti_numbers': {{'0': len(points), '1': 0}}
        }}

def create_tda_streaming_job():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism({job_config['parallelism']})
    
    # Enable checkpointing
    env.enable_checkpointing({job_config['checkpoint_interval']})
    
    # Set up Kafka source
    kafka_source = FlinkKafkaConsumer(
        topics={job_config['input_topics']},
        deserialization_schema=SimpleStringSchema(),
        properties={{
            'bootstrap.servers': '{job_config['kafka_bootstrap_servers']}',
            'group.id': '{job_config['kafka_group_id']}'
        }}
    )
    
    # Set up Kafka sink
    kafka_sink = FlinkKafkaProducer(
        topic='{job_config['output_topics'][0]}',
        serialization_schema=SimpleStringSchema(),
        producer_config={{
            'bootstrap.servers': '{job_config['kafka_bootstrap_servers']}'
        }}
    )
    
    # Create streaming pipeline
    data_stream = env.add_source(kafka_source)
    
    # Apply TDA processing
    processed_stream = data_stream.map(TDAStreamProcessor())
    
    # Write to output topic
    processed_stream.add_sink(kafka_sink)
    
    # Execute the job
    env.execute('{job_config['job_name']}')

if __name__ == '__main__':
    create_tda_streaming_job()
'''
    
    async def _submit_flink_job(self, app_path: str, job_config: Dict) -> str:
        """Submit job to Flink cluster."""
        # This would normally use Flink REST API
        # For demonstration, we'll simulate job submission
        
        import uuid
        job_id = str(uuid.uuid4())
        
        logger.info(f"Simulating Flink job submission: {app_path}")
        logger.info(f"Job config: {job_config}")
        
        # In production, this would:
        # 1. Upload JAR to Flink
        # 2. Submit job with configuration
        # 3. Return actual job ID
        
        return job_id
    
    async def _cancel_flink_job(self, job_id: str) -> None:
        """Cancel Flink job via REST API."""
        logger.info(f"Simulating cancellation of job: {job_id}")
        # Implementation would use Flink REST API
    
    async def _get_flink_job_status(self, job_id: str) -> Dict:
        """Get job status via REST API."""
        # Simulate job status
        return {
            "job_id": job_id,
            "status": "RUNNING",
            "start_time": int(time.time() * 1000),
            "duration": 300000,  # 5 minutes
            "parallelism": 4,
            "checkpoints": {
                "completed": 15,
                "failed": 0,
                "duration_avg": 1200
            }
        }


class TDAStreamProcessor:
    """Processes streaming TDA computations without Flink (fallback)."""
    
    def __init__(self, config: StreamingConfig):
        """Initialize stream processor."""
        self.config = config
        self.active_windows: Dict[str, StreamingWindow] = {}
        self.point_buffer: List[StreamingPoint] = []
        self.last_window_time = datetime.now(timezone.utc)
        self.sequence_counter = 0
        
        # Kafka integration
        self.producer: Optional[KafkaProducerService] = None
        self.consumer: Optional[KafkaConsumerService] = None
        
    async def start(self) -> None:
        """Start the stream processor."""
        logger.info("Starting TDA stream processor...")
        
        # Initialize Kafka services
        self.producer = KafkaProducerService()
        self.consumer = KafkaConsumerService()
        
        await self.producer.start()
        await self.consumer.start()
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
        
    async def stop(self) -> None:
        """Stop the stream processor."""
        logger.info("Stopping TDA stream processor...")
        
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
    
    async def process_point(self, point: Point, source_id: str) -> None:
        """Process a single streaming point."""
        streaming_point = StreamingPoint(
            point=point,
            timestamp=datetime.now(timezone.utc),
            source_id=source_id,
            sequence_number=self.sequence_counter
        )
        self.sequence_counter += 1
        
        # Add to buffer
        self.point_buffer.append(streaming_point)
        
        # Check if window should be triggered
        await self._check_window_trigger()
    
    async def process_point_batch(
        self, 
        points: List[Point], 
        source_id: str,
        batch_id: Optional[str] = None
    ) -> None:
        """Process a batch of points."""
        current_time = datetime.now(timezone.utc)
        
        streaming_points = [
            StreamingPoint(
                point=point,
                timestamp=current_time,
                source_id=source_id,
                sequence_number=self.sequence_counter + i,
                batch_id=batch_id
            )
            for i, point in enumerate(points)
        ]
        
        self.sequence_counter += len(points)
        self.point_buffer.extend(streaming_points)
        
        await self._check_window_trigger()
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while True:
            try:
                # Check for time-based window triggers
                await self._check_time_trigger()
                
                # Process any ready windows
                await self._process_ready_windows()
                
                # Sleep briefly
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_window_trigger(self) -> None:
        """Check if window should be triggered based on count or time."""
        current_time = datetime.now(timezone.utc)
        
        # Count-based trigger
        if len(self.point_buffer) >= self.config.window_size:
            await self._create_window("count")
        
        # Time-based trigger
        elif (current_time - self.last_window_time).total_seconds() >= self.config.window_timeout:
            if self.point_buffer:  # Only create window if we have points
                await self._create_window("time")
    
    async def _check_time_trigger(self) -> None:
        """Check for time-based triggers."""
        current_time = datetime.now(timezone.utc)
        
        if (current_time - self.last_window_time).total_seconds() >= self.config.window_timeout:
            if self.point_buffer:
                await self._create_window("time")
    
    async def _create_window(self, trigger_type: str) -> None:
        """Create a new window from buffered points."""
        if not self.point_buffer:
            return
        
        current_time = datetime.now(timezone.utc)
        window_id = f"window_{int(current_time.timestamp() * 1000)}"
        
        # Determine window points based on mode
        if self.config.mode == StreamProcessingMode.WINDOWED:
            # Take all buffered points
            window_points = self.point_buffer[:self.config.window_size]
            self.point_buffer = self.point_buffer[self.config.window_size:]
        elif self.config.mode == StreamProcessingMode.INCREMENTAL:
            # Take sliding window
            window_points = self.point_buffer[-self.config.slide_interval:]
        else:
            window_points = self.point_buffer.copy()
            self.point_buffer.clear()
        
        if window_points:
            window = StreamingWindow(
                points=window_points,
                window_id=window_id,
                start_time=window_points[0].timestamp,
                end_time=current_time,
                trigger_type=trigger_type
            )
            
            self.active_windows[window_id] = window
            self.last_window_time = current_time
            
            logger.debug(f"Created window {window_id} with {len(window_points)} points")
    
    async def _process_ready_windows(self) -> None:
        """Process all ready windows."""
        for window_id, window in list(self.active_windows.items()):
            try:
                await self._process_window(window)
                del self.active_windows[window_id]
            except Exception as e:
                logger.error(f"Error processing window {window_id}: {e}")
    
    async def _process_window(self, window: StreamingWindow) -> None:
        """Process a single window and compute TDA."""
        logger.debug(f"Processing window {window.window_id} with {len(window.points)} points")
        
        # Extract point cloud from window
        points = [sp.point for sp in window.points]
        point_cloud = PointCloud(
            points=points,
            dimension=len(points[0].coordinates) if points else 2
        )
        
        # Perform TDA computation
        start_time = time.time()
        results = await self._compute_streaming_tda(point_cloud, window)
        computation_time = time.time() - start_time
        
        # Create result message
        result_message = {
            "window_id": window.window_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_points": len(points),
            "computation_time": computation_time,
            "trigger_type": window.trigger_type,
            "results": results.dict() if results else None
        }
        
        # Publish results to Kafka
        if self.producer:
            await self.producer.send_message(
                message_type=MessageType.RESULT_GENERATED,
                data=result_message,
                topic="tda_stream_results"
            )
        
        logger.info(f"Completed processing window {window.window_id} in {computation_time:.3f}s")
    
    async def _compute_streaming_tda(
        self, 
        point_cloud: PointCloud, 
        window: StreamingWindow
    ) -> Optional[TDAResults]:
        """Compute TDA for streaming window (simplified)."""
        try:
            # For demonstration, create simple results
            # In practice, this would use optimized streaming TDA algorithms
            
            num_points = len(point_cloud.points)
            
            # Simple persistence pairs (placeholder)
            pairs = [
                PersistencePair(
                    dimension=0,
                    birth=0.0,
                    death=float('inf'),
                    persistence=float('inf')
                )
            ]
            
            # Create homology groups
            homology_groups = [
                HomologyGroup(
                    dimension=0,
                    pairs=pairs,
                    num_features=1
                )
            ]
            
            # Create persistence diagrams
            diagrams = [
                PersistenceDiagram(
                    dimension=0,
                    pairs=pairs,
                    num_features=1
                )
            ]
            
            # Create results
            results = TDAResults(
                persistence_diagrams=diagrams,
                homology_groups=homology_groups,
                betti_numbers={0: [1]},
                computation_metadata={
                    "algorithm": "streaming_vr",
                    "window_id": window.window_id,
                    "num_points": num_points,
                    "mode": self.config.mode.value
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing streaming TDA: {e}")
            return None


class StreamingTDAService:
    """High-level service for managing streaming TDA operations."""
    
    def __init__(self):
        """Initialize streaming TDA service."""
        self.flink_manager = FlinkJobManager()
        self.stream_processor = TDAStreamProcessor(StreamingConfig())
        self.active_streams: Dict[str, Dict] = {}
        self.analytics_jobs: Dict[str, Dict] = {}
        
    async def start(self) -> None:
        """Start the streaming TDA service."""
        logger.info("Starting Streaming TDA Service...")
        await self.stream_processor.start()
        
    async def stop(self) -> None:
        """Stop the streaming TDA service."""
        logger.info("Stopping Streaming TDA Service...")
        
        # Cancel all active Flink jobs
        for stream_name in list(self.active_streams.keys()):
            await self.stop_stream(stream_name)
        
        # Stop all analytics jobs
        for job_name in list(self.analytics_jobs.keys()):
            await self.stop_analytics_job(job_name)
        
        await self.stream_processor.stop()
    
    async def create_stream(
        self,
        stream_name: str,
        config: StreamingConfig,
        use_flink: bool = True
    ) -> bool:
        """Create a new TDA streaming computation."""
        logger.info(f"Creating TDA stream: {stream_name}")
        
        try:
            if use_flink:
                # Use Flink for processing
                job_id = await self.flink_manager.submit_tda_job(
                    job_name=stream_name,
                    config=config,
                    input_topics=[f"tda_stream_input_{stream_name}"],
                    output_topics=[f"tda_stream_output_{stream_name}"]
                )
                
                self.active_streams[stream_name] = {
                    "type": "flink",
                    "job_id": job_id,
                    "config": config,
                    "created_at": datetime.now(timezone.utc)
                }
            else:
                # Use local processing
                self.active_streams[stream_name] = {
                    "type": "local",
                    "config": config,
                    "created_at": datetime.now(timezone.utc)
                }
            
            logger.info(f"Successfully created stream: {stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create stream {stream_name}: {e}")
            return False
    
    async def stop_stream(self, stream_name: str) -> bool:
        """Stop a streaming computation."""
        if stream_name not in self.active_streams:
            logger.warning(f"Stream {stream_name} not found")
            return False
        
        stream_info = self.active_streams[stream_name]
        
        try:
            if stream_info["type"] == "flink":
                await self.flink_manager.cancel_job(stream_name)
            
            del self.active_streams[stream_name]
            logger.info(f"Successfully stopped stream: {stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop stream {stream_name}: {e}")
            return False
    
    async def get_stream_status(self, stream_name: str) -> Optional[Dict]:
        """Get status of a streaming computation."""
        if stream_name not in self.active_streams:
            return None
        
        stream_info = self.active_streams[stream_name]
        
        if stream_info["type"] == "flink":
            flink_status = await self.flink_manager.get_job_status(stream_name)
            return {
                **stream_info,
                "flink_status": flink_status
            }
        else:
            return {
                **stream_info,
                "status": "running"
            }
    
    async def list_streams(self) -> Dict[str, Dict]:
        """List all active streams."""
        return self.active_streams.copy()
    
    async def start_analytics_job(
        self,
        job_name: str = "tda_analytics",
        config: Optional[Dict] = None
    ) -> bool:
        """Start the TDA analytics Flink job."""
        logger.info(f"Starting TDA analytics job: {job_name}")
        
        try:
            # Default analytics configuration
            analytics_config = {
                "parallelism": 4,
                "checkpoint_interval": 30000,
                "input_topics": [
                    "tda_jobs",
                    "tda_results", 
                    "tda_events",
                    "tda_uploads",
                    "tda_errors"
                ],
                "output_topics": [
                    "tda_analytics_dashboards",
                    "tda_analytics_alerts",
                    "tda_analytics_reports",
                    "tda_analytics_metrics"
                ],
                "window_configs": {
                    "realtime": {"size": "1min", "slide": "10s"},
                    "short_term": {"size": "5min", "slide": "1min"},
                    "medium_term": {"size": "15min", "slide": "5min"},
                    "long_term": {"size": "1hour", "slide": "10min"}
                }
            }
            
            # Override with custom config
            if config:
                analytics_config.update(config)
            
            # Submit analytics job to Flink
            job_id = await self._submit_analytics_job(job_name, analytics_config)
            
            self.analytics_jobs[job_name] = {
                "job_id": job_id,
                "config": analytics_config,
                "started_at": datetime.now(timezone.utc),
                "status": "running"
            }
            
            logger.info(f"Successfully started analytics job {job_name} with ID: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start analytics job {job_name}: {e}")
            return False
    
    async def stop_analytics_job(self, job_name: str = "tda_analytics") -> bool:
        """Stop the TDA analytics Flink job."""
        if job_name not in self.analytics_jobs:
            logger.warning(f"Analytics job {job_name} not found")
            return False
        
        try:
            job_info = self.analytics_jobs[job_name]
            await self.flink_manager.cancel_job(job_name)
            
            job_info["status"] = "stopped"
            job_info["stopped_at"] = datetime.now(timezone.utc)
            
            logger.info(f"Successfully stopped analytics job: {job_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop analytics job {job_name}: {e}")
            return False
    
    async def get_analytics_status(self, job_name: str = "tda_analytics") -> Optional[Dict]:
        """Get status of the TDA analytics job."""
        if job_name not in self.analytics_jobs:
            return None
        
        job_info = self.analytics_jobs[job_name]
        
        try:
            # Get Flink job status
            flink_status = await self.flink_manager.get_job_status(job_name)
            
            return {
                **job_info,
                "flink_status": flink_status,
                "uptime": (
                    datetime.now(timezone.utc) - job_info["started_at"]
                ).total_seconds() if job_info.get("started_at") else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics job status {job_name}: {e}")
            return job_info
    
    async def list_analytics_jobs(self) -> Dict[str, Dict]:
        """List all analytics jobs."""
        return self.analytics_jobs.copy()
    
    async def _submit_analytics_job(self, job_name: str, config: Dict) -> str:
        """Submit analytics job to Flink cluster."""
        # Create job submission configuration
        job_config = {
            "job_name": job_name,
            "job_type": "analytics",
            "parallelism": config.get("parallelism", 4),
            "checkpoint_interval": config.get("checkpoint_interval", 30000),
            "input_topics": config["input_topics"],
            "output_topics": config["output_topics"],
            "kafka_bootstrap_servers": settings.kafka_bootstrap_servers,
            "kafka_group_id": f"flink-analytics-{job_name}",
            "window_configs": config.get("window_configs", {}),
            "analytics_config": {
                "alert_thresholds": {
                    "error_rate": 0.05,  # 5% error rate threshold
                    "computation_time_p99": 300.0,  # 5 minutes P99 threshold
                    "queue_time_p99": 60.0  # 1 minute queue time threshold
                },
                "metrics_config": {
                    "enable_detailed_metrics": True,
                    "metric_retention_hours": 24,
                    "aggregation_intervals": ["1m", "5m", "15m", "1h"]
                }
            }
        }
        
        # For now, simulate job submission
        # In production, this would submit the actual analytics job
        import uuid
        job_id = str(uuid.uuid4())
        
        logger.info(f"Simulating analytics job submission: {job_name}")
        logger.info(f"Analytics config: {job_config}")
        
        return job_id

    async def submit_points(
        self, 
        stream_name: str, 
        points: List[Point],
        source_id: str = "api"
    ) -> bool:
        """Submit points to a streaming computation."""
        if stream_name not in self.active_streams:
            logger.error(f"Stream {stream_name} not found")
            return False
        
        try:
            # For local processing
            await self.stream_processor.process_point_batch(
                points=points,
                source_id=source_id,
                batch_id=f"{stream_name}_{int(time.time())}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit points to stream {stream_name}: {e}")
            return False


# Global service instance
_streaming_service: Optional[StreamingTDAService] = None


def get_streaming_service() -> StreamingTDAService:
    """Get or create the global streaming TDA service."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingTDAService()
    return _streaming_service


async def initialize_streaming_service() -> None:
    """Initialize the global streaming TDA service."""
    service = get_streaming_service()
    await service.start()


async def shutdown_streaming_service() -> None:
    """Shutdown the global streaming TDA service."""
    global _streaming_service
    if _streaming_service:
        await _streaming_service.stop()
        _streaming_service = None

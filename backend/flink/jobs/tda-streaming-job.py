#!/usr/bin/env python3
"""
TDA Streaming Job for Apache Flink
==================================

Real-time topological data analysis using PyFlink for stream processing.
Processes point cloud data from Kafka topics and computes persistence diagrams
in real-time with configurable windowing strategies.

Usage:
    python tda-streaming-job.py [--config config.json] [--job-name tda-stream]
    
Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka cluster endpoints
    SCHEMA_REGISTRY_URL: Schema registry URL
    FLINK_PARALLELISM: Job parallelism (default: 4)
    TDA_WINDOW_SIZE: Processing window size (default: 100)
    TDA_SLIDE_INTERVAL: Window slide interval (default: 10)
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import numpy as np
import tempfile
import subprocess
import shlex

# PyFlink imports
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.formats.json import JsonRowSerializationSchema, JsonRowDeserializationSchema
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import (
    MapFunction, FlatMapFunction, FilterFunction, 
    KeyedProcessFunction, WindowFunction, AggregateFunction
)
from pyflink.datastream.window import (
    TumblingEventTimeWindows, SlidingEventTimeWindows, 
    SessionWindows, TimeWindow, Time
)
from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor
from pyflink.common.types import Row
from pyflink.common.watermark_strategy import WatermarkStrategy

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TDAJobConfig:
    """Configuration for TDA streaming job."""
    job_name: str = "tda-streaming-job"
    parallelism: int = 4
    checkpoint_interval: int = 30000  # 30 seconds
    window_size: int = 100           # points
    slide_interval: int = 10         # points  
    window_timeout: int = 60         # seconds
    max_dimension: int = 2           # homology dimension
    max_persistence: float = 1.0     # filtration threshold
    
    # Kafka configuration
    kafka_bootstrap_servers: str = "kafka1:9092,kafka2:9093,kafka3:9094"
    schema_registry_url: str = "http://schema-registry:8081"
    input_topic: str = "tda_jobs"
    output_topic: str = "tda_results"
    consumer_group: str = "flink-tda-consumer"
    
    # Performance tuning
    buffer_timeout: int = 100        # ms
    network_buffer_size: int = 32768  # 32KB
    enable_object_reuse: bool = True
    
    @classmethod
    def from_env(cls) -> 'TDAJobConfig':
        """Create configuration from environment variables."""
        return cls(
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", cls.kafka_bootstrap_servers),
            schema_registry_url=os.getenv("SCHEMA_REGISTRY_URL", cls.schema_registry_url),
            parallelism=int(os.getenv("FLINK_PARALLELISM", cls.parallelism)),
            window_size=int(os.getenv("TDA_WINDOW_SIZE", cls.window_size)),
            slide_interval=int(os.getenv("TDA_SLIDE_INTERVAL", cls.slide_interval))
        )


class PointCloudParser(MapFunction):
    """Parse incoming JSON messages into point cloud data."""
    
    def map(self, value: str) -> Row:
        """Parse JSON message to extract point cloud data."""
        try:
            data = json.loads(value)
            
            # Extract job information
            job_id = data.get('job_id', 'unknown')
            timestamp = data.get('timestamp', datetime.now(timezone.utc).isoformat())
            algorithm = data.get('algorithm', 'vietoris_rips')
            
            # Extract point cloud
            point_cloud = data.get('point_cloud', {})
            points = point_cloud.get('points', [])
            dimension = point_cloud.get('dimension', 2)
            
            # Convert points to numpy-compatible format
            point_array = []
            for point in points:
                if isinstance(point, dict):
                    coords = point.get('coordinates', [])
                elif isinstance(point, list):
                    coords = point
                else:
                    coords = []
                
                # Ensure consistent dimension
                if len(coords) < dimension:
                    coords.extend([0.0] * (dimension - len(coords)))
                elif len(coords) > dimension:
                    coords = coords[:dimension]
                
                point_array.append(coords)
            
            return Row(
                job_id=job_id,
                timestamp=timestamp,
                algorithm=algorithm,
                dimension=dimension,
                points=point_array,
                num_points=len(point_array),
                event_time=int(datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp() * 1000)
            )
            
        except Exception as e:
            logger.error(f"Error parsing point cloud data: {e}")
            # Return empty row for failed parsing
            return Row(
                job_id='error',
                timestamp=datetime.now(timezone.utc).isoformat(),
                algorithm='unknown',
                dimension=2,
                points=[],
                num_points=0,
                event_time=int(datetime.now().timestamp() * 1000)
            )


class TDAComputation(WindowFunction):
    """Compute TDA features for windowed point cloud data."""
    
    def __init__(self, config: TDAJobConfig):
        self.config = config
        
    def apply(self, key: str, window: TimeWindow, inputs: List[Row]) -> List[Row]:
        """Apply TDA computation to windowed data."""
        try:
            if not inputs:
                return []
            
            # Collect all points from the window
            all_points = []
            job_ids = set()
            algorithms = set()
            
            for row in inputs:
                all_points.extend(row.points)
                job_ids.add(row.job_id)
                algorithms.add(row.algorithm)
            
            if not all_points:
                return []
            
            # Convert to numpy array for computation
            point_matrix = np.array(all_points)
            num_points = len(all_points)
            dimension = point_matrix.shape[1] if point_matrix.size > 0 else 2
            
            # Compute TDA features
            computation_result = self._compute_tda_features(point_matrix)
            
            # Create result row
            result = Row(
                job_id=list(job_ids)[0] if len(job_ids) == 1 else f"batch_{len(job_ids)}",
                window_start=window.start,
                window_end=window.end,
                timestamp=datetime.now(timezone.utc).isoformat(),
                algorithm=list(algorithms)[0] if len(algorithms) == 1 else "mixed",
                num_points=num_points,
                dimension=dimension,
                betti_numbers=computation_result['betti_numbers'],
                persistence_pairs=computation_result['persistence_pairs'],
                homology_groups=computation_result['homology_groups'],
                computation_time=computation_result['computation_time'],
                memory_usage=computation_result.get('memory_usage', 0),
                success=computation_result['success']
            )
            
            return [result]
            
        except Exception as e:
            logger.error(f"Error in TDA computation: {e}")
            return [Row(
                job_id=key,
                window_start=window.start,
                window_end=window.end,
                timestamp=datetime.now(timezone.utc).isoformat(),
                algorithm="error",
                num_points=0,
                dimension=0,
                betti_numbers={},
                persistence_pairs=[],
                homology_groups=[],
                computation_time=0.0,
                memory_usage=0,
                success=False,
                error=str(e)
            )]
    
    def _compute_tda_features(self, points: np.ndarray) -> Dict[str, Any]:
        """Compute TDA features from point cloud."""
        import time
        start_time = time.time()
        
        try:
            if points.size == 0:
                return {
                    'betti_numbers': {},
                    'persistence_pairs': [],
                    'homology_groups': [],
                    'computation_time': 0.0,
                    'success': False
                }
            # Preferred path: invoke optimized C++ harness if configured
            use_harness = os.getenv("TDA_USE_HARNESS", "1") == "1"
            harness_path = os.getenv("TDA_HARNESS_PATH", "build/release/tests/cpp/test_streaming_cech_perf")
            mode = os.getenv("TDA_MODE", "vr")  # vr or cech
            epsilon = float(os.getenv("TDA_EPSILON", "0.5"))
            radius = float(os.getenv("TDA_RADIUS", "0.5"))
            soft_knn_cap = int(os.getenv("TDA_SOFT_K", "16"))
            parallel_threshold = int(os.getenv("TDA_PAR_THRESH", "0"))
            max_dim = int(os.getenv("TDA_MAX_DIM", str(self.config.max_dimension)))
            time_limit = int(os.getenv("TDA_TIME_LIMIT", "0"))
            use_gpu = os.getenv("TDA_USE_CUDA", "0") == "1"

            if use_harness:
                # Ensure harness binary path exists
                if not os.path.exists(harness_path) or not os.access(harness_path, os.X_OK):
                    logger.warning(f"Harness not found or not executable at {harness_path}; falling back to Python path.")
                    raise FileNotFoundError(harness_path)

                with tempfile.TemporaryDirectory() as tmpd:
                    csv_path = os.path.join(tmpd, "points.csv")
                    jsonl_path = os.path.join(tmpd, "out.jsonl")
                    # Write CSV: one point per line, comma-separated floats
                    np.savetxt(csv_path, points, delimiter=",", fmt="%.10f")
                    cmd = [
                        harness_path,
                        "--mode", mode,
                        "--points-csv", csv_path,
                        "--epsilon", str(epsilon),
                        "--radius", str(radius),
                        "--maxDim", str(max_dim),
                        "--soft-knn-cap", str(soft_knn_cap),
                        "--parallel-threshold", str(parallel_threshold),
                        "--json", jsonl_path,
                    ]
                    if time_limit > 0:
                        cmd.extend(["--time-limit", str(time_limit)])
                    logger.info(f"Invoking harness: {' '.join(shlex.quote(c) for c in cmd)}")
                    try:
                        env = os.environ.copy()
                        if use_gpu:
                            env["TDA_USE_CUDA"] = "1"
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=max(5, time_limit or 30), env=env)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Harness failed: rc={e.returncode}, stderr={e.stderr[:500]}...")
                        raise
                    except subprocess.TimeoutExpired as e:
                        logger.error("Harness timed out")
                        raise
                    # Read last JSONL line for metrics
                    last = {}
                    try:
                        with open(jsonl_path, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        last = json.loads(line)
                                    except Exception:
                                        pass
                    except FileNotFoundError:
                        logger.error("Harness JSONL output not found")
                        last = {}
                    computation_time = time.time() - start_time
                    # Map harness telemetry to streaming result shape
                    betti_numbers = {"0": [int(last.get("betti0", 0))] } if isinstance(last.get("betti0", 0), (int,float)) else {}
                    homology_groups = []
                    persistence_pairs = []
                    return {
                        'betti_numbers': betti_numbers,
                        'persistence_pairs': persistence_pairs,
                        'homology_groups': homology_groups,
                        'computation_time': computation_time,
                        'memory_usage': int( (last.get('rss_peakMB') or 0) * 1024 * 1024 ),
                        'success': True
                    }

            # Fallback placeholder path: local Python approximation
            distances = self._compute_distance_matrix(points)
            persistence_pairs = self._compute_simple_persistence(distances)
            betti_numbers = self._compute_betti_numbers(persistence_pairs)
            homology_groups = self._create_homology_groups(persistence_pairs)
            computation_time = time.time() - start_time
            return {
                'betti_numbers': betti_numbers,
                'persistence_pairs': persistence_pairs,
                'homology_groups': homology_groups,
                'computation_time': computation_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"TDA computation failed: {e}")
            return {
                'betti_numbers': {},
                'persistence_pairs': [],
                'homology_groups': [],
                'computation_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _compute_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        from scipy.spatial.distance import pdist, squareform
        
        if len(points) < 2:
            return np.array([[0.0]])
        
        # Compute pairwise distances
        distances = pdist(points, metric='euclidean')
        return squareform(distances)
    
    def _compute_simple_persistence(self, distance_matrix: np.ndarray) -> List[Dict]:
        """Compute simple persistence pairs (placeholder implementation)."""
        persistence_pairs = []
        
        # Simple connected components (0-dimensional homology)
        num_points = len(distance_matrix)
        
        # Find birth and death times for components
        # This is a simplified implementation
        for i in range(min(num_points, 10)):  # Limit for performance
            birth = 0.0
            death = float('inf') if i == 0 else np.mean(distance_matrix) * (i + 1) / 10
            
            persistence_pairs.append({
                'dimension': 0,
                'birth': birth,
                'death': death,
                'persistence': death - birth if death != float('inf') else float('inf')
            })
        
        return persistence_pairs
    
    def _compute_betti_numbers(self, persistence_pairs: List[Dict]) -> Dict[str, List[int]]:
        """Compute Betti numbers from persistence pairs."""
        betti_numbers = {}
        
        # Count features by dimension
        dimensions = set(pair['dimension'] for pair in persistence_pairs)
        
        for dim in dimensions:
            dim_pairs = [p for p in persistence_pairs if p['dimension'] == dim]
            # Count infinite persistence features
            infinite_features = len([p for p in dim_pairs if p['death'] == float('inf')])
            betti_numbers[str(dim)] = [infinite_features]
        
        return betti_numbers
    
    def _create_homology_groups(self, persistence_pairs: List[Dict]) -> List[Dict]:
        """Create homology group summaries."""
        homology_groups = []
        
        # Group by dimension
        dimensions = set(pair['dimension'] for pair in persistence_pairs)
        
        for dim in dimensions:
            dim_pairs = [p for p in persistence_pairs if p['dimension'] == dim]
            
            homology_groups.append({
                'dimension': dim,
                'num_features': len(dim_pairs),
                'persistent_features': len([p for p in dim_pairs if p['death'] == float('inf')]),
                'avg_persistence': np.mean([p['persistence'] for p in dim_pairs if p['persistence'] != float('inf')]) if dim_pairs else 0.0
            })
        
        return homology_groups


class ResultFormatter(MapFunction):
    """Format TDA computation results for output."""
    
    def map(self, value: Row) -> str:
        """Convert TDA result to JSON string."""
        try:
            result = {
                'job_id': value.job_id,
                'timestamp': value.timestamp,
                'window_start': value.window_start,
                'window_end': value.window_end,
                'algorithm': value.algorithm,
                'computation_info': {
                    'num_points': value.num_points,
                    'dimension': value.dimension,
                    'computation_time': value.computation_time,
                    'memory_usage': getattr(value, 'memory_usage', 0),
                    'success': value.success
                },
                'tda_results': {
                    'betti_numbers': value.betti_numbers,
                    'persistence_pairs': value.persistence_pairs,
                    'homology_groups': value.homology_groups
                },
                'metadata': {
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'processor': 'flink-tda-streaming',
                    'version': '1.0.0'
                }
            }
            
            if hasattr(value, 'error'):
                result['error'] = value.error
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error formatting result: {e}")
            return json.dumps({
                'job_id': getattr(value, 'job_id', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': f"Formatting failed: {str(e)}",
                'success': False
            })


def create_kafka_source(config: TDAJobConfig) -> FlinkKafkaConsumer:
    """Create Kafka source for TDA job inputs."""
    properties = {
        'bootstrap.servers': config.kafka_bootstrap_servers,
        'group.id': config.consumer_group,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': 'false',
        'max.poll.records': '1000',
        'fetch.max.bytes': '52428800',  # 50MB
        'session.timeout.ms': '45000',
        'heartbeat.interval.ms': '15000'
    }
    
    return FlinkKafkaConsumer(
        topics=config.input_topic,
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )


def create_kafka_sink(config: TDAJobConfig) -> FlinkKafkaProducer:
    """Create Kafka sink for TDA results."""
    properties = {
        'bootstrap.servers': config.kafka_bootstrap_servers,
        'acks': 'all',
        'retries': '2147483647',
        'max.in.flight.requests.per.connection': '1',
        'enable.idempotence': 'true',
        'compression.type': 'lz4',
        'batch.size': '65536',
        'linger.ms': '100'
    }
    
    return FlinkKafkaProducer(
        topic=config.output_topic,
        serialization_schema=SimpleStringSchema(),
        producer_config=properties
    )


def create_tda_streaming_job(config: TDAJobConfig) -> None:
    """Create and execute TDA streaming job."""
    logger.info(f"Starting TDA streaming job: {config.job_name}")
    
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(config.parallelism)
    env.enable_checkpointing(config.checkpoint_interval)
    env.get_checkpoint_config().set_checkpointing_mode(1)  # EXACTLY_ONCE
    env.get_checkpoint_config().set_min_pause_between_checkpoints(5000)
    env.get_checkpoint_config().set_checkpoint_timeout(300000)  # 5 minutes
    
    # Configure object reuse for performance
    if config.enable_object_reuse:
        env.get_config().enable_object_reuse()
    
    # Set buffer timeout
    env.set_buffer_timeout(config.buffer_timeout)
    
    # Set time characteristic
    env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
    
    # Create Kafka source
    kafka_source = create_kafka_source(config)
    
    # Create data stream
    input_stream = env.add_source(kafka_source)
    
    # Parse input data
    parsed_stream = input_stream.map(
        PointCloudParser(),
        output_type=Types.ROW_NAMED(
            ['job_id', 'timestamp', 'algorithm', 'dimension', 'points', 'num_points', 'event_time'],
            [Types.STRING(), Types.STRING(), Types.STRING(), Types.INT(), 
             Types.OBJECT_ARRAY(Types.OBJECT_ARRAY(Types.FLOAT())), Types.INT(), Types.LONG()]
        )
    )
    
    # Filter valid data
    valid_stream = parsed_stream.filter(lambda row: row.num_points > 0)
    
    # Assign timestamps and watermarks (bounded out-of-orderness)
    watermark_strategy = WatermarkStrategy \
        .for_bounded_out_of_orderness(timedelta(seconds=5)) \
        .with_timestamp_assigner(lambda e, ts: e.event_time)
    timestamped_stream = valid_stream.assign_timestamps_and_watermarks(watermark_strategy)
    
    # Key by job_id for parallel processing
    keyed_stream = timestamped_stream.key_by(lambda row: row.job_id)
    
    # Apply windowing - sliding window for continuous processing
    windowed_stream = keyed_stream.window(
        SlidingEventTimeWindows.of(
            Time.seconds(config.window_size),  # window size
            Time.seconds(config.slide_interval)  # slide interval
        )
    )
    
    # Apply TDA computation
    tda_computation = TDAComputation(config)
    result_stream = windowed_stream.apply(
        tda_computation,
        output_type=Types.ROW_NAMED(
            ['job_id', 'window_start', 'window_end', 'timestamp', 'algorithm', 
             'num_points', 'dimension', 'betti_numbers', 'persistence_pairs', 
             'homology_groups', 'computation_time', 'memory_usage', 'success'],
            [Types.STRING(), Types.LONG(), Types.LONG(), Types.STRING(), Types.STRING(),
             Types.INT(), Types.INT(), Types.OBJECT_ARRAY(Types.STRING()), 
             Types.OBJECT_ARRAY(Types.STRING()), Types.OBJECT_ARRAY(Types.STRING()),
             Types.FLOAT(), Types.LONG(), Types.BOOLEAN()]
        )
    )
    
    # Format results for output
    formatted_stream = result_stream.map(
        ResultFormatter(),
        output_type=Types.STRING()
    )
    
    # Create Kafka sink
    kafka_sink = create_kafka_sink(config)
    
    # Add sink to pipeline
    formatted_stream.add_sink(kafka_sink)
    
    # Execute the job
    logger.info("Executing TDA streaming job...")
    env.execute(config.job_name)


def main():
    """Main entry point for TDA streaming job."""
    parser = argparse.ArgumentParser(description="TDA Streaming Job for Apache Flink")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--job-name", type=str, default="tda-streaming-job", help="Flink job name")
    parser.add_argument("--parallelism", type=int, help="Job parallelism")
    parser.add_argument("--window-size", type=int, help="Processing window size")
    parser.add_argument("--slide-interval", type=int, help="Window slide interval")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = TDAJobConfig(**config_data)
    else:
        config = TDAJobConfig.from_env()
    
    # Override with command line arguments
    if args.job_name:
        config.job_name = args.job_name
    if args.parallelism:
        config.parallelism = args.parallelism
    if args.window_size:
        config.window_size = args.window_size
    if args.slide_interval:
        config.slide_interval = args.slide_interval
    
    logger.info(f"TDA Job Configuration: {asdict(config)}")
    
    try:
        # Install required dependencies
        import subprocess
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "numpy", "scipy", "apache-flink"
        ], check=True, capture_output=True)
        
        # Create and run the streaming job
        create_tda_streaming_job(config)
        
    except KeyboardInterrupt:
        logger.info("Job interrupted by user")
    except Exception as e:
        logger.error(f"Job failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
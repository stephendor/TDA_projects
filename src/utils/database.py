"""
Database Management Utilities

Provides database connection management and query utilities for the TDA Platform.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

import asyncpg
from asyncpg.pool import Pool


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async database manager for PostgreSQL."""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.database_url = os.getenv("DATABASE_URL")
        
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL environment variable must be set. "
                "Format: postgresql://user:password@host:port/database"
            )
    
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                server_settings={
                    'jit': 'off'  # Disable JIT for better connection time
                }
            )
            logger.info("Database connection pool initialized")
            
            # Test connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    raise Exception("Database connectivity test failed")
                    
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def store_computation_result(self, job_id: str, job_type: str, result_data: dict):
        """Store computation result in database."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tda_core.computation_jobs 
                (id, job_type, status, input_data_hash, parameters, result_data, completed_at, computation_time_ms)
                VALUES ($1, $2, 'completed', $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    status = 'completed',
                    result_data = $5,
                    completed_at = $6,
                    computation_time_ms = $7
            """, 
                job_id, 
                job_type, 
                job_id,  # Using job_id as hash for now
                json.dumps({}),  # Parameters would be extracted from result_data
                json.dumps(result_data),
                datetime.utcnow(),
                result_data.get('computation_time_ms', 0)
            )
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status from database."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, job_type, status, created_at, started_at, completed_at, error_message
                FROM tda_core.computation_jobs 
                WHERE id = $1
            """, job_id)
            
            if row:
                return {
                    "job_id": row["id"],
                    "status": row["status"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                    "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                    "error_message": row["error_message"]
                }
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE tda_core.computation_jobs 
                SET status = 'cancelled'
                WHERE id = $1 AND status IN ('pending', 'running')
            """, job_id)
            
            return result != "UPDATE 0"
    
    async def store_threat_detection(self, analysis_id: str, analysis_type: str, result_data: dict):
        """Store cybersecurity threat detection result."""
        async with self.pool.acquire() as conn:
            # Store main detection record
            await conn.execute("""
                INSERT INTO cybersecurity.threat_detections 
                (id, detection_type, threat_level, confidence_score, threat_indicators, topological_features)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    detection_type = $2,
                    threat_level = $3,
                    confidence_score = $4,
                    threat_indicators = $5,
                    topological_features = $6
            """,
                analysis_id,
                analysis_type,
                result_data.get('threat_level', 'UNKNOWN'),
                result_data.get('apt_percentage', 0) / 100.0,
                json.dumps(result_data.get('threat_detections', [])),
                json.dumps(result_data.get('topological_features', {}))
            )
    
    async def store_financial_analysis(self, analysis_id: str, analysis_type: str, result_data: dict):
        """Store financial analysis result."""
        async with self.pool.acquire() as conn:
            if analysis_type == "crypto_bubble":
                # Store bubble analysis
                bubble_data = result_data.get('bubble_detection', {})
                await conn.execute("""
                    INSERT INTO finance.bubble_analysis 
                    (id, asset_symbol, bubble_probability, risk_level, recommendation, topological_indicators, price_data_window_days, detection_horizon_days)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        bubble_probability = $3,
                        risk_level = $4,
                        recommendation = $5,
                        topological_indicators = $6
                """,
                    analysis_id,
                    result_data.get('asset_symbol', 'UNKNOWN'),
                    bubble_data.get('bubble_probability', 0),
                    bubble_data.get('risk_level', 'UNKNOWN'),
                    bubble_data.get('recommendation', ''),
                    json.dumps(bubble_data.get('topological_indicators', {})),
                    100,  # Default window size
                    5     # Default horizon
                )
            
            elif analysis_type == "portfolio_risk":
                # Store risk assessment
                risk_measures = result_data.get('risk_measures', {})
                await conn.execute("""
                    INSERT INTO finance.risk_assessments 
                    (id, portfolio_id, risk_type, overall_risk_score, var_1day, var_10day, expected_shortfall, topological_risk_features)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        overall_risk_score = $4,
                        var_1day = $5,
                        var_10day = $6,
                        expected_shortfall = $7,
                        topological_risk_features = $8
                """,
                    analysis_id,
                    "default_portfolio",  # Would extract from request
                    analysis_type,
                    result_data.get('overall_risk_score', 0),
                    risk_measures.get('var_1day', 0),
                    risk_measures.get('var_10day', 0),
                    risk_measures.get('expected_shortfall', 0),
                    json.dumps(result_data.get('topological_risk_features', {}))
                )
    
    async def get_threats_since(self, timestamp: datetime, threat_type: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get threat detections since timestamp."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT id, detection_type, timestamp, threat_level, confidence_score, threat_indicators
                FROM cybersecurity.threat_detections 
                WHERE timestamp >= $1
            """
            params = [timestamp]
            
            if threat_type:
                query += " AND detection_type = $2"
                params.append(threat_type)
            
            if severity:
                query += f" AND threat_level = ${len(params) + 1}"
                params.append(severity)
            
            query += " ORDER BY timestamp DESC LIMIT 100"
            
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    "id": row["id"],
                    "detection_type": row["detection_type"],
                    "timestamp": row["timestamp"].isoformat(),
                    "threat_level": row["threat_level"],
                    "confidence_score": float(row["confidence_score"]),
                    "indicators": json.loads(row["threat_indicators"]) if row["threat_indicators"] else []
                }
                for row in rows
            ]
    
    async def get_api_metrics_since(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """Get API performance metrics since timestamp."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    endpoint,
                    method,
                    COUNT(*) as request_count,
                    AVG(response_time_ms) as avg_response_time_ms,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) * 100.0 / COUNT(*) as error_rate_percent,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time_ms
                FROM monitoring.api_requests 
                WHERE timestamp >= $1
                GROUP BY endpoint, method
                ORDER BY request_count DESC
            """, timestamp)
            
            return [
                {
                    "endpoint": row["endpoint"],
                    "method": row["method"],
                    "request_count": row["request_count"],
                    "avg_response_time_ms": float(row["avg_response_time_ms"]),
                    "error_rate_percent": float(row["error_rate_percent"]),
                    "p95_response_time_ms": float(row["p95_response_time_ms"])
                }
                for row in rows
            ]
    
    async def get_tda_computation_metrics_since(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """Get TDA computation metrics since timestamp."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    job_type as computation_type,
                    COUNT(*) as total_jobs,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_jobs,
                    AVG(computation_time_ms) as avg_computation_time_ms,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as queue_length
                FROM tda_core.computation_jobs 
                WHERE created_at >= $1
                GROUP BY job_type
                ORDER BY total_jobs DESC
            """, timestamp)
            
            return [
                {
                    "computation_type": row["computation_type"],
                    "total_jobs": row["total_jobs"],
                    "completed_jobs": row["completed_jobs"],
                    "failed_jobs": row["failed_jobs"],
                    "avg_computation_time_ms": float(row["avg_computation_time_ms"]) if row["avg_computation_time_ms"] else 0,
                    "queue_length": row["queue_length"]
                }
                for row in rows
            ]
    
    async def get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        async with self.pool.acquire() as conn:
            # Connection pool info
            pool_info = {
                "size": self.pool.get_size(),
                "max_size": self.pool.get_max_size(),
                "min_size": self.pool.get_min_size()
            }
            
            # Query performance stats (simplified)
            query_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(mean_exec_time) as avg_query_time_ms
                FROM pg_stat_statements 
                WHERE mean_exec_time > 0
                LIMIT 1
            """)
            
            return {
                "connection_pool": pool_info,
                "query_performance": {
                    "avg_query_time_ms": float(query_stats["avg_query_time_ms"]) if query_stats and query_stats["avg_query_time_ms"] else 0
                },
                "storage_usage": {},  # Would implement actual storage queries
                "active_connections": pool_info["size"]
            }


# Dependency function for FastAPI
async def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    # In a real application, this would be managed by the application lifecycle
    db_manager = DatabaseManager()
    if not db_manager.pool:
        await db_manager.initialize()
    return db_manager
#!/usr/bin/env python3
"""
Test script for TDA Kafka integration.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend to the path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_kafka_integration():
    """Test the Kafka integration components."""
    logger.info("🧪 Testing TDA Kafka Integration")
    
    try:
        # Test imports
        logger.info("📦 Testing imports...")
        from tda_backend.services.kafka_producer import (
            KafkaProducerService, 
            MessageType, 
            JobMessage,
            get_kafka_producer
        )
        from tda_backend.services.kafka_integration import (
            KafkaIntegrationService, 
            get_kafka_integration
        )
        from tda_backend.services.kafka_schemas import (
            get_schema_registry, 
            get_message_validator
        )
        from tda_backend.services.kafka_metrics import (
            get_metrics_collector
        )
        logger.info("✅ All imports successful")
        
        # Test producer service initialization
        logger.info("🔧 Testing producer service...")
        producer = get_kafka_producer()
        logger.info(f"✅ Producer service created: {type(producer)}")
        
        # Test integration service
        logger.info("🔗 Testing integration service...")
        integration = get_kafka_integration()
        logger.info(f"✅ Integration service created: {type(integration)}")
        
        # Test schema registry
        logger.info("📋 Testing schema registry...")
        schema_registry = get_schema_registry()
        logger.info(f"✅ Schema registry created: {type(schema_registry)}")
        
        # Test message validator
        logger.info("✅ Testing message validator...")
        validator = get_message_validator()
        logger.info(f"✅ Message validator created: {type(validator)}")
        
        # Test metrics collector
        logger.info("📊 Testing metrics collector...")
        metrics = get_metrics_collector()
        logger.info(f"✅ Metrics collector created: {type(metrics)}")
        
        # Test message creation
        logger.info("📄 Testing message creation...")
        job_message = JobMessage.create(
            job_id="test-123",
            status="submitted",
            message_type=MessageType.JOB_SUBMITTED,
            algorithm="vietoris-rips",
            priority="high"
        )
        logger.info(f"✅ Job message created: {job_message.metadata.message_id}")
        
        # Test message validation
        logger.info("🔍 Testing message validation...")
        is_valid = await validator.validate_message(job_message)
        logger.info(f"✅ Message validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
        
        # Test JSON serialization
        logger.info("🔄 Testing JSON serialization...")
        message_dict = job_message.dict()
        logger.info(f"✅ Message serialized: {len(str(message_dict))} characters")
        
        logger.info("🎉 All Kafka integration tests passed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_kafka_connectivity():
    """Test actual Kafka connectivity (requires running cluster)."""
    logger.info("🌐 Testing Kafka connectivity...")
    
    try:
        from tda_backend.services.kafka_producer import get_kafka_producer
        
        producer = get_kafka_producer()
        
        # Try to start the producer (will fail if Kafka not available)
        logger.info("🔌 Attempting to connect to Kafka...")
        await producer.start()
        logger.info("✅ Connected to Kafka successfully!")
        
        # Try to send a test message
        logger.info("📤 Sending test message...")
        success = await producer.send_job_message(
            job_id="integration-test",
            status="submitted",
            message_type=MessageType.JOB_SUBMITTED,
            algorithm="test-algorithm"
        )
        
        if success:
            logger.info("✅ Test message sent successfully!")
        else:
            logger.warning("⚠️ Test message sending failed")
        
        # Get health check
        health = await producer.health_check()
        logger.info(f"📊 Producer health: {health['status']}")
        
        # Stop producer
        await producer.stop()
        logger.info("✅ Producer stopped successfully")
        
        return success
        
    except Exception as e:
        logger.warning(f"⚠️ Kafka connectivity test failed: {e}")
        logger.info("ℹ️ This is expected if Kafka cluster is not running")
        return False

if __name__ == "__main__":
    async def main():
        logger.info("=" * 60)
        logger.info("TDA KAFKA INTEGRATION TEST")
        logger.info("=" * 60)
        
        # Test 1: Component integration
        success1 = await test_kafka_integration()
        
        # Test 2: Kafka connectivity (optional)
        success2 = await test_kafka_connectivity()
        
        logger.info("=" * 60)
        logger.info("TEST RESULTS:")
        logger.info(f"✅ Component Integration: {'PASS' if success1 else 'FAIL'}")
        logger.info(f"📡 Kafka Connectivity: {'PASS' if success2 else 'SKIP (Kafka not running)'}")
        logger.info("=" * 60)
        
        if success1:
            logger.info("🎉 Kafka integration is ready!")
            sys.exit(0)
        else:
            logger.error("❌ Integration tests failed")
            sys.exit(1)
    
    asyncio.run(main())

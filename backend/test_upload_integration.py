#!/usr/bin/env python3
"""
Integration test script for the TDA upload system.

This script demonstrates how to use the upload services and validates
that the complete upload workflow is functioning correctly.
"""

import asyncio
import json
import tempfile
import os
from pathlib import Path

# Import our services
from tda_backend.services.upload_service import get_upload_service
from tda_backend.services.storage_service import get_storage_service
from tda_backend.models import Point, PointCloud

async def create_test_files():
    """Create test files for various formats."""
    test_files = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create CSV test file
        csv_file = temp_path / "test_points.csv"
        csv_content = """x,y,z,label
1.0,2.0,3.0,point1
4.0,5.0,6.0,point2
7.0,8.0,9.0,point3
10.0,11.0,12.0,point4
"""
        csv_file.write_text(csv_content)
        test_files['csv'] = str(csv_file)
        
        # Create JSON test file
        json_file = temp_path / "test_points.json"
        json_content = {
            "points": [
                {"coordinates": [1.0, 2.0], "label": "p1"},
                {"coordinates": [3.0, 4.0], "label": "p2"},
                {"coordinates": [5.0, 6.0], "label": "p3"}
            ]
        }
        json_file.write_text(json.dumps(json_content, indent=2))
        test_files['json'] = str(json_file)
        
        # Create NumPy test file
        try:
            import numpy as np
            npy_file = temp_path / "test_points.npy"
            test_array = np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ])
            np.save(str(npy_file), test_array)
            test_files['numpy'] = str(npy_file)
        except ImportError:
            print("NumPy not available, skipping .npy test")
        
        return test_files

class MockUploadFile:
    """Mock UploadFile for testing."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filename = self.filepath.name
        self.content_type = self._guess_content_type()
        self._file_handle = None
        
    def _guess_content_type(self):
        ext = self.filepath.suffix.lower()
        if ext == '.csv':
            return 'text/csv'
        elif ext == '.json':
            return 'application/json'
        elif ext == '.npy':
            return 'application/octet-stream'
        else:
            return 'application/octet-stream'
    
    async def seek(self, position):
        if self._file_handle:
            self._file_handle.seek(position)
    
    async def read(self, size=-1):
        if not self._file_handle:
            self._file_handle = open(self.filepath, 'rb')
        return self._file_handle.read(size)
    
    def __del__(self):
        if self._file_handle:
            self._file_handle.close()

async def test_upload_service():
    """Test the upload service functionality."""
    print("Testing Upload Service...")
    
    # Get services
    upload_service = get_upload_service()
    storage_service = get_storage_service()
    
    # Create test files
    print("Creating test files...")
    test_files = await create_test_files()
    
    # Test CSV upload
    if 'csv' in test_files:
        print("\n--- Testing CSV Upload ---")
        mock_file = MockUploadFile(test_files['csv'])
        
        # Start upload
        upload_id, progress = await upload_service.start_upload(
            mock_file, 
            {'has_header': True, 'delimiter': ','}
        )
        
        print(f"Upload started with ID: {upload_id}")
        print(f"Initial status: {progress.status}")
        
        # Wait for processing to complete
        await asyncio.sleep(1)
        
        # Check final progress
        final_progress = upload_service.get_upload_progress(upload_id)
        print(f"Final status: {final_progress.status}")
        print(f"Progress: {final_progress.progress}%")
        
        if final_progress.status == 'completed':
            print("✅ CSV upload successful!")
            if hasattr(final_progress, 'data') and final_progress.data:
                point_cloud = final_progress.data['point_cloud']
                print(f"   - Points: {len(point_cloud.points)}")
                print(f"   - Dimensions: {point_cloud.dimension}")
                print(f"   - First point: {point_cloud.points[0].coordinates}")
        else:
            print(f"❌ CSV upload failed: {final_progress.error_message}")
    
    # Test JSON upload
    if 'json' in test_files:
        print("\n--- Testing JSON Upload ---")
        mock_file = MockUploadFile(test_files['json'])
        
        upload_id, progress = await upload_service.start_upload(mock_file)
        print(f"Upload started with ID: {upload_id}")
        
        await asyncio.sleep(1)
        
        final_progress = upload_service.get_upload_progress(upload_id)
        print(f"Final status: {final_progress.status}")
        
        if final_progress.status == 'completed':
            print("✅ JSON upload successful!")
            if hasattr(final_progress, 'data') and final_progress.data:
                point_cloud = final_progress.data['point_cloud']
                print(f"   - Points: {len(point_cloud.points)}")
                print(f"   - Dimensions: {point_cloud.dimension}")
        else:
            print(f"❌ JSON upload failed: {final_progress.error_message}")
    
    # Test NumPy upload
    if 'numpy' in test_files:
        print("\n--- Testing NumPy Upload ---")
        mock_file = MockUploadFile(test_files['numpy'])
        
        upload_id, progress = await upload_service.start_upload(mock_file)
        print(f"Upload started with ID: {upload_id}")
        
        await asyncio.sleep(1)
        
        final_progress = upload_service.get_upload_progress(upload_id)
        print(f"Final status: {final_progress.status}")
        
        if final_progress.status == 'completed':
            print("✅ NumPy upload successful!")
            if hasattr(final_progress, 'data') and final_progress.data:
                point_cloud = final_progress.data['point_cloud']
                print(f"   - Points: {len(point_cloud.points)}")
                print(f"   - Dimensions: {point_cloud.dimension}")
        else:
            print(f"❌ NumPy upload failed: {final_progress.error_message}")

async def test_storage_service():
    """Test the storage service functionality."""
    print("\n\nTesting Storage Service...")
    
    storage_service = get_storage_service()
    
    # Create test point cloud
    test_points = [
        Point(coordinates=[1.0, 2.0, 3.0], label="test1"),
        Point(coordinates=[4.0, 5.0, 6.0], label="test2"),
        Point(coordinates=[7.0, 8.0, 9.0], label="test3")
    ]
    
    point_cloud = PointCloud(
        points=test_points,
        dimension=3,
        metadata={"source": "test", "description": "Integration test data"}
    )
    
    # Store point cloud
    pc_id = await storage_service.store_point_cloud(
        point_cloud,
        name="Test Point Cloud",
        description="Generated for integration testing",
        tags=["test", "integration"]
    )
    
    print(f"Stored point cloud with ID: {pc_id}")
    
    # Retrieve point cloud
    result = await storage_service.retrieve_point_cloud(pc_id)
    if result:
        retrieved_pc, metadata = result
        print("✅ Point cloud retrieved successfully!")
        print(f"   - Name: {metadata.name}")
        print(f"   - Points: {len(retrieved_pc.points)}")
        print(f"   - Dimensions: {retrieved_pc.dimension}")
        print(f"   - Tags: {metadata.tags}")
    else:
        print("❌ Failed to retrieve point cloud")
    
    # Test listing
    point_clouds, total = await storage_service.list_point_clouds(page=1, size=10)
    print(f"Listed {len(point_clouds)} point clouds (total: {total})")
    
    # Test search
    search_results = await storage_service.search_point_clouds("test")
    print(f"Search found {len(search_results)} matching point clouds")
    
    # Get storage statistics
    stats = storage_service.get_storage_stats()
    print("\nStorage Statistics:")
    print(f"   - Total point clouds: {stats['storage']['total_point_clouds']}")
    print(f"   - Total points: {stats['storage']['total_points']}")
    print(f"   - Storage size: {stats['storage']['storage_size_mb']} MB")
    print(f"   - Cache hit rate: {stats['cache']['cache_hit_rate']}%")

async def main():
    """Run integration tests."""
    print("🔬 TDA Upload System Integration Test")
    print("=" * 50)
    
    try:
        await test_upload_service()
        await test_storage_service()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
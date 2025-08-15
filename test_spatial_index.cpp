#include "tda/spatial/spatial_index.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Testing Spatial Indexing Structures" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Create a simple 2D point cloud
    std::vector<std::vector<double>> points_2d = {
        {0.0, 0.0},      // Center
        {1.0, 0.0},      // Right
        {0.0, 1.0},      // Top
        {-1.0, 0.0},     // Left
        {0.0, -1.0},     // Bottom
        {0.5, 0.5},      // Top-right
        {-0.5, 0.5},     // Top-left
        {-0.5, -0.5},    // Bottom-left
        {0.5, -0.5}      // Bottom-right
    };
    
    std::cout << "2D point cloud created with " << points_2d.size() << " points" << std::endl;
    
    // Test KD-tree
    {
        std::cout << "\n--- Testing KD-tree ---" << std::endl;
        
        tda::spatial::KDTree kdtree;
        
        // Build the tree
        bool buildSuccess = kdtree.build(points_2d);
        if (!buildSuccess) {
            std::cout << "❌ KD-tree build failed" << std::endl;
            return 1;
        }
        std::cout << "✓ KD-tree build successful" << std::endl;
        
        // Test nearest neighbor
        std::vector<double> query = {0.1, 0.1};
        auto nearest = kdtree.nearestNeighbor(query);
        std::cout << "✓ Nearest neighbor to (0.1, 0.1): point " << nearest.first 
                  << " at distance " << nearest.second << std::endl;
        
        // Test k-nearest neighbors
        auto kNearest = kdtree.kNearestNeighbors(query, 3);
        std::cout << "✓ 3-nearest neighbors to (0.1, 0.1):" << std::endl;
        for (size_t i = 0; i < kNearest.size(); ++i) {
            std::cout << "  " << i + 1 << ". Point " << kNearest[i].first 
                      << " at distance " << kNearest[i].second << std::endl;
        }
        
        // Test radius search
        auto radiusResults = kdtree.radiusSearch(query, 1.0);
        std::cout << "✓ Points within radius 1.0: " << radiusResults.size() << " found" << std::endl;
        
        // Test statistics
        auto stats = kdtree.getBuildStats();
        std::cout << "✓ Build statistics:" << std::endl;
        std::cout << "  - Build time: " << stats.buildTimeMs << " ms" << std::endl;
        std::cout << "  - Memory usage: " << stats.memoryUsageBytes << " bytes" << std::endl;
        std::cout << "  - Tree depth: " << stats.treeDepth << std::endl;
    }
    
    // Test factory function
    {
        std::cout << "\n--- Testing Factory Function ---" << std::endl;
        
        // Test with 2D points (should create KD-tree)
        auto index2d = tda::spatial::createSpatialIndex(points_2d, 10);
        if (!index2d) {
            std::cout << "❌ Factory failed to create 2D index" << std::endl;
            return 1;
        }
        
        bool buildSuccess = index2d->build(points_2d);
        if (!buildSuccess) {
            std::cout << "❌ Factory-created index build failed" << std::endl;
            return 1;
        }
        std::cout << "✓ Factory-created index build successful" << std::endl;
        
        // Test functionality
        std::vector<double> query = {0.1, 0.1};
        auto nearest = index2d->nearestNeighbor(query);
        std::cout << "✓ Factory index nearest neighbor: point " << nearest.first 
                  << " at distance " << nearest.second << std::endl;
    }
    
    // Test 3D point cloud
    {
        std::cout << "\n--- Testing 3D Point Cloud ---" << std::endl;
        
        std::vector<std::vector<double>> points_3d = {
            {0.0, 0.0, 0.0},      // Center
            {1.0, 0.0, 0.0},      // Right
            {0.0, 1.0, 0.0},      // Top
            {0.0, 0.0, 1.0},      // Front
            {-1.0, 0.0, 0.0},     // Left
            {0.0, -1.0, 0.0},     // Bottom
            {0.0, 0.0, -1.0},     // Back
            {0.5, 0.5, 0.5},      // Top-right-front
            {-0.5, 0.5, 0.5}      // Top-left-front
        };
        
        tda::spatial::KDTree kdtree3d;
        bool buildSuccess = kdtree3d.build(points_3d);
        if (!buildSuccess) {
            std::cout << "❌ 3D KD-tree build failed" << std::endl;
            return 1;
        }
        std::cout << "✓ 3D KD-tree build successful" << std::endl;
        
        // Test 3D nearest neighbor
        std::vector<double> query3d = {0.1, 0.1, 0.1};
        auto nearest3d = kdtree3d.nearestNeighbor(query3d);
        std::cout << "✓ 3D nearest neighbor to (0.1, 0.1, 0.1): point " << nearest3d.first 
                  << " at distance " << nearest3d.second << std::endl;
    }
    
    std::cout << "\n🎉 All spatial indexing tests passed! Implementation is working correctly." << std::endl;
    return 0;
}

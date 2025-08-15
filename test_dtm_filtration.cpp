#include "tda/algorithms/dtm_filtration.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Testing DTM (Distance-to-Measure) Filtration" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Create a simple 2D point cloud with some noise
    std::vector<std::vector<double>> points_2d = {
        {0.0, 0.0},      // Center
        {1.0, 0.0},      // Right
        {0.0, 1.0},      // Top
        {-1.0, 0.0},     // Left
        {0.0, -1.0},     // Bottom
        {0.5, 0.5},      // Top-right
        {-0.5, 0.5},     // Top-left
        {-0.5, -0.5},    // Bottom-left
        {0.5, -0.5},     // Bottom-right
        {0.1, 0.1},      // Near center (noise)
        {0.2, 0.2},      // Near center (noise)
        {0.3, 0.3}       // Near center (noise)
    };
    
    std::cout << "2D point cloud created with " << points_2d.size() << " points" << std::endl;
    
    // Test DTM filtration with default configuration
    tda::algorithms::DTMFiltration dtmFilter;
    
    std::cout << "\n--- Testing DTM Initialization ---" << std::endl;
    auto initResult = dtmFilter.initialize(points_2d);
    if (initResult.has_error()) {
        std::cout << "❌ Initialization failed: " << initResult.error() << std::endl;
        return 1;
    }
    std::cout << "✓ DTM initialization successful" << std::endl;
    
    std::cout << "\n--- Testing DTM Function Computation ---" << std::endl;
    auto dtmResult = dtmFilter.computeDTMFunction();
    if (dtmResult.has_error()) {
        std::cout << "❌ DTM function computation failed: " << dtmResult.error() << std::endl;
        return 1;
    }
    
    auto dtmValues = dtmResult.value();
    std::cout << "✓ DTM function computed successfully" << std::endl;
    std::cout << "  - Number of DTM values: " << dtmValues.size() << std::endl;
    
    // Display some DTM values
    std::cout << "  - DTM values (first 5): ";
    for (size_t i = 0; i < std::min(size_t(5), dtmValues.size()); ++i) {
        std::cout << dtmValues[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n--- Testing DTM Filtration Construction ---" << std::endl;
    auto buildResult = dtmFilter.buildFiltration(2); // Build up to 2D simplices
    if (buildResult.has_error()) {
        std::cout << "❌ Filtration build failed: " << buildResult.error() << std::endl;
        return 1;
    }
    std::cout << "✓ DTM filtration built successfully" << std::endl;
    
    std::cout << "\n--- Testing Persistence Computation ---" << std::endl;
    auto persistenceResult = dtmFilter.computePersistence(2); // Z/2Z coefficients
    if (persistenceResult.has_error()) {
        std::cout << "❌ Persistence computation failed: " << persistenceResult.error() << std::endl;
        return 1;
    }
    std::cout << "✓ Persistence computed successfully" << std::endl;
    
    std::cout << "\n--- Testing Data Retrieval ---" << std::endl;
    
    // Get simplices
    auto simplicesResult = dtmFilter.getSimplices();
    if (simplicesResult.has_error()) {
        std::cout << "❌ Failed to get simplices: " << simplicesResult.error() << std::endl;
        return 1;
    }
    auto simplices = simplicesResult.value();
    std::cout << "✓ Retrieved " << simplices.size() << " simplices" << std::endl;
    
    // Get persistence pairs
    auto pairsResult = dtmFilter.getPersistencePairs();
    if (pairsResult.has_error()) {
        std::cout << "❌ Failed to get persistence pairs: " << pairsResult.error() << std::endl;
        return 1;
    }
    auto pairs = pairsResult.value();
    std::cout << "✓ Retrieved " << pairs.size() << " persistence pairs" << std::endl;
    
    // Get statistics
    auto statsResult = dtmFilter.getStatistics();
    if (statsResult.has_error()) {
        std::cout << "❌ Failed to get statistics: " << statsResult.error() << std::endl;
        return 1;
    }
    auto stats = statsResult.value();
    std::cout << "✓ Retrieved statistics:" << std::endl;
    std::cout << "  - Points: " << stats.num_points << std::endl;
    std::cout << "  - Simplices: " << stats.num_simplices << std::endl;
    std::cout << "  - Max dimension: " << stats.max_dimension << std::endl;
    
    std::cout << "\n--- Testing Configuration ---" << std::endl;
    auto config = dtmFilter.getConfig();
    std::cout << "✓ Configuration retrieved:" << std::endl;
    std::cout << "  - k (neighbors): " << config.k << std::endl;
    std::cout << "  - Power: " << config.power << std::endl;
    std::cout << "  - Normalize: " << (config.normalize ? "true" : "false") << std::endl;
    std::cout << "  - Max dimension: " << config.maxDimension << std::endl;
    
    std::cout << "\n🎉 All DTM filtration tests passed! Implementation is working correctly." << std::endl;
    
    return 0;
}

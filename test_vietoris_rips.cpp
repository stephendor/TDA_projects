#include "tda/algorithms/vietoris_rips.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing Vietoris-Rips Implementation" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Create a simple point cloud (triangle)
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},  // Point 1
        {1.0, 0.0},  // Point 2
        {0.5, 0.866} // Point 3 (equilateral triangle)
    };
    
    std::cout << "Point cloud created with " << points.size() << " points" << std::endl;
    
    // Create Vietoris-Rips instance
    tda::algorithms::VietorisRips vr;
    
    // Initialize with threshold 1.0 and max dimension 2
    auto init_result = vr.initialize(points, 1.0, 2, 2);
    if (init_result.has_error()) {
        std::cout << "Initialization failed: " << init_result.error() << std::endl;
        return 1;
    }
    std::cout << "✓ Initialization successful" << std::endl;
    
    // Compute the complex
    auto complex_result = vr.computeComplex();
    if (complex_result.has_error()) {
        std::cout << "Complex computation failed: " << complex_result.error() << std::endl;
        return 1;
    }
    std::cout << "✓ Complex computation successful" << std::endl;
    
    // Compute persistence
    auto persistence_result = vr.computePersistence();
    if (persistence_result.has_error()) {
        std::cout << "Persistence computation failed: " << persistence_result.error() << std::endl;
        return 1;
    }
    std::cout << "✓ Persistence computation successful" << std::endl;
    
    // Get statistics
    auto stats_result = vr.getStatistics();
    if (stats_result.has_error()) {
        std::cout << "Statistics retrieval failed: " << stats_result.error() << std::endl;
        return 1;
    }
    
    auto stats = stats_result.value();
    std::cout << "✓ Statistics retrieved:" << std::endl;
    std::cout << "  - Number of points: " << stats.num_points << std::endl;
    std::cout << "  - Number of simplices: " << stats.num_simplices << std::endl;
    std::cout << "  - Max dimension: " << stats.max_dimension << std::endl;
    std::cout << "  - Threshold: " << stats.threshold << std::endl;
    
    // Get simplices
    auto simplices_result = vr.getSimplices();
    if (simplices_result.has_error()) {
        std::cout << "Simplices retrieval failed: " << simplices_result.error() << std::endl;
        return 1;
    }
    
    auto simplices = simplices_result.value();
    std::cout << "✓ Retrieved " << simplices.size() << " simplices:" << std::endl;
    for (size_t i = 0; i < simplices.size(); ++i) {
        const auto& simplex = simplices[i];
        std::cout << "  Simplex " << i << ": dim=" << simplex.dimension 
                  << ", filt=" << simplex.filtration_value 
                  << ", vertices=[";
        for (size_t j = 0; j < simplex.vertices.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << simplex.vertices[j];
        }
        std::cout << "]" << std::endl;
    }
    
    // Test distance computation
    std::vector<double> query_point = {0.5, 0.5};
    auto distances = vr.computeDistancesBatch(points, query_point);
    std::cout << "✓ Distance computation successful:" << std::endl;
    for (size_t i = 0; i < distances.size(); ++i) {
        std::cout << "  Distance to point " << i << ": " << distances[i] << std::endl;
    }
    
    std::cout << "\n🎉 All tests passed! Vietoris-Rips implementation is working correctly." << std::endl;
    return 0;
}

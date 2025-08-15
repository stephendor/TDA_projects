#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "tda/algorithms/vietoris_rips.hpp"
#include "tda/core/types.hpp"

// Minimal test file for Vietoris-Rips implementation
// Uses standard assert instead of GTest framework

int main() {
        // Create a simple 2D point cloud: square with diagonal
        // Points: (0,0), (1,0), (0,1), (1,1)
        // This will create interesting topology for testing
    // Create a simple 2D point cloud: square
    // Points: (0,0), (1,0), (0,1), (1,1)
    // This will create interesting topology for testing
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},  // Point 0
        {1.0, 0.0},  // Point 1
        {0.0, 1.0},  // Point 2
        {1.0, 1.0}   // Point 3
    };
    
    // Threshold that will include the diagonal but not create 3-simplices
    double threshold = 1.5;
    int max_dimension = 2;

    tda::algorithms::VietorisRips vr;
    
    // Test successful initialization
    std::cout << "Testing initialization..." << std::endl;
    auto result = vr.initialize(points, threshold, max_dimension);
    assert(result.has_value());
    
    // Test empty point cloud
    auto empty_result = vr.initialize({}, threshold, max_dimension);
    assert(!empty_result.has_value());
    assert(empty_result.error().find("empty") != std::string::npos);
    
    // Test negative threshold
    auto neg_threshold_result = vr.initialize(points, -1.0, max_dimension);
    assert(!neg_threshold_result.has_value());
    assert(neg_threshold_result.error().find("positive") != std::string::npos);
    
    // Test negative dimension
    auto neg_dim_result = vr.initialize(points, threshold, -1);
    assert(!neg_dim_result.has_value());
    assert(neg_dim_result.error().find("non-negative") != std::string::npos);
    
    // Reset VR object with valid initialization for further tests
    result = vr.initialize(points, threshold, max_dimension);
    assert(result.has_value());
    
    // Test complex computation
    std::cout << "Testing complex computation..." << std::endl;
    auto complex_result = vr.computeComplex();
    assert(complex_result.has_value());
    
    // Get statistics
    auto stats_result = vr.getStatistics();
    assert(stats_result.has_value());
    
    auto stats = stats_result.value();
    assert(stats.num_points == 4);
    assert(stats.num_simplices > 0);
    assert(stats.max_dimension == 2);
    assert(std::abs(stats.threshold - threshold) < 1e-10);
    
    // Verify simplex counts by dimension
    assert(stats.simplex_count_by_dim[0] == 4);  // 4 vertices
    assert(stats.simplex_count_by_dim[1] >= 5);  // At least 5 edges (including diagonal)
    assert(stats.simplex_count_by_dim[2] > 0);   // At least 1 triangle

    // Test persistence computation
    std::cout << "Testing persistence computation..." << std::endl;
    auto persistence_result = vr.computePersistence();
    assert(persistence_result.has_value());
    
    // Get persistence pairs
    auto pairs_result = vr.getPersistencePairs();
    assert(pairs_result.has_value());
    
    auto pairs = pairs_result.value();
    assert(!pairs.empty());
    
    // Verify persistence pair properties
    for (const auto& pair : pairs) {
        // dimension is unsigned, so no need to check >= 0
        assert(static_cast<int>(pair.dimension) <= max_dimension);
        assert(pair.birth >= 0.0);
        assert(pair.death > pair.birth || pair.is_infinite());  // Death > birth unless infinite
        if (pair.is_finite()) {
            // For finite pairs, persistence equals death - birth
            assert(std::abs(pair.get_persistence() - (pair.death - pair.birth)) < 1e-10);
        } else {
            // For infinite pairs, death is +inf and persistence is infinite
            assert(pair.is_infinite());
        }
    }

    // Test Betti numbers
    std::cout << "Testing Betti numbers..." << std::endl;
    auto betti_result = vr.getBettiNumbers();
    assert(betti_result.has_value());
    
    auto betti_numbers = betti_result.value();
    assert(betti_numbers.size() == static_cast<size_t>(max_dimension + 1));
    
    // β₀ should be 1 (one connected component)
    assert(betti_numbers[0] == 1);
    
    // β₁ should be at least 0 (no holes guaranteed)
    assert(betti_numbers[1] >= 0);
    
    // β₂ should be 0 (no 2D voids in 2D space)
    if (betti_numbers.size() > 2) {
        assert(betti_numbers[2] == 0);
    }

    // Test simplex retrieval
    std::cout << "Testing simplex retrieval..." << std::endl;
    auto simplices_result = vr.getSimplices();
    assert(simplices_result.has_value());
    
    auto simplices = simplices_result.value();
    assert(!simplices.empty());
    
    // Verify simplex properties
    for (const auto& simplex : simplices) {
    // simplex.dimension is signed in SimplexInfo, ensure bounds
    assert(simplex.dimension >= 0);
    assert(simplex.dimension <= max_dimension);
        assert(simplex.filtration_value >= 0.0);
        assert(simplex.vertices.size() == static_cast<size_t>(simplex.dimension + 1));
        
        // Verify vertex indices are valid
        for (int vertex : simplex.vertices) {
            assert(vertex >= 0);
            assert(vertex < static_cast<int>(points.size()));
        }
    }

    // Test batch distance computation
    std::cout << "Testing distance computation..." << std::endl;
    std::vector<double> query_point = {0.5, 0.5};  // Center of the square
    
    auto distances = vr.computeDistancesBatch(points, query_point);
    assert(distances.size() == points.size());
    
    // Verify distances are reasonable
    for (size_t i = 0; i < distances.size(); ++i) {
        assert(distances[i] >= 0.0);
        
        // Manual distance calculation for verification
        double expected_dist = 0.0;
        for (size_t j = 0; j < points[i].size(); ++j) {
            double diff = points[i][j] - query_point[j];
            expected_dist += diff * diff;
        }
        expected_dist = std::sqrt(expected_dist);
        
        assert(std::abs(distances[i] - expected_dist) < 1e-10);
    }

    // Test error handling
    std::cout << "Testing error handling..." << std::endl;
    tda::algorithms::VietorisRips vr2;
    
    // Try to get simplices before computing complex
    auto simplices_result2 = vr2.getSimplices();
    assert(!simplices_result2.has_value());
    assert(simplices_result2.error().find("not computed") != std::string::npos);
    
    // Try to get persistence pairs before computing persistence
    auto init_result2 = vr2.initialize(points, threshold, max_dimension);
    assert(init_result2.has_value());
    
    auto complex_result2 = vr2.computeComplex();
    assert(complex_result2.has_value());
    
    auto pairs_result2 = vr2.getPersistencePairs();
    assert(!pairs_result2.has_value());
    assert(pairs_result2.error().find("not computed") != std::string::npos);
    
    // Try to get Betti numbers before computing persistence
    auto betti_result2 = vr2.getBettiNumbers();
    assert(!betti_result2.has_value());
    assert(betti_result2.error().find("not computed") != std::string::npos);

    // Test move semantics
    std::cout << "Testing move semantics..." << std::endl;
    tda::algorithms::VietorisRips vr1;
    
    // Initialize and compute complex
    auto init_result1 = vr1.initialize(points, threshold, max_dimension);
    assert(init_result1.has_value());
    
    auto complex_result1 = vr1.computeComplex();
    assert(complex_result1.has_value());
    
    // Move to new instance
    tda::algorithms::VietorisRips vr3 = std::move(vr1);
    
    // Original should be in moved-from state
    auto stats_result1 = vr1.getStatistics();
    assert(!stats_result1.has_value());
    
    // New instance should work
    auto stats_result3 = vr3.getStatistics();
    assert(stats_result3.has_value());
    
    auto stats3 = stats_result3.value();
    assert(stats3.num_points == 4);
    assert(stats3.num_simplices > 0);

    // Create a slightly larger point cloud for scale testing
    std::cout << "Testing with larger point cloud..." << std::endl;
    std::vector<std::vector<double>> large_points;
    const int num_points = 20; // Keep small for quick test
    const int dimension = 3;
    
    for (int i = 0; i < num_points; ++i) {
        std::vector<double> point;
        for (int j = 0; j < dimension; ++j) {
            point.push_back(static_cast<double>(i + j) / num_points);
        }
        large_points.push_back(std::move(point));
    }
    
    tda::algorithms::VietorisRips vr4;
    
    // Test with larger point cloud
    auto init_result4 = vr4.initialize(large_points, 0.5, 2);
    assert(init_result4.has_value());
    
    auto complex_result4 = vr4.computeComplex();
    assert(complex_result4.has_value());
    
    auto stats_result4 = vr4.getStatistics();
    assert(stats_result4.has_value());
    
    auto stats4 = stats_result4.value();
    assert(stats4.num_points == static_cast<size_t>(num_points));
    assert(stats4.num_simplices > 0);

    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}

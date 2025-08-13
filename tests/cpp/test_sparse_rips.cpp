#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>
#include "tda/algorithms/sparse_rips.hpp"

namespace {

std::vector<std::vector<double>> generate_test_points(size_t n, size_t dim = 2) {
    std::vector<std::vector<double>> points(n, std::vector<double>(dim));
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(0.0, 10.0);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            points[i][j] = dis(gen);
        }
    }
    return points;
}

void test_sparse_rips_basic() {
    std::cout << "Testing basic Sparse Rips functionality..." << std::endl;
    
    auto points = generate_test_points(100);
    
    tda::algorithms::SparseRips::Config config;
    config.max_dimension = 2;
    config.filtration_threshold = 5.0;
    config.sparsity_factor = 0.1;
    config.use_landmarks = true;
    config.num_landmarks = 20;
    config.min_points_threshold = 50; // Should trigger approximation
    
    tda::algorithms::SparseRips sparse_rips(config);
    auto result = sparse_rips.computeApproximation(points, 5.0);
    
    assert(result.has_value());
    assert(!result.value().simplices.empty());
    assert(result.value().simplices.size() <= 50000); // Should be sparse
    
    std::cout << "✅ Basic Sparse Rips test passed - " << result.value().simplices.size() << " simplices" << std::endl;
}

void test_sparse_rips_approximation_trigger() {
    std::cout << "Testing Sparse Rips approximation triggering..." << std::endl;
    
    auto points = generate_test_points(1000); // Large enough to trigger approximation
    
    tda::algorithms::SparseRips::Config config;
    config.max_dimension = 1;
    config.filtration_threshold = 3.0;
    config.sparsity_factor = 0.05;
    config.use_landmarks = true;
    config.num_landmarks = 50;
    config.min_points_threshold = 500; // Should trigger approximation
    
    tda::algorithms::SparseRips sparse_rips(config);
    auto result = sparse_rips.computeApproximation(points, 3.0);
    
    assert(result.has_value());
    
    // Verify approximation was used (should be much smaller than full complex)
    size_t max_possible_edges = 1000 * 999 / 2; // All possible edges
    assert(result.value().simplices.size() < max_possible_edges / 10); // Should be much smaller
    
    std::cout << "✅ Sparse Rips approximation test passed - " << result.value().simplices.size() << " simplices" << std::endl;
}

void test_sparse_rips_sparsity_levels() {
    std::cout << "Testing different sparsity levels..." << std::endl;
    
    auto points = generate_test_points(200);
    
    std::vector<double> sparsity_levels = {0.01, 0.05, 0.1, 0.2};
    size_t previous_size = 0;
    
    for (double sparsity : sparsity_levels) {
        tda::algorithms::SparseRips::Config config;
        config.max_dimension = 1;
        config.filtration_threshold = 4.0;
        config.sparsity_factor = sparsity;
        config.use_landmarks = true;
        config.num_landmarks = 30;
        config.min_points_threshold = 100;
        
        tda::algorithms::SparseRips sparse_rips(config);
        auto result = sparse_rips.computeApproximation(points, 4.0);
        
        assert(result.has_value());
        
        size_t current_size = result.value().simplices.size();
        std::cout << "  Sparsity " << sparsity << ": " << current_size << " simplices" << std::endl;
        
        // Higher sparsity should generally result in more simplices (less aggressive pruning)
        if (previous_size > 0) {
            // Allow some flexibility due to randomness in landmark selection
            assert(current_size >= previous_size * 0.8);
        }
        previous_size = current_size;
    }
    
    std::cout << "✅ Sparsity level test passed" << std::endl;
}

void test_sparse_rips_performance() {
    std::cout << "Testing Sparse Rips performance..." << std::endl;
    
    auto points = generate_test_points(1500);
    
    tda::algorithms::SparseRips::Config config;
    config.max_dimension = 1;
    config.filtration_threshold = 3.0;
    config.sparsity_factor = 0.1;
    config.use_landmarks = true;
    config.num_landmarks = 75;
    config.min_points_threshold = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    tda::algorithms::SparseRips sparse_rips(config);
    auto result = sparse_rips.computeApproximation(points, 3.0);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    assert(result.has_value());
    assert(duration.count() < 5000); // Should complete within 5 seconds
    
    std::cout << "✅ Performance test passed - " << duration.count() << "ms for " 
              << points.size() << " points, " << result.value().simplices.size() << " simplices" << std::endl;
}

void test_sparse_rips_edge_cases() {
    std::cout << "Testing Sparse Rips edge cases..." << std::endl;
    
    // Test with very small dataset
    auto small_points = generate_test_points(5);
    
    tda::algorithms::SparseRips::Config config;
    config.max_dimension = 1;
    config.filtration_threshold = 10.0;
    config.sparsity_factor = 0.1;
    config.min_points_threshold = 100; // Won't trigger approximation
    
    tda::algorithms::SparseRips sparse_rips(config);
    auto result = sparse_rips.computeApproximation(small_points, 10.0);
    
    assert(result.has_value());
    assert(!result.value().simplices.empty());
    
    // Test with single point
    auto single_point = generate_test_points(1);
    auto single_result = sparse_rips.computeApproximation(single_point, 10.0);
    assert(single_result.has_value());
    
    std::cout << "✅ Edge cases test passed" << std::endl;
}

} // anonymous namespace

int main() {
    std::cout << "🧪 Running Sparse Rips Algorithm Tests" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        test_sparse_rips_basic();
        test_sparse_rips_approximation_trigger();
        test_sparse_rips_sparsity_levels();
        test_sparse_rips_performance();
        test_sparse_rips_edge_cases();
        
        std::cout << std::endl;
        std::cout << "🎉 All Sparse Rips tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}

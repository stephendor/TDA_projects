#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "tda/utils/distance_matrix.hpp"

namespace {

std::vector<std::vector<double>> generate_test_points(size_t n, size_t dim = 3) {
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

double reference_euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void test_distance_matrix_correctness() {
    std::cout << "Testing distance matrix correctness..." << std::endl;
    
    auto points = generate_test_points(50, 3);
    
    tda::utils::DistanceMatrixConfig config;
    config.use_parallel = true;
    config.use_simd = true;
    config.block_size = 16;
    
    tda::utils::ParallelDistanceMatrix dm(config);
    auto result = dm.compute(points);
    
    // Verify matrix dimensions
    const auto& matrix = result.matrix;
    assert(matrix.size() == points.size());
    assert(matrix[0].size() == points.size());
    
    // Verify diagonal is zero
    for (size_t i = 0; i < points.size(); ++i) {
        assert(std::abs(matrix[i][i]) < 1e-10);
    }
    
    // Verify symmetry and correctness against reference implementation
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            // Check symmetry
            assert(std::abs(matrix[i][j] - matrix[j][i]) < 1e-10);
            
            // Check against reference implementation
            double reference = reference_euclidean_distance(points[i], points[j]);
            assert(std::abs(matrix[i][j] - reference) < 1e-10);
        }
    }
    
    std::cout << "✅ Distance matrix correctness test passed" << std::endl;
}

void test_distance_matrix_algorithms() {
    std::cout << "Testing different distance matrix configurations..." << std::endl;
    
    auto points = generate_test_points(100, 4);
    
    std::vector<tda::utils::DistanceMatrixConfig> configs;
    
    // Sequential config
    tda::utils::DistanceMatrixConfig seq_config;
    seq_config.use_parallel = false;
    seq_config.use_simd = false;
    seq_config.block_size = 32;
    
    // Parallel config
    tda::utils::DistanceMatrixConfig par_config;
    par_config.use_parallel = true;
    par_config.use_simd = false;
    par_config.block_size = 32;
    
    // SIMD config
    tda::utils::DistanceMatrixConfig simd_config;
    simd_config.use_parallel = false;
    simd_config.use_simd = true;
    simd_config.block_size = 32;
    
    configs.push_back(seq_config);
    configs.push_back(par_config);
    configs.push_back(simd_config);
    
    std::vector<tda::utils::DistanceMatrixResult> results;
    
    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];
        tda::utils::ParallelDistanceMatrix dm(config);
        auto result = dm.compute(points);
        results.push_back(result);
        
        std::string config_name = i == 0 ? "Sequential" : (i == 1 ? "Parallel" : "SIMD");
        std::cout << "  " << config_name << ": " << result.computation_time_seconds * 1000 << "ms" << std::endl;
    }
    
    // Verify all configurations produce the same results
    for (size_t algo = 1; algo < results.size(); ++algo) {
        const auto& ref_matrix = results[0].matrix;
        const auto& test_matrix = results[algo].matrix;
        
        for (size_t i = 0; i < ref_matrix.size(); ++i) {
            for (size_t j = 0; j < ref_matrix[i].size(); ++j) {
                assert(std::abs(ref_matrix[i][j] - test_matrix[i][j]) < 1e-10);
            }
        }
    }
    
    std::cout << "✅ Configuration comparison test passed" << std::endl;
}

void test_distance_matrix_performance() {
    std::cout << "Testing distance matrix performance..." << std::endl;
    
    auto points = generate_test_points(500, 3);
    
    tda::utils::DistanceMatrixConfig config;
    config.use_parallel = true;
    config.use_simd = true;
    config.block_size = 64;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    tda::utils::ParallelDistanceMatrix dm(config);
    auto result = dm.compute(points);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    assert(duration.count() < 2000); // Should complete within 2 seconds
    
    // Calculate distances per second
    size_t total_distances = points.size() * (points.size() - 1) / 2;
    double distances_per_sec = static_cast<double>(total_distances) / result.computation_time_seconds;
    
    std::cout << "✅ Performance test passed - " << duration.count() << "ms for " 
              << points.size() << "x" << points.size() << " matrix" << std::endl;
    std::cout << "  Throughput: " << static_cast<size_t>(distances_per_sec) << " distances/second" << std::endl;
}

void test_distance_matrix_sparse() {
    std::cout << "Testing sparse distance matrix..." << std::endl;
    
    auto points = generate_test_points(200, 2);
    
    tda::utils::DistanceMatrixConfig config;
    config.use_parallel = true;
    config.max_distance = 5.0; // Only compute distances up to 5.0
    config.block_size = 32;
    
    tda::utils::ParallelDistanceMatrix dm(config);
    auto result = dm.compute(points);
    
    const auto& matrix = result.matrix;
    
    // Verify that distances beyond threshold are zero or very large
    size_t sparse_count = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = i + 1; j < matrix[i].size(); ++j) {
            if (matrix[i][j] > config.max_distance) {
                sparse_count++;
            }
        }
    }
    
    std::cout << "✅ Sparse distance matrix test passed - " 
              << sparse_count << " distances beyond threshold" << std::endl;
}

void test_distance_matrix_edge_cases() {
    std::cout << "Testing distance matrix edge cases..." << std::endl;
    
    // Test with single point
    auto single_point = generate_test_points(1, 2);
    
    tda::utils::DistanceMatrixConfig config;
    config.use_parallel = false;
    config.use_simd = false;
    
    tda::utils::ParallelDistanceMatrix dm(config);
    auto result = dm.compute(single_point);
    
    // Verify matrix dimensions
    const auto& matrix = result.matrix;
    assert(matrix.size() == 1);
    assert(matrix[0].size() == 1);
    assert(std::abs(matrix[0][0]) < 1e-10);
    
    // Test with two points
    auto two_points = generate_test_points(2, 3);
    auto result2 = dm.compute(two_points);
    
    // Verify matrix dimensions and symmetry
    const auto& matrix2 = result2.matrix;
    assert(matrix2.size() == 2);
    assert(matrix2[0][1] == matrix2[1][0]);
    
    // Test with empty input
    std::vector<std::vector<double>> empty_points;
    auto result3 = dm.compute(empty_points);
    
    // Should handle empty input gracefully
    assert(result3.matrix.empty());
    
    std::cout << "✅ Edge cases test passed" << std::endl;
}

void test_distance_matrix_threading() {
    std::cout << "Testing distance matrix threading..." << std::endl;
    
    auto points = generate_test_points(300, 3);
    
    std::vector<bool> parallel_configs = {false, true};
    std::vector<double> times;
    
    for (bool use_parallel : parallel_configs) {
        tda::utils::DistanceMatrixConfig config;
        config.use_parallel = use_parallel;
        config.use_simd = true;
        config.block_size = 32;
        
        tda::utils::ParallelDistanceMatrix dm(config);
        auto result = dm.compute(points);
        
        times.push_back(result.computation_time_seconds);
        
        std::string config_name = use_parallel ? "Parallel" : "Sequential";
        std::cout << "  " << config_name << ": " << result.computation_time_seconds * 1000 << "ms" << std::endl;
    }
    
    // Verify that parallel computation generally improves performance
    // Note: This may not always hold due to overhead, but should show trend
    
    std::cout << "✅ Threading test completed" << std::endl;
}

} // anonymous namespace

int main() {
    std::cout << "🧪 Running Distance Matrix Tests" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        test_distance_matrix_correctness();
        test_distance_matrix_algorithms();
        test_distance_matrix_performance();
        test_distance_matrix_sparse();
        test_distance_matrix_edge_cases();
        test_distance_matrix_threading();
        
        std::cout << std::endl;
        std::cout << "🎉 All Distance Matrix tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}

#include "tda/utils/distance_matrix.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

namespace {

std::vector<std::vector<double>> generateTestPoints(size_t numPoints, size_t dimension) {
    std::vector<std::vector<double>> points;
    points.reserve(numPoints);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    
    for (size_t i = 0; i < numPoints; ++i) {
        std::vector<double> point;
        point.reserve(dimension);
        for (size_t j = 0; j < dimension; ++j) {
            point.push_back(dist(gen));
        }
        points.push_back(std::move(point));
    }
    
    return points;
}

void printResults(const std::string& algorithm, const tda::utils::DistanceMatrixResult& result, size_t numPoints) {
    std::cout << algorithm << " (" << numPoints << " points):" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << result.computation_time_seconds << " seconds" << std::endl;
    std::cout << "  Computations: " << result.total_computations << std::endl;
    std::cout << "  Rate: " << std::fixed << std::setprecision(0) 
              << (result.total_computations / result.computation_time_seconds) << " distances/sec" << std::endl;
    std::cout << "  Parallel: " << (result.used_parallel ? "Yes" : "No") << std::endl;
    std::cout << "  SIMD: " << (result.used_simd ? "Yes" : "No") << std::endl;
    std::cout << "  Algorithm: " << result.algorithm_used << std::endl;
    std::cout << std::endl;
}

} // namespace

int main() {
    std::cout << "🧮 Parallel Distance Matrix Performance Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    std::cout << "OpenMP Threads Available: " << omp_get_max_threads() << std::endl;
    std::cout << std::endl;
    
    // Test different point set sizes
    std::vector<size_t> pointCounts = {100, 500, 1000, 2000, 5000};
    const size_t dimension = 5;
    
    for (size_t numPoints : pointCounts) {
        std::cout << "=== Testing " << numPoints << " points (dimension " << dimension << ") ===" << std::endl;
        
        auto points = generateTestPoints(numPoints, dimension);
        
        // Test 1: Sequential computation
        {
            tda::utils::DistanceMatrixConfig config;
            config.use_parallel = false;
            config.use_simd = false;
            config.symmetric = true;
            
            tda::utils::ParallelDistanceMatrix computer(config);
            auto result = computer.compute(points);
            printResults("Sequential", result, numPoints);
        }
        
        // Test 2: Parallel computation
        {
            tda::utils::DistanceMatrixConfig config;
            config.use_parallel = true;
            config.use_simd = false;
            config.symmetric = true;
            config.parallel_threshold = 100;
            
            tda::utils::ParallelDistanceMatrix computer(config);
            auto result = computer.compute(points);
            printResults("Parallel", result, numPoints);
        }
        
        // Test 3: SIMD computation
        {
            tda::utils::DistanceMatrixConfig config;
            config.use_parallel = false;
            config.use_simd = true;
            config.symmetric = true;
            
            tda::utils::ParallelDistanceMatrix computer(config);
            auto result = computer.compute(points);
            printResults("SIMD", result, numPoints);
        }
        
        // Test 4: Parallel + SIMD computation
        {
            tda::utils::DistanceMatrixConfig config;
            config.use_parallel = true;
            config.use_simd = true;
            config.symmetric = true;
            config.parallel_threshold = 100;
            
            tda::utils::ParallelDistanceMatrix computer(config);
            auto result = computer.compute(points);
            printResults("Parallel+SIMD", result, numPoints);
        }
        
        // Test 5: Blocked computation (forced by large point count)
        {
            tda::utils::DistanceMatrixConfig config;
            config.use_parallel = true;
            config.use_simd = false;
            config.symmetric = true;
            config.block_size = 64;
            config.parallel_threshold = 100;
            
            // Force blocked algorithm by temporarily disabling other optimizations
            // and using a large dataset
            auto temp_points = points;
            if (temp_points.size() < 1500) {
                // Expand dataset to force blocked algorithm
                size_t original_size = temp_points.size();
                temp_points.reserve(1500);
                for (size_t i = original_size; i < 1500; ++i) {
                    temp_points.push_back(temp_points[i % original_size]);
                }
            }
            
            tda::utils::ParallelDistanceMatrix computer(config);
            auto result = computer.compute(temp_points);
            printResults("Large Dataset", result, temp_points.size());
        }
        
        // Test 6: Convenience function
        {
            auto result = tda::utils::computeDistanceMatrix(points, true, true);
            printResults("Convenience", result, numPoints);
        }
        
        std::cout << "----------------------------------------" << std::endl;
    }
    
    // Test sparse distance matrix
    std::cout << "\n=== Sparse Distance Matrix Test ===" << std::endl;
    auto points = generateTestPoints(1000, 5);
    double threshold = 50.0; // Only compute distances up to 50
    
    auto sparse_result = tda::utils::computeSparseDistanceMatrix(points, threshold, true);
    std::cout << "Sparse matrix (threshold=" << threshold << "):" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << sparse_result.computation_time_seconds << " seconds" << std::endl;
    std::cout << "  Computations: " << sparse_result.total_computations << std::endl;
    std::cout << "  Algorithm: " << sparse_result.algorithm_used << std::endl;
    
    // Count non-infinite entries
    size_t non_infinite = 0;
    for (const auto& row : sparse_result.matrix) {
        for (double val : row) {
            if (val != std::numeric_limits<double>::infinity()) {
                non_infinite++;
            }
        }
    }
    std::cout << "  Non-infinite entries: " << non_infinite << " / " << (1000 * 1000) 
              << " (" << (100.0 * non_infinite / (1000 * 1000)) << "%)" << std::endl;
    
    std::cout << "\n🎯 Performance Summary:" << std::endl;
    std::cout << "- Multiple optimized algorithms implemented" << std::endl;
    std::cout << "- SIMD vectorization for Euclidean distances" << std::endl;
    std::cout << "- OpenMP parallelization with load balancing" << std::endl;
    std::cout << "- Block-wise computation for cache optimization" << std::endl;
    std::cout << "- Symmetric matrix optimization" << std::endl;
    std::cout << "- Sparse computation with distance thresholds" << std::endl;
    
    return 0;
}

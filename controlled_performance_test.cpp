#include "tda/algorithms/vietoris_rips.hpp"
#include "tda/algorithms/sparse_rips.hpp"
#include "tda/algorithms/witness_complex.hpp"
#include "tda/algorithms/adaptive_sampling.hpp"
#include "tda/utils/simd_utils.hpp"
#include "tda/spatial/spatial_index.hpp"
#include "tda/core/performance_profiler.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <functional>

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

double timeFunction(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Return milliseconds
}

void testSIMDPerformance() {
    std::cout << "\n🚀 SIMD Performance Test" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Test vectorized distance computation vs scalar
    const size_t numPoints = 10000;
    const size_t dimension = 10;
    
    auto points1 = generateTestPoints(numPoints, dimension);
    auto points2 = generateTestPoints(numPoints, dimension);
    
    std::vector<double> distances_scalar(numPoints);
    std::vector<double> distances_simd(numPoints);
    
    // Scalar implementation
    auto scalar_time = timeFunction([&]() {
        for (size_t i = 0; i < numPoints; ++i) {
            double sum = 0.0;
            for (size_t d = 0; d < dimension; ++d) {
                double diff = points1[i][d] - points2[i][d];
                sum += diff * diff;
            }
            distances_scalar[i] = std::sqrt(sum);
        }
    });
    
    // SIMD implementation
    auto simd_time = timeFunction([&]() {
        for (size_t i = 0; i < numPoints; ++i) {
            distances_simd[i] = tda::utils::SIMDUtils::vectorizedEuclideanDistance(
                points1[i].data(), points2[i].data(), dimension);
        }
    });
    
    // Calculate speedup
    double speedup = scalar_time / simd_time;
    
    std::cout << "Distance computation (" << numPoints << " x " << dimension << "D):" << std::endl;
    std::cout << "  Scalar time: " << std::fixed << std::setprecision(2) << scalar_time << " ms" << std::endl;
    std::cout << "  SIMD time:   " << std::fixed << std::setprecision(2) << simd_time << " ms" << std::endl;
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    // Verify results are approximately equal
    double max_error = 0.0;
    for (size_t i = 0; i < std::min(size_t(100), numPoints); ++i) {
        double error = std::abs(distances_scalar[i] - distances_simd[i]);
        max_error = std::max(max_error, error);
    }
    std::cout << "  Max error:   " << std::scientific << std::setprecision(2) << max_error << std::endl;
}

void testSpatialIndexPerformance() {
    std::cout << "\n🔍 Spatial Index Performance Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    const size_t numPoints = 50000;
    const size_t dimension = 5;
    const size_t numQueries = 1000;
    const size_t k = 10;
    
    auto points = generateTestPoints(numPoints, dimension);
    auto queries = generateTestPoints(numQueries, dimension);
    
    // Test KDTree performance
    auto kdtree = std::make_unique<tda::spatial::KDTree>();
    
    auto build_time = timeFunction([&]() {
        kdtree->build(points);
    });
    
    auto query_time = timeFunction([&]() {
        for (const auto& query : queries) {
            auto results = kdtree->kNearestNeighbors(query, k);
        }
    });
    
    std::cout << "KDTree (" << numPoints << " points, " << dimension << "D):" << std::endl;
    std::cout << "  Build time:      " << std::fixed << std::setprecision(2) << build_time << " ms" << std::endl;
    std::cout << "  Query time:      " << std::fixed << std::setprecision(2) << query_time << " ms" << std::endl;
    std::cout << "  Queries/second:  " << std::fixed << std::setprecision(0) << (numQueries / (query_time / 1000.0)) << std::endl;
}

void testApproximationAlgorithms() {
    std::cout << "\n📊 Approximation Algorithms Performance Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    const size_t numPoints = 20000;
    const size_t dimension = 5;
    const double threshold = 50.0;
    
    auto points = generateTestPoints(numPoints, dimension);
    
    // Test Sparse Rips
    {
        tda::algorithms::SparseRips sparse_rips;
        tda::algorithms::SparseRips::Config config;
        config.sparsity_factor = 0.05;     // 5% of edges for reasonable performance
        config.max_edges = 50000;          // Reasonable limit for testing
        config.strategy = "density";       // Use simpler strategy
        config.min_points_threshold = 1000; // Enable approximation for test datasets
        config.use_landmarks = false;      // Disable landmarks for cleaner testing
        
        auto sparse_time = timeFunction([&]() {
            auto result = sparse_rips.computeApproximation(points, threshold, config);
            if (result.has_value()) {
                std::cout << "  Sparse Rips: " << result.value().edges_retained 
                          << " edges retained" << std::endl;
            }
        });
        
        std::cout << "Sparse Rips (" << numPoints << " points, sparsity=" << (config.sparsity_factor * 100) << "%):" << std::endl;
        std::cout << "  Computation time: " << std::fixed << std::setprecision(2) << sparse_time << " ms" << std::endl;
    }
    
    // Test Witness Complex
    {
        tda::algorithms::WitnessComplex witness_complex;
        tda::algorithms::WitnessComplex::WitnessConfig config;
        config.num_landmarks = 500;
        config.max_dimension = 2;
        
        auto witness_time = timeFunction([&]() {
            auto result = witness_complex.computeWitnessComplex(points, config);
            if (result.has_value()) {
                std::cout << "  Witness Complex: " << result.value().simplices.size() 
                          << " simplices generated" << std::endl;
            }
        });
        
        std::cout << "Witness Complex (" << numPoints << " points, " << config.num_landmarks << " landmarks):" << std::endl;
        std::cout << "  Computation time: " << std::fixed << std::setprecision(2) << witness_time << " ms" << std::endl;
    }
    
    // Test Adaptive Sampling
    {
        tda::algorithms::AdaptiveSampling adaptive_sampling;
        tda::algorithms::AdaptiveSampling::SamplingConfig config;
        config.strategy = "hybrid";
        config.max_samples = 5000;
        
        auto sampling_time = timeFunction([&]() {
            auto result = adaptive_sampling.adaptiveSample(points, config);
            if (result.has_value()) {
                std::cout << "  Adaptive Sampling: " << result.value().selected_indices.size() 
                          << " points selected, quality=" << std::fixed << std::setprecision(3) 
                          << result.value().achieved_quality << std::endl;
            }
        });
        
        std::cout << "Adaptive Sampling (" << numPoints << " points, strategy=" << config.strategy << "):" << std::endl;
        std::cout << "  Computation time: " << std::fixed << std::setprecision(2) << sampling_time << " ms" << std::endl;
    }
}

void testVietorisRipsScaling() {
    std::cout << "\n📈 Vietoris-Rips Scaling Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    std::vector<size_t> pointCounts = {1000, 5000, 10000, 20000};
    const size_t dimension = 5;
    const double threshold = 50.0;
    
    for (size_t numPoints : pointCounts) {
        auto points = generateTestPoints(numPoints, dimension);
        
        tda::algorithms::VietorisRips vr;
        
        auto total_time = timeFunction([&]() {
            auto init_result = vr.initialize(points, threshold, 2, 2);
            if (init_result.has_value()) {
                auto complex_result = vr.computeComplex();
                if (complex_result.has_value()) {
                    auto persistence_result = vr.computePersistence();
                }
            }
        });
        
        auto stats_result = vr.getStatistics();
        size_t num_simplices = 0;
        if (stats_result.has_value()) {
            num_simplices = stats_result.value().num_simplices;
        }
        
        std::cout << "VR(" << std::setw(5) << numPoints << " pts): " 
                  << std::setw(8) << std::fixed << std::setprecision(1) << total_time << " ms, "
                  << std::setw(6) << num_simplices << " simplices" << std::endl;
    }
}

} // namespace

int main() {
    std::cout << "🔬 TDA Platform Controlled Performance Tests" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        testSIMDPerformance();
        testSpatialIndexPerformance();
        testApproximationAlgorithms();
        testVietorisRipsScaling();
        
        std::cout << "\n✅ All performance tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during performance testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

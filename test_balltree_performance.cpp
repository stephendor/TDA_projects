#include "tda/spatial/spatial_index.hpp"
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

double timeFunction(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Return milliseconds
}

} // namespace

int main() {
    std::cout << "🌳 Ball Tree Performance Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    std::cout << "OpenMP Threads Available: " << omp_get_max_threads() << std::endl;
    
    // Test with different sizes to see parallel scaling
    std::vector<size_t> pointCounts = {1000, 5000, 10000, 20000, 50000};
    const size_t dimension = 5;
    const size_t numQueries = 100;
    const size_t k = 10;
    
    for (size_t numPoints : pointCounts) {
        std::cout << "\n=== Testing " << numPoints << " points ===" << std::endl;
        
        auto points = generateTestPoints(numPoints, dimension);
        auto queries = generateTestPoints(numQueries, dimension);
        
        // Test Ball Tree performance
        auto ballTree = std::make_unique<tda::spatial::BallTree>();
        
        // Test single-threaded vs multi-threaded build
        double build_time_single, build_time_parallel;
        
        // Single-threaded build
        omp_set_num_threads(1);
        build_time_single = timeFunction([&]() {
            ballTree->build(points);
        });
        
        // Multi-threaded build
        omp_set_num_threads(omp_get_max_threads());
        ballTree->clear();
        build_time_parallel = timeFunction([&]() {
            ballTree->build(points);
        });
        
        // Test query performance
        auto query_time = timeFunction([&]() {
            for (const auto& query : queries) {
                auto results = ballTree->kNearestNeighbors(query, k);
            }
        });
        
        // Calculate speedup
        double build_speedup = build_time_single / build_time_parallel;
        
        std::cout << "Build time (1 thread):  " << std::fixed << std::setprecision(2) 
                  << build_time_single << " ms" << std::endl;
        std::cout << "Build time (parallel):  " << std::fixed << std::setprecision(2) 
                  << build_time_parallel << " ms" << std::endl;
        std::cout << "Build speedup:          " << std::fixed << std::setprecision(2) 
                  << build_speedup << "x" << std::endl;
        std::cout << "Query time:             " << std::fixed << std::setprecision(2) 
                  << query_time << " ms" << std::endl;
        std::cout << "Queries/second:         " << std::fixed << std::setprecision(0) 
                  << (numQueries / (query_time / 1000.0)) << std::endl;
        
        // Performance analysis
        if (build_speedup > 1.2) {
            std::cout << "✅ Good parallel speedup achieved" << std::endl;
        } else if (build_speedup > 1.0) {
            std::cout << "⚠️  Modest parallel speedup" << std::endl;
        } else {
            std::cout << "❌ No parallel benefit (overhead)" << std::endl;
        }
        
        if (query_time / numQueries < 1.0) {
            std::cout << "✅ Excellent query performance" << std::endl;
        } else {
            std::cout << "⚠️  Query performance could be better" << std::endl;
        }
    }
    
    std::cout << "\n🎯 Ball Tree Optimization Summary:" << std::endl;
    std::cout << "- Parallel tree construction implemented" << std::endl;
    std::cout << "- SIMD-optimized distance calculations" << std::endl;
    std::cout << "- Parallel centroid and radius computation" << std::endl;
    std::cout << "- OpenMP reduction for thread-safe aggregation" << std::endl;
    
    return 0;
}

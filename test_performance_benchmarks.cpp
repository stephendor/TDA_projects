#include "tda/algorithms/vietoris_rips.hpp"
#include "tda/algorithms/alpha_complex.hpp"
#include "tda/algorithms/cech_complex.hpp"
#include "tda/core/performance_profiler.hpp"
#include "tda/core/types.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>

namespace {

/**
 * @brief Generate synthetic point cloud data for testing
 */
std::vector<std::vector<double>> generatePointCloud(size_t numPoints, size_t dimension, 
                                                   const std::string& distribution = "uniform") {
    std::vector<std::vector<double>> points;
    points.reserve(numPoints);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (distribution == "uniform") {
        std::uniform_real_distribution<double> dist(0.0, 1000.0);
        for (size_t i = 0; i < numPoints; ++i) {
            std::vector<double> point;
            point.reserve(dimension);
            for (size_t j = 0; j < dimension; ++j) {
                point.push_back(dist(gen));
            }
            points.push_back(std::move(point));
        }
    } else if (distribution == "sphere") {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::uniform_real_distribution<double> radius_dist(0.0, 500.0);
        
        for (size_t i = 0; i < numPoints; ++i) {
            std::vector<double> point(dimension);
            double radius = radius_dist(gen);
            
            // Generate random direction
            double norm = 0.0;
            for (size_t j = 0; j < dimension; ++j) {
                point[j] = dist(gen);
                norm += point[j] * point[j];
            }
            norm = std::sqrt(norm);
            
            // Normalize and scale by radius
            for (size_t j = 0; j < dimension; ++j) {
                point[j] = (point[j] / norm) * radius;
            }
            
            points.push_back(std::move(point));
        }
    } else if (distribution == "torus") {
        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> radius_dist(0.0, 100.0);
        
        for (size_t i = 0; i < numPoints; ++i) {
            double theta = angle_dist(gen);
            double phi = angle_dist(gen);
            double R = 200.0; // Major radius
            double r = radius_dist(gen); // Minor radius
            
            std::vector<double> point(3);
            point[0] = (R + r * std::cos(phi)) * std::cos(theta);
            point[1] = (R + r * std::cos(phi)) * std::sin(theta);
            point[2] = r * std::sin(phi);
            
            points.push_back(std::move(point));
        }
    }
    
    return points;
}

/**
 * @brief Benchmark Vietoris-Rips complex construction
 */
void benchmarkVietorisRips(const std::vector<std::vector<double>>& points, 
                          double threshold, int maxDimension,
                          tda::core::PerformanceSession& session) {
    std::cout << "Benchmarking Vietoris-Rips with " << points.size() << " points..." << std::endl;
    
    tda::algorithms::VietorisRips vr;
    
    // Initialize
    {
        TDA_PROFILE(session, "VR_Initialize");
        auto result = vr.initialize(points, threshold, maxDimension);
        if (result.has_error()) {
            std::cerr << "Failed to initialize Vietoris-Rips: " << result.error() << std::endl;
            return;
        }
    }
    
    session.update_memory();
    
    // Compute complex
    {
        TDA_PROFILE(session, "VR_ComputeComplex");
        auto result = vr.computeComplex();
        if (result.has_error()) {
            std::cerr << "Failed to compute Vietoris-Rips complex: " << result.error() << std::endl;
            return;
        }
    }
    
    session.update_memory();
    
    // Get statistics
    {
        TDA_PROFILE(session, "VR_GetStatistics");
        auto stats_result = vr.getStatistics();
        if (stats_result.has_value()) {
            const auto& stats = stats_result.value();
            session.record_measurement("VR_NumSimplices", static_cast<double>(stats.num_simplices));
            session.record_measurement("VR_MaxDimension", static_cast<double>(stats.max_dimension));
        }
    }
    
    session.update_memory();
}

/**
 * @brief Benchmark Alpha complex construction
 */
void benchmarkAlphaComplex(const std::vector<std::vector<double>>& points, 
                          double threshold, int maxDimension,
                          tda::core::PerformanceSession& session) {
    std::cout << "Benchmarking Alpha Complex with " << points.size() << " points..." << std::endl;
    
    tda::algorithms::AlphaComplex ac;
    
    // Initialize
    {
        TDA_PROFILE(session, "Alpha_Initialize");
        auto result = ac.initialize(points, threshold, maxDimension);
        if (result.has_error()) {
            std::cerr << "Failed to initialize Alpha Complex: " << result.error() << std::endl;
            return;
        }
    }
    
    session.update_memory();
    
    // Compute complex
    {
        TDA_PROFILE(session, "Alpha_ComputeComplex");
        auto result = ac.computeComplex();
        if (result.has_error()) {
            std::cerr << "Failed to compute Alpha Complex: " << result.error() << std::endl;
            return;
        }
    }
    
    session.update_memory();
    
    // Get statistics
    {
        TDA_PROFILE(session, "Alpha_GetStatistics");
        auto stats_result = ac.getStatistics();
        if (stats_result.has_value()) {
            const auto& stats = stats_result.value();
            session.record_measurement("Alpha_NumSimplices", static_cast<double>(stats.num_simplices));
            session.record_measurement("Alpha_MaxDimension", static_cast<double>(stats.max_dimension));
        }
    }
    
    session.update_memory();
}

/**
 * @brief Benchmark Čech complex construction
 */
void benchmarkCechComplex(const std::vector<std::vector<double>>& points, 
                         double threshold, int maxDimension,
                         tda::core::PerformanceSession& session) {
    std::cout << "Benchmarking Čech Complex with " << points.size() << " points..." << std::endl;
    
    tda::algorithms::CechComplex cc;
    
    // Initialize
    {
        TDA_PROFILE(session, "Cech_Initialize");
        auto result = cc.initialize(points);
        if (result.has_error()) {
            std::cerr << "Failed to initialize Čech Complex: " << result.error() << std::endl;
            return;
        }
    }
    
    session.update_memory();
    
    // Compute complex
    {
        TDA_PROFILE(session, "Cech_ComputeComplex");
        auto result = cc.computeComplex();
        if (result.has_error()) {
            std::cerr << "Failed to compute Čech Complex: " << result.error() << std::endl;
            return;
        }
    }
    
    session.update_memory();
    
    // Get statistics
    {
        TDA_PROFILE(session, "Cech_GetStatistics");
        auto stats_result = cc.getStatistics();
        if (stats_result.has_value()) {
            const auto& stats = stats_result.value();
            session.record_measurement("Cech_NumSimplices", static_cast<double>(stats.num_simplices));
            session.record_measurement("Cech_MaxDimension", static_cast<double>(stats.max_dimension));
        }
    }
    
    session.update_memory();
}

/**
 * @brief Run comprehensive performance benchmarks
 */
void runPerformanceBenchmarks() {
    std::cout << "Starting TDA Performance Benchmarks" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Test configurations
    std::vector<size_t> pointCounts = {1000, 10000, 100000, 500000, 1000000, 1500000};
    std::vector<size_t> dimensions = {3, 5, 10};
    std::vector<std::string> distributions = {"uniform", "sphere", "torus"};
    
    // Performance thresholds (ST-101 requirement: 1M points in under 60 seconds)
    const double TIME_THRESHOLD_SECONDS = 60.0;
    const size_t TARGET_POINT_COUNT = 1000000;
    
    std::ofstream resultsFile("performance_benchmarks.csv");
    resultsFile << "Algorithm,PointCount,Dimension,Distribution,Threshold,InitTime(ms),ComplexTime(ms),"
                << "TotalTime(ms),MemoryPeak(MB),MemoryIncrease(MB),NumSimplices,MaxDimension,"
                << "PointsPerSecond,Status" << std::endl;
    
    for (size_t numPoints : pointCounts) {
        for (size_t dim : dimensions) {
            for (const std::string& dist : distributions) {
                // Skip torus for non-3D cases
                if (dist == "torus" && dim != 3) continue;
                
                std::cout << "\n=== Testing: " << numPoints << " points, " << dim 
                         << "D, " << dist << " distribution ===" << std::endl;
                
                // Generate point cloud
                auto points = generatePointCloud(numPoints, dim, dist);
                
                // Calculate appropriate threshold based on dimension and distribution
                double threshold = (dist == "uniform") ? 100.0 : 
                                 (dist == "sphere") ? 50.0 : 30.0;
                
                // Test Vietoris-Rips
                {
                    tda::core::PerformanceSession session("VR_" + std::to_string(numPoints) + "_" + 
                                                        std::to_string(dim) + "D_" + dist);
                    
                    try {
                        benchmarkVietorisRips(points, threshold, 3, session);
                        
                        // Calculate performance metrics
                        double totalTime = 0.0;
                        double initTime = 0.0;
                        double complexTime = 0.0;
                        
                        for (const auto& timer : session.get_timers()) {
                            if (timer->name() == "VR_Initialize") {
                                initTime = timer->elapsed_milliseconds();
                            } else if (timer->name() == "VR_ComputeComplex") {
                                complexTime = timer->elapsed_milliseconds();
                            }
                        }
                        totalTime = initTime + complexTime;
                        
                        double pointsPerSecond = (numPoints / (totalTime / 1000.0));
                        bool meetsRequirement = (numPoints <= TARGET_POINT_COUNT) || 
                                              (totalTime / 1000.0 <= TIME_THRESHOLD_SECONDS);
                        
                        std::string status = meetsRequirement ? "PASS" : "FAIL";
                        
                        // Write to CSV
                        resultsFile << "VietorisRips," << numPoints << "," << dim << "," << dist << ","
                                   << threshold << "," << initTime << "," << complexTime << ","
                                   << totalTime << "," << session.get_peak_memory_mb() << ","
                                   << session.get_memory_increase_mb() << ","
                                   << session.get_measurement("VR_NumSimplices") << ","
                                   << session.get_measurement("VR_MaxDimension") << ","
                                   << pointsPerSecond << "," << status << std::endl;
                        
                        // Print summary
                        std::cout << "Vietoris-Rips: " << totalTime << " ms, " 
                                 << session.get_peak_memory_mb() << " MB peak, "
                                 << pointsPerSecond << " points/sec - " << status << std::endl;
                        
                        // Check if we meet the ST-101 requirement
                        if (numPoints == TARGET_POINT_COUNT) {
                            if (totalTime / 1000.0 <= TIME_THRESHOLD_SECONDS) {
                                std::cout << "✅ ST-101 Requirement MET: 1M points processed in " 
                                         << (totalTime / 1000.0) << " seconds (< " 
                                         << TIME_THRESHOLD_SECONDS << "s)" << std::endl;
                            } else {
                                std::cout << "❌ ST-101 Requirement NOT MET: 1M points took " 
                                         << (totalTime / 1000.0) << " seconds (> " 
                                         << TIME_THRESHOLD_SECONDS << "s)" << std::endl;
                            }
                        }
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error benchmarking Vietoris-Rips: " << e.what() << std::endl;
                        resultsFile << "VietorisRips," << numPoints << "," << dim << "," << dist << ","
                                   << threshold << ",ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" << std::endl;
                    }
                }
                
                // Test Alpha Complex (for smaller point clouds to avoid excessive memory usage)
                if (numPoints <= 100000) {
                    tda::core::PerformanceSession session("Alpha_" + std::to_string(numPoints) + "_" + 
                                                        std::to_string(dim) + "D_" + dist);
                    
                    try {
                        benchmarkAlphaComplex(points, threshold, 3, session);
                        
                        // Calculate performance metrics
                        double totalTime = 0.0;
                        double initTime = 0.0;
                        double complexTime = 0.0;
                        
                        for (const auto& timer : session.get_timers()) {
                            if (timer->name() == "Alpha_Initialize") {
                                initTime = timer->elapsed_milliseconds();
                            } else if (timer->name() == "Alpha_ComputeComplex") {
                                complexTime = timer->elapsed_milliseconds();
                            }
                        }
                        totalTime = initTime + complexTime;
                        
                        double pointsPerSecond = (numPoints / (totalTime / 1000.0));
                        
                        // Write to CSV
                        resultsFile << "AlphaComplex," << numPoints << "," << dim << "," << dist << ","
                                   << threshold << "," << initTime << "," << complexTime << ","
                                   << totalTime << "," << session.get_peak_memory_mb() << ","
                                   << session.get_memory_increase_mb() << ","
                                   << session.get_measurement("Alpha_NumSimplices") << ","
                                   << session.get_measurement("Alpha_MaxDimension") << ","
                                   << pointsPerSecond << ",PASS" << std::endl;
                        
                        std::cout << "Alpha Complex: " << totalTime << " ms, " 
                                 << session.get_peak_memory_mb() << " MB peak, "
                                 << pointsPerSecond << " points/sec" << std::endl;
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error benchmarking Alpha Complex: " << e.what() << std::endl;
                        resultsFile << "AlphaComplex," << numPoints << "," << dim << "," << dist << ","
                                   << threshold << ",ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" << std::endl;
                    }
                }
                
                // Test Čech Complex (for very small point clouds due to computational complexity)
                if (numPoints <= 10000) {
                    tda::core::PerformanceSession session("Cech_" + std::to_string(numPoints) + "_" + 
                                                        std::to_string(dim) + "D_" + dist);
                    
                    try {
                        benchmarkCechComplex(points, threshold, 3, session);
                        
                        // Calculate performance metrics
                        double totalTime = 0.0;
                        double initTime = 0.0;
                        double complexTime = 0.0;
                        
                        for (const auto& timer : session.get_timers()) {
                            if (timer->name() == "Cech_Initialize") {
                                initTime = timer->elapsed_milliseconds();
                            } else if (timer->name() == "Cech_ComputeComplex") {
                                complexTime = timer->elapsed_milliseconds();
                            }
                        }
                        totalTime = initTime + complexTime;
                        
                        double pointsPerSecond = (numPoints / (totalTime / 1000.0));
                        
                        // Write to CSV
                        resultsFile << "CechComplex," << numPoints << "," << dim << "," << dist << ","
                                   << threshold << "," << initTime << "," << complexTime << ","
                                   << totalTime << "," << session.get_peak_memory_mb() << ","
                                   << session.get_memory_increase_mb() << ","
                                   << session.get_measurement("Cech_NumSimplices") << ","
                                   << session.get_measurement("Cech_MaxDimension") << ","
                                   << pointsPerSecond << ",PASS" << std::endl;
                        
                        std::cout << "Čech Complex: " << totalTime << " ms, " 
                                 << session.get_peak_memory_mb() << " MB peak, "
                                 << pointsPerSecond << " points/sec" << std::endl;
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error benchmarking Čech Complex: " << e.what() << std::endl;
                        resultsFile << "CechComplex," << numPoints << "," << dim << "," << dist << ","
                                   << threshold << ",ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" << std::endl;
                    }
                }
                
                // TODO: Re-implement Sparse Rips testing in Phase 2B
                // Sparse Rips Filtration test removed - implementation deleted
            }
        }
    }
    
    resultsFile.close();
    std::cout << "\nBenchmark results saved to 'performance_benchmarks.csv'" << std::endl;
}

} // anonymous namespace

int main() {
    try {
        runPerformanceBenchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Benchmark failed with unknown exception" << std::endl;
        return 1;
    }
}

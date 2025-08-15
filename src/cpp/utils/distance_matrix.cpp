#include "tda/utils/distance_matrix.hpp"
#include "tda/utils/simd_utils.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

namespace tda::utils {

ParallelDistanceMatrix::ParallelDistanceMatrix(const DistanceMatrixConfig& config) 
    : config_(config) {}

DistanceMatrixResult ParallelDistanceMatrix::compute(const PointContainer& points) {
    return compute(points, euclideanDistance);
}

DistanceMatrixResult ParallelDistanceMatrix::compute(const PointContainer& points, DistanceFunction distance_func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (points.empty()) {
        return DistanceMatrixResult{};
    }
    
    // Select optimal algorithm based on configuration and data characteristics
    std::string algorithm = selectAlgorithm(points);
    DistanceMatrixResult result;
    result.algorithm_used = algorithm;
    
    // Route to appropriate computation method
    if (algorithm == "simd" && config_.use_simd) {
        result = computeSIMD(points);
    } else if (algorithm == "parallel" && config_.use_parallel) {
        result = computeParallel(points, distance_func);
    } else if (algorithm == "blocked") {
        result = computeBlocked(points, distance_func);
    } else if (algorithm == "symmetric" && config_.symmetric) {
        result = computeSymmetricOptimized(points, distance_func);
    } else if (algorithm == "threshold" && config_.max_distance != std::numeric_limits<double>::infinity()) {
        result = computeWithThreshold(points, distance_func, config_.max_distance);
    } else {
        result = computeSequential(points, distance_func);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.computation_time_seconds = duration.count() / 1000000.0;
    result.algorithm_used = algorithm;
    
    return result;
}

DistanceMatrixResult ParallelDistanceMatrix::computeBetween(const PointContainer& points1, 
                                                           const PointContainer& points2,
                                                           DistanceFunction distance_func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (points1.empty() || points2.empty()) {
        return DistanceMatrixResult{};
    }
    
    if (!distance_func) {
        distance_func = euclideanDistance;
    }
    
    DistanceMatrixResult result;
    result.matrix.resize(points1.size(), std::vector<double>(points2.size()));
    result.total_computations = points1.size() * points2.size();
    
    if (points1.size() >= config_.parallel_threshold && config_.use_parallel && omp_get_max_threads() > 1) {
        // Parallel computation for cross-matrix
        result.used_parallel = true;
        
        #pragma omp parallel for collapse(2) schedule(dynamic, 32)
        for (size_t i = 0; i < points1.size(); ++i) {
            for (size_t j = 0; j < points2.size(); ++j) {
                result.matrix[i][j] = distance_func(points1[i], points2[j]);
            }
        }
    } else {
        // Sequential computation
        for (size_t i = 0; i < points1.size(); ++i) {
            for (size_t j = 0; j < points2.size(); ++j) {
                result.matrix[i][j] = distance_func(points1[i], points2[j]);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.computation_time_seconds = duration.count() / 1000000.0;
    
    return result;
}

DistanceMatrixResult ParallelDistanceMatrix::computeSequential(const PointContainer& points, DistanceFunction distance_func) {
    DistanceMatrixResult result;
    const size_t n = points.size();
    result.matrix.resize(n, std::vector<double>(n));
    
    if (config_.symmetric) {
        // Compute only upper triangle for symmetric matrices
        result.total_computations = n * (n - 1) / 2;
        for (size_t i = 0; i < n; ++i) {
            result.matrix[i][i] = 0.0; // Diagonal is zero for distance matrices
            for (size_t j = i + 1; j < n; ++j) {
                double dist = distance_func(points[i], points[j]);
                result.matrix[i][j] = dist;
                result.matrix[j][i] = dist; // Symmetric
            }
        }
    } else {
        // Compute full matrix
        result.total_computations = n * n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result.matrix[i][j] = distance_func(points[i], points[j]);
            }
        }
    }
    
    return result;
}

DistanceMatrixResult ParallelDistanceMatrix::computeParallel(const PointContainer& points, DistanceFunction distance_func) {
    DistanceMatrixResult result;
    const size_t n = points.size();
    result.matrix.resize(n, std::vector<double>(n));
    result.used_parallel = true;
    
    if (config_.symmetric) {
        // Parallel symmetric computation
        result.total_computations = n * (n - 1) / 2;
        
        #pragma omp parallel for schedule(dynamic, 32)
        for (size_t i = 0; i < n; ++i) {
            result.matrix[i][i] = 0.0;
            for (size_t j = i + 1; j < n; ++j) {
                double dist = distance_func(points[i], points[j]);
                result.matrix[i][j] = dist;
                result.matrix[j][i] = dist;
            }
        }
    } else {
        // Parallel full matrix computation
        result.total_computations = n * n;
        
        #pragma omp parallel for collapse(2) schedule(dynamic, 32)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result.matrix[i][j] = distance_func(points[i], points[j]);
            }
        }
    }
    
    return result;
}

DistanceMatrixResult ParallelDistanceMatrix::computeSIMD(const PointContainer& points) {
    DistanceMatrixResult result;
    const size_t n = points.size();
    result.matrix.resize(n, std::vector<double>(n));
    result.used_simd = true;
    
    // SIMD optimization works best with Euclidean distance
    if (points.empty() || points[0].empty()) {
        return result;
    }
    
    const size_t dimension = points[0].size();
    
    if (config_.symmetric) {
        result.total_computations = n * (n - 1) / 2;
        
        // Use SIMD-optimized distance computation
        #pragma omp parallel for schedule(dynamic, 16) if(config_.use_parallel && n >= config_.parallel_threshold)
        for (size_t i = 0; i < n; ++i) {
            result.matrix[i][i] = 0.0;
            for (size_t j = i + 1; j < n; ++j) {
                double dist = SIMDUtils::vectorizedEuclideanDistance(
                    points[i].data(), points[j].data(), dimension);
                result.matrix[i][j] = dist;
                result.matrix[j][i] = dist;
            }
        }
        
        if (config_.use_parallel && n >= config_.parallel_threshold) {
            result.used_parallel = true;
        }
    } else {
        result.total_computations = n * n;
        
        #pragma omp parallel for collapse(2) schedule(dynamic, 16) if(config_.use_parallel && n >= config_.parallel_threshold)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result.matrix[i][j] = SIMDUtils::vectorizedEuclideanDistance(
                    points[i].data(), points[j].data(), dimension);
            }
        }
        
        if (config_.use_parallel && n >= config_.parallel_threshold) {
            result.used_parallel = true;
        }
    }
    
    return result;
}

DistanceMatrixResult ParallelDistanceMatrix::computeBlocked(const PointContainer& points, DistanceFunction distance_func) {
    DistanceMatrixResult result;
    const size_t n = points.size();
    result.matrix.resize(n, std::vector<double>(n));
    result.cache_misses = estimateCacheMisses(n, config_.block_size);
    
    // Block-wise computation for better cache locality
    const size_t block_size = config_.block_size;
    result.total_computations = config_.symmetric ? n * (n - 1) / 2 : n * n;
    
    if (config_.symmetric) {
        for (size_t bi = 0; bi < n; bi += block_size) {
            for (size_t bj = bi; bj < n; bj += block_size) {
                // Process block (bi,bj)
                size_t end_i = std::min(bi + block_size, n);
                size_t end_j = std::min(bj + block_size, n);
                
                // Handle diagonal blocks specially for symmetric matrices
                if (bi == bj) {
                    // Diagonal block - compute only upper triangle
                    for (size_t i = bi; i < end_i; ++i) {
                        for (size_t j = std::max(bj, i + 1); j < end_j; ++j) {
                            double dist = distance_func(points[i], points[j]);
                            result.matrix[i][j] = dist;
                            result.matrix[j][i] = dist;
                        }
                        if (i < end_j) {
                            result.matrix[i][i] = 0.0;
                        }
                    }
                } else {
                    // Off-diagonal block - compute full block
                    #pragma omp parallel for collapse(2) if(config_.use_parallel && (end_i - bi) * (end_j - bj) > 256)
                    for (size_t i = bi; i < end_i; ++i) {
                        for (size_t j = bj; j < end_j; ++j) {
                            double dist = distance_func(points[i], points[j]);
                            result.matrix[i][j] = dist;
                            result.matrix[j][i] = dist;
                        }
                    }
                }
            }
        }
    } else {
        for (size_t bi = 0; bi < n; bi += block_size) {
            for (size_t bj = 0; bj < n; bj += block_size) {
                size_t end_i = std::min(bi + block_size, n);
                size_t end_j = std::min(bj + block_size, n);
                
                #pragma omp parallel for collapse(2) if(config_.use_parallel && (end_i - bi) * (end_j - bj) > 256)
                for (size_t i = bi; i < end_i; ++i) {
                    for (size_t j = bj; j < end_j; ++j) {
                        result.matrix[i][j] = distance_func(points[i], points[j]);
                    }
                }
            }
        }
    }
    
    if (config_.use_parallel) {
        result.used_parallel = true;
    }
    
    return result;
}

DistanceMatrixResult ParallelDistanceMatrix::computeSymmetricOptimized(const PointContainer& points, DistanceFunction distance_func) {
    DistanceMatrixResult result;
    const size_t n = points.size();
    result.matrix.resize(n, std::vector<double>(n));
    result.total_computations = n * (n - 1) / 2;
    
    // Optimized triangular computation with load balancing
    std::vector<std::pair<size_t, size_t>> work_items;
    work_items.reserve(result.total_computations);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            work_items.emplace_back(i, j);
        }
    }
    
    // Parallel computation with dynamic scheduling
    #pragma omp parallel for schedule(dynamic, 64) if(config_.use_parallel && n >= config_.parallel_threshold)
    for (size_t k = 0; k < work_items.size(); ++k) {
        size_t i = work_items[k].first;
        size_t j = work_items[k].second;
        double dist = distance_func(points[i], points[j]);
        result.matrix[i][j] = dist;
        result.matrix[j][i] = dist;
    }
    
    // Set diagonal to zero
    for (size_t i = 0; i < n; ++i) {
        result.matrix[i][i] = 0.0;
    }
    
    if (config_.use_parallel && n >= config_.parallel_threshold) {
        result.used_parallel = true;
    }
    
    return result;
}

DistanceMatrixResult ParallelDistanceMatrix::computeWithThreshold(const PointContainer& points, 
                                                                  DistanceFunction distance_func, 
                                                                  double threshold) {
    DistanceMatrixResult result;
    const size_t n = points.size();
    result.matrix.resize(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));
    
    size_t computations_performed = 0;
    
    if (config_.symmetric) {
        #pragma omp parallel for schedule(dynamic, 32) reduction(+:computations_performed) if(config_.use_parallel && n >= config_.parallel_threshold)
        for (size_t i = 0; i < n; ++i) {
            result.matrix[i][i] = 0.0;
            for (size_t j = i + 1; j < n; ++j) {
                double dist = distance_func(points[i], points[j]);
                if (dist <= threshold) {
                    result.matrix[i][j] = dist;
                    result.matrix[j][i] = dist;
                }
                computations_performed++;
            }
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(dynamic, 32) reduction(+:computations_performed) if(config_.use_parallel && n >= config_.parallel_threshold)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double dist = distance_func(points[i], points[j]);
                if (dist <= threshold) {
                    result.matrix[i][j] = dist;
                }
                computations_performed++;
            }
        }
    }
    
    result.total_computations = computations_performed;
    
    if (config_.use_parallel && n >= config_.parallel_threshold) {
        result.used_parallel = true;
    }
    
    return result;
}

double ParallelDistanceMatrix::euclideanDistance(const Point& a, const Point& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::max();
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

size_t ParallelDistanceMatrix::estimateCacheMisses(size_t n, size_t block_size) const {
    // Simple cache miss estimation model
    const size_t cache_size = 8 * 1024 * 1024; // 8MB L3 cache estimate
    
    size_t matrix_size_bytes = n * n * sizeof(double);
    size_t blocks_per_dimension = (n + block_size - 1) / block_size;
    size_t total_blocks = blocks_per_dimension * blocks_per_dimension;
    
    if (matrix_size_bytes <= cache_size) {
        return 0; // Fits in cache
    }
    
    return total_blocks * (matrix_size_bytes / cache_size);
}

std::string ParallelDistanceMatrix::selectAlgorithm(const PointContainer& points) const {
    const size_t n = points.size();
    
    if (n < 100) {
        return "sequential";
    } else if (config_.use_simd && !points.empty() && points[0].size() <= 16) {
        return "simd";
    } else if (config_.max_distance != std::numeric_limits<double>::infinity()) {
        return "threshold";
    } else if (config_.symmetric && config_.use_parallel) {
        return "symmetric";
    } else if (n > 1000) {
        return "blocked";
    } else if (config_.use_parallel && n >= config_.parallel_threshold) {
        return "parallel";
    } else {
        return "sequential";
    }
}

// Convenience functions
DistanceMatrixResult computeDistanceMatrix(const std::vector<std::vector<double>>& points,
                                           bool use_parallel,
                                           bool use_simd) {
    DistanceMatrixConfig config;
    config.use_parallel = use_parallel;
    config.use_simd = use_simd;
    
    ParallelDistanceMatrix computer(config);
    return computer.compute(points);
}

DistanceMatrixResult computeSparseDistanceMatrix(const std::vector<std::vector<double>>& points,
                                                 double threshold,
                                                 bool use_parallel) {
    DistanceMatrixConfig config;
    config.use_parallel = use_parallel;
    config.max_distance = threshold;
    
    ParallelDistanceMatrix computer(config);
    return computer.compute(points);
}

} // namespace tda::utils

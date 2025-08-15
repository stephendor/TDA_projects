#pragma once

#include "tda/core/types.hpp"
#include <vector>
#include <functional>
#include <memory>
#include <limits>

namespace tda::utils {

/**
 * @brief Configuration for distance matrix computation
 */
struct DistanceMatrixConfig {
    bool use_parallel = true;           ///< Enable parallel computation
    bool use_simd = true;              ///< Enable SIMD optimizations
    bool symmetric = true;             ///< Matrix is symmetric (compute only upper triangle)
    size_t block_size = 64;            ///< Block size for cache optimization
    size_t parallel_threshold = 1000;  ///< Minimum points to use parallel computation
    double max_distance = std::numeric_limits<double>::infinity(); ///< Maximum distance to compute
};

/**
 * @brief Result structure for distance matrix computation
 */
struct DistanceMatrixResult {
    std::vector<std::vector<double>> matrix;  ///< Distance matrix
    double computation_time_seconds = 0.0;    ///< Computation time
    size_t total_computations = 0;            ///< Total distance computations performed
    size_t cache_misses = 0;                  ///< Estimated cache misses
    bool used_parallel = false;               ///< Whether parallel computation was used
    bool used_simd = false;                   ///< Whether SIMD was used
    std::string algorithm_used = "";          ///< Algorithm variant used
};

/**
 * @brief High-performance parallel distance matrix computation
 */
class ParallelDistanceMatrix {
public:
    using Point = std::vector<double>;
    using PointContainer = std::vector<Point>;
    using DistanceFunction = std::function<double(const Point&, const Point&)>;

    /**
     * @brief Construct with default configuration
     */
    ParallelDistanceMatrix() = default;

    /**
     * @brief Construct with custom configuration
     */
    explicit ParallelDistanceMatrix(const DistanceMatrixConfig& config);

    /**
     * @brief Compute distance matrix with default Euclidean distance
     * @param points Point cloud to compute distances for
     * @return Distance matrix result
     */
    DistanceMatrixResult compute(const PointContainer& points);

    /**
     * @brief Compute distance matrix with custom distance function
     * @param points Point cloud to compute distances for
     * @param distance_func Custom distance function
     * @return Distance matrix result
     */
    DistanceMatrixResult compute(const PointContainer& points, DistanceFunction distance_func);

    /**
     * @brief Compute distance matrix between two different point sets
     * @param points1 First point set
     * @param points2 Second point set  
     * @param distance_func Distance function to use
     * @return Distance matrix result
     */
    DistanceMatrixResult computeBetween(const PointContainer& points1, 
                                        const PointContainer& points2,
                                        DistanceFunction distance_func = nullptr);

    /**
     * @brief Set configuration
     */
    void setConfig(const DistanceMatrixConfig& config) { config_ = config; }

    /**
     * @brief Get current configuration
     */
    const DistanceMatrixConfig& getConfig() const { return config_; }

private:
    DistanceMatrixConfig config_;

    // Core computation methods
    DistanceMatrixResult computeSequential(const PointContainer& points, DistanceFunction distance_func);
    DistanceMatrixResult computeParallel(const PointContainer& points, DistanceFunction distance_func);
    DistanceMatrixResult computeSIMD(const PointContainer& points);
    DistanceMatrixResult computeBlocked(const PointContainer& points, DistanceFunction distance_func);

    // Specialized algorithms
    DistanceMatrixResult computeSymmetricOptimized(const PointContainer& points, DistanceFunction distance_func);
    DistanceMatrixResult computeWithThreshold(const PointContainer& points, DistanceFunction distance_func, double threshold);

    // Utility methods
    static double euclideanDistance(const Point& a, const Point& b);
    size_t estimateCacheMisses(size_t n, size_t block_size) const;
    std::string selectAlgorithm(const PointContainer& points) const;
};

/**
 * @brief Convenience function for computing distance matrix with default settings
 */
DistanceMatrixResult computeDistanceMatrix(const std::vector<std::vector<double>>& points,
                                           bool use_parallel = true,
                                           bool use_simd = true);

/**
 * @brief Convenience function for computing sparse distance matrix (with threshold)
 */
DistanceMatrixResult computeSparseDistanceMatrix(const std::vector<std::vector<double>>& points,
                                                 double threshold,
                                                 bool use_parallel = true);

} // namespace tda::utils

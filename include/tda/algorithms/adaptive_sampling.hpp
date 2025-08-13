#pragma once

#include "tda/core/types.hpp"
#include <vector>
#include <memory>
#include <string>

namespace tda::algorithms {

/**
 * @brief Adaptive Sampling strategies for intelligent point cloud subsampling
 * 
 * This class implements various adaptive sampling techniques to intelligently
 * select representative subsets from large point clouds while preserving
 * topological and geometric properties.
 */
class AdaptiveSampling {
public:
    /**
     * @brief Configuration parameters for adaptive sampling
     */
    struct SamplingConfig {
        double density_threshold = 0.1;        ///< Minimum local density for sampling
        double coverage_radius = 1.0;          ///< Sampling coverage radius
        size_t min_samples = 100;              ///< Minimum sample size
        size_t max_samples = 10000;            ///< Maximum sample size
        std::string strategy = "density";      ///< "density", "geometric", "hybrid", "curvature"
        double noise_tolerance = 0.05;         ///< Tolerance for noise filtering
        bool preserve_boundary = true;         ///< Preserve boundary points
        double boundary_threshold = 0.8;       ///< Threshold for boundary detection
        size_t neighborhood_size = 10;         ///< Size of local neighborhood for analysis
        bool adaptive_radius = true;           ///< Use adaptive radius based on local density
        double quality_target = 0.85;          ///< Target quality for adaptive algorithms
    };

    /**
     * @brief Result structure for adaptive sampling
     */
    struct Result {
        std::vector<size_t> selected_indices;      ///< Indices of selected points
        std::vector<double> local_densities;       ///< Local density at each original point
        std::vector<double> sampling_weights;      ///< Computed sampling weights
        std::vector<bool> boundary_points;         ///< Boundary point indicators
        std::vector<double> curvature_estimates;   ///< Local curvature estimates (if computed)
        double achieved_quality = 0.0;             ///< Achieved sampling quality [0,1]
        double coverage_efficiency = 0.0;          ///< Coverage efficiency metric [0,1]
        double computation_time_seconds = 0.0;     ///< Total computation time
        std::string strategy_used = "";            ///< Strategy that was used
        size_t iterations_performed = 0;           ///< Number of iterations for adaptive methods
    };

    /**
     * @brief Construct adaptive sampling with default configuration
     */
    AdaptiveSampling() = default;

    /**
     * @brief Construct adaptive sampling with custom configuration
     * @param config Configuration parameters
     */
    explicit AdaptiveSampling(const SamplingConfig& config) : config_(config) {}

    /**
     * @brief Perform adaptive sampling on point cloud
     * @param points Input point cloud as nested vectors
     * @param config Configuration parameters
     * @return Result containing sampled indices and analysis data
     */
    tda::core::Result<Result> adaptiveSample(
        const std::vector<std::vector<double>>& points,
        const SamplingConfig& config
    );

    /**
     * @brief Perform adaptive sampling with default configuration
     * @param points Input point cloud as nested vectors
     * @return Result containing sampled indices and analysis data
     */
    tda::core::Result<Result> adaptiveSample(
        const std::vector<std::vector<double>>& points
    );

    /**
     * @brief Get the current configuration
     * @return Current configuration parameters
     */
    const SamplingConfig& getConfig() const { return config_; }

    /**
     * @brief Update configuration parameters
     * @param config New configuration parameters
     */
    void setConfig(const SamplingConfig& config) { config_ = config; }

private:
    SamplingConfig config_;  ///< Current configuration

    /**
     * @brief Density-based adaptive sampling
     * @param points Input point cloud
     * @param config Configuration parameters
     * @return Sampling result
     */
    Result densityBasedSampling(
        const std::vector<std::vector<double>>& points,
        const SamplingConfig& config
    );

    /**
     * @brief Geometric adaptive sampling (Blue noise, Poisson disk)
     * @param points Input point cloud
     * @param config Configuration parameters
     * @return Sampling result
     */
    Result geometricSampling(
        const std::vector<std::vector<double>>& points,
        const SamplingConfig& config
    );

    /**
     * @brief Curvature-based adaptive sampling
     * @param points Input point cloud
     * @param config Configuration parameters
     * @return Sampling result
     */
    Result curvatureBasedSampling(
        const std::vector<std::vector<double>>& points,
        const SamplingConfig& config
    );

    /**
     * @brief Hybrid adaptive sampling combining multiple strategies
     * @param points Input point cloud
     * @param config Configuration parameters
     * @return Sampling result
     */
    Result hybridSampling(
        const std::vector<std::vector<double>>& points,
        const SamplingConfig& config
    );

    /**
     * @brief Compute local density for each point
     * @param points Input point cloud
     * @param radius Neighborhood radius
     * @return Local density values
     */
    std::vector<double> computeLocalDensities(
        const std::vector<std::vector<double>>& points,
        double radius
    );

    /**
     * @brief Detect boundary points in the point cloud
     * @param points Input point cloud
     * @param densities Local densities
     * @param threshold Boundary detection threshold
     * @return Boundary point indicators
     */
    std::vector<bool> detectBoundaryPoints(
        const std::vector<std::vector<double>>& points,
        const std::vector<double>& densities,
        double threshold
    );

    /**
     * @brief Estimate local curvature for each point
     * @param points Input point cloud
     * @param neighborhood_size Size of local neighborhood
     * @return Curvature estimates
     */
    std::vector<double> estimateLocalCurvature(
        const std::vector<std::vector<double>>& points,
        size_t neighborhood_size
    );

    /**
     * @brief Compute sampling weights based on various criteria
     * @param points Input point cloud
     * @param densities Local densities
     * @param boundary_points Boundary indicators
     * @param curvatures Curvature estimates
     * @param strategy Sampling strategy
     * @return Sampling weights
     */
    std::vector<double> computeSamplingWeights(
        const std::vector<std::vector<double>>& points,
        const std::vector<double>& densities,
        const std::vector<bool>& boundary_points,
        const std::vector<double>& curvatures,
        const std::string& strategy
    );

    /**
     * @brief Select points based on computed weights
     * @param weights Sampling weights
     * @param target_samples Target number of samples
     * @param min_samples Minimum samples
     * @param max_samples Maximum samples
     * @return Selected point indices
     */
    std::vector<size_t> selectPointsByWeight(
        const std::vector<double>& weights,
        size_t target_samples,
        size_t min_samples,
        size_t max_samples
    );

    /**
     * @brief Poisson disk sampling for geometric distribution
     * @param points Input point cloud
     * @param radius Minimum distance between samples
     * @param max_samples Maximum number of samples
     * @return Selected point indices
     */
    std::vector<size_t> poissonDiskSampling(
        const std::vector<std::vector<double>>& points,
        double radius,
        size_t max_samples
    );

    /**
     * @brief Compute adaptive radius based on local density
     * @param points Input point cloud
     * @param densities Local densities
     * @param base_radius Base radius
     * @return Adaptive radius for each point
     */
    std::vector<double> computeAdaptiveRadius(
        const std::vector<std::vector<double>>& points,
        const std::vector<double>& densities,
        double base_radius
    );

    /**
     * @brief Evaluate sampling quality
     * @param points Original point cloud
     * @param selected_indices Selected sample indices
     * @param coverage_radius Coverage radius
     * @return Quality metric [0,1]
     */
    double evaluateSamplingQuality(
        const std::vector<std::vector<double>>& points,
        const std::vector<size_t>& selected_indices,
        double coverage_radius
    );

    /**
     * @brief Compute coverage efficiency
     * @param points Original point cloud
     * @param selected_indices Selected sample indices
     * @param coverage_radius Coverage radius
     * @return Coverage efficiency [0,1]
     */
    double computeCoverageEfficiency(
        const std::vector<std::vector<double>>& points,
        const std::vector<size_t>& selected_indices,
        double coverage_radius
    );

    /**
     * @brief Compute Euclidean distance between two points
     * @param p1 First point
     * @param p2 Second point
     * @return Euclidean distance
     */
    double euclideanDistance(
        const std::vector<double>& p1,
        const std::vector<double>& p2
    ) const;

    /**
     * @brief Find k-nearest neighbors of a point
     * @param points Point cloud
     * @param query_idx Index of query point
     * @param k Number of neighbors
     * @return Indices of k-nearest neighbors
     */
    std::vector<size_t> findKNearestNeighbors(
        const std::vector<std::vector<double>>& points,
        size_t query_idx,
        size_t k
    );

    /**
     * @brief Find neighbors within radius
     * @param points Point cloud
     * @param query_idx Index of query point
     * @param radius Search radius
     * @return Indices of neighbors within radius
     */
    std::vector<size_t> findNeighborsInRadius(
        const std::vector<std::vector<double>>& points,
        size_t query_idx,
        double radius
    );
};

} // namespace tda::algorithms

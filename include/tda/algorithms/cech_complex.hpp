#pragma once

#include "tda/core/types.hpp"
#include "tda/spatial/spatial_index.hpp"
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include "tda/core/simplex_pool.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace tda::algorithms {

/**
 * @brief Čech Complex Approximation Algorithm implementation
 * 
 * Implements an efficient approximation of the Čech complex using techniques
 * like witness complexes and alpha-shape approximations. This provides a
 * computationally feasible alternative to exact Čech complex construction
 * while maintaining topological accuracy.
 * 
 * The Čech complex is constructed by considering intersections of balls
 * centered at data points, but exact computation is prohibitively expensive
 * for large point clouds. This implementation uses approximation techniques
 * to achieve similar topological results efficiently.
 */
class CechComplex {
public:
    using Point = std::vector<double>;
    using PointContainer = std::vector<Point>;
    using DistanceFunction = std::function<double(const Point&, const Point&)>;

    /**
     * @brief Configuration parameters for Čech complex approximation
     */
    struct Config {
        double radius;                    // Base radius for ball intersections
        double radiusMultiplier;          // Multiplier for adaptive radius adjustment
        size_t maxDimension;              // Maximum dimension of simplices to compute
        size_t maxNeighbors;              // Maximum neighbors to consider for each point
        bool useAdaptiveRadius;           // Whether to use adaptive radius based on local density
        bool useWitnessComplex;           // Whether to use witness complex approximation
        size_t maxDimensionForSpatialIndex; // Threshold for choosing spatial index type
        DistanceFunction distanceFunc;    // Custom distance function (optional)
        double epsilon;                   // Numerical precision for geometric computations

        Config();
        Config(double radius, double radiusMultiplier = 1.5, size_t maxDim = 3,
               size_t maxNeighbors = 50, bool adaptiveRadius = true, bool witnessComplex = true,
               size_t maxDimForSpatial = 10, double epsilon = 1e-6);
    };

    explicit CechComplex(const Config& config = Config{});
    ~CechComplex() = default;

    // Disable copy constructor and assignment
    CechComplex(const CechComplex&) = delete;
    CechComplex& operator=(const CechComplex&) = delete;

    // Enable move constructor and assignment
    CechComplex(CechComplex&&) noexcept;
    CechComplex& operator=(CechComplex&&) noexcept;

    /**
     * @brief Initialize the Čech complex algorithm with a point cloud
     * @param points The input point cloud
     * @return Result indicating success or failure
     */
    tda::core::Result<void> initialize(const PointContainer& points);

    /**
     * @brief Compute the Čech complex approximation
     * 
     * This method constructs the simplicial complex using the approximation
     * algorithm. Must be called after initialize().
     * 
     * @return Result indicating success or failure
     */
    tda::core::Result<void> computeComplex();

    /**
     * @brief Compute persistent homology
     * 
     * This method computes the persistent homology of the constructed
     * complex. Must be called after computeComplex().
     * 
     * @return Result indicating success or failure
     */
    tda::core::Result<void> computePersistence(int coefficientField = 2);

    /**
     * @brief Get all simplices in the complex
     * 
     * @return Result containing simplex information or error
     */
    tda::core::Result<std::vector<tda::core::SimplexInfo>> getSimplices() const;

    /**
     * @brief Get persistence pairs (birth-death pairs)
     * 
     * @return Result containing persistence pairs or error
     */
    tda::core::Result<std::vector<tda::core::PersistencePair>> getPersistencePairs() const;

    /**
     * @brief Get Betti numbers for each dimension
     * 
     * @return Result containing Betti numbers by dimension or error
     */
    tda::core::Result<std::vector<int>> getBettiNumbers() const;

    /**
     * @brief Get comprehensive statistics about the complex
     * 
     * @return Result containing complex statistics or error
     */
    tda::core::Result<tda::core::ComplexStatistics> getStatistics() const;

    /**
     * @brief Get the configuration used for this computation
     * @return Current configuration
     */
    const Config& getConfig() const;

    /**
     * @brief Update the configuration parameters
     * @param newConfig New configuration to apply
     */
    void updateConfig(const Config& newConfig);

    /**
     * @brief Clear all computed data and reset to initial state
     */
    void clear();

private:
    Config config_;
    PointContainer points_;
    std::unique_ptr<tda::spatial::SpatialIndex> spatialIndex_;
    // Pooled temporaries to reduce allocation churn when constructing simplices
    tda::core::SimplexPool simplex_pool_{};

    // GUDHI objects
    using Simplex_tree = Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
    using Filtration_value = double;
    using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Gudhi::persistent_cohomology::Field_Zp>;

    std::unique_ptr<Simplex_tree> simplex_tree_;
    std::unique_ptr<Persistent_cohomology> persistent_cohomology_;

    // Helper methods
    bool validateInput() const;
    double computeAdaptiveRadius(size_t pointIndex);
    bool checkSimplexIntersection(const std::vector<size_t>& simplex);
    void buildComplexFromNeighbors();
    void addSimplexToTree(const std::vector<size_t>& simplex, double filtrationValue);
    static double euclideanDistance(const Point& a, const Point& b);
    std::vector<size_t> findNeighborsInRadius(size_t pointIndex, double radius);
};

} // namespace tda::algorithms

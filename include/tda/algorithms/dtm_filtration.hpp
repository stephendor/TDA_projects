#pragma once

#include "tda/core/types.hpp"
#include "tda/spatial/spatial_index.hpp"
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <vector>
#include <memory>
#include <functional>
#include "tda/core/simplex_pool.hpp"

namespace tda::algorithms {

/**
 * @brief Distance-to-Measure (DTM) filtration implementation
 * 
 * Implements the DTM function and filtration construction for robust
 * topological analysis that is resistant to noise and outliers.
 */
class DTMFiltration {
public:
    using Point = std::vector<double>;
    using PointContainer = std::vector<Point>;
    using DistanceFunction = std::function<double(const Point&, const Point&)>;
    
    /**
     * @brief Configuration parameters for DTM computation
     */
    struct Config {
        size_t k;                           // Number of nearest neighbors for density estimation
        double power;                       // Power for distance averaging (typically 2.0)
        bool normalize;                     // Whether to normalize DTM values
        size_t maxDimension;                // Threshold for choosing spatial index type
        DistanceFunction distanceFunc;      // Custom distance function (optional)
        
        Config() : k(10), power(2.0), normalize(true), maxDimension(10) {}
        Config(size_t k, double power = 2.0, bool normalize = true, size_t maxDim = 10)
            : k(k), power(power), normalize(normalize), maxDimension(maxDim) {}
    };
    
    explicit DTMFiltration(const Config& config = Config{});
    ~DTMFiltration() = default;
    
    /**
     * @brief Initialize the DTM filtration with a point cloud
     * @param points The input point cloud
     * @return Result indicating success or failure
     */
    tda::core::Result<void> initialize(const PointContainer& points);
    
    /**
     * @brief Compute DTM function values for all points
     * @return Result containing DTM values or error
     */
    tda::core::Result<std::vector<double>> computeDTMFunction();
    
    /**
     * @brief Build the DTM-based filtration
     * @param maxDimension Maximum dimension for the filtration
     * @return Result indicating success or failure
     */
    tda::core::Result<void> buildFiltration(int maxDimension);
    
    /**
     * @brief Compute persistent homology using the DTM filtration
     * @param coefficientField Coefficient field for homology computation
     * @return Result indicating success or failure
     */
    tda::core::Result<void> computePersistence(int coefficientField = 2);
    
    /**
     * @brief Get the computed DTM function values
     * @return Vector of DTM values for each point
     */
    std::vector<double> getDTMValues() const;
    
    /**
     * @brief Get the computed persistence pairs
     * @return Result containing persistence pairs or error
     */
    tda::core::Result<std::vector<tda::core::PersistencePair>> getPersistencePairs() const;
    
    /**
     * @brief Get the computed simplices with their filtration values
     * @return Result containing simplex information or error
     */
    tda::core::Result<std::vector<tda::core::SimplexInfo>> getSimplices() const;
    
    /**
     * @brief Get statistics about the computed complex
     * @return Result containing complex statistics or error
     */
    tda::core::Result<tda::core::ComplexStatistics> getStatistics() const;
    
    /**
     * @brief Get the configuration used for this DTM computation
     * @return Current configuration
     */
    const Config& getConfig() const;
    
    /**
     * @brief Update the configuration (requires reinitialization)
     * @param newConfig New configuration parameters
     */
    void updateConfig(const Config& newConfig);
    
    /**
     * @brief Clear all computed data
     */
    void clear();

private:
    Config config_;
    PointContainer points_;
    std::vector<double> dtmValues_;
    std::unique_ptr<tda::spatial::SpatialIndex> spatialIndex_;
    
    // GUDHI objects for filtration and persistence
    using Simplex_tree = Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
    using Filtration_value = double;
    using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Gudhi::persistent_cohomology::Field_Zp>;
    
    std::unique_ptr<Simplex_tree> simplex_tree_;
    std::unique_ptr<Persistent_cohomology> persistent_cohomology_;
    tda::core::SimplexPool simplex_pool_{};
    
    /**
     * @brief Compute DTM value for a single point
     * @param pointIndex Index of the point to compute DTM for
     * @return DTM value
     */
    double computeDTMForPoint(size_t pointIndex);
    
    /**
     * @brief Normalize DTM values to [0, 1] range
     */
    void normalizeDTMValues();
    
    /**
     * @brief Build the filtration using DTM values
     * @param maxDimension Maximum dimension for the filtration
     */
    void buildFiltrationFromDTM(int maxDimension);
    
    /**
     * @brief Validate input parameters
     * @return true if valid, false otherwise
     */
    bool validateInput() const;
    
    /**
     * @brief Default distance function (Euclidean)
     */
    static double euclideanDistance(const Point& a, const Point& b);
};

} // namespace tda::algorithms

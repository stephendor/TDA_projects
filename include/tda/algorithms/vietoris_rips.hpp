#pragma once

#include "tda/core/types.hpp"
#include "tda/core/simplex.hpp"
#include "tda/core/point_cloud.hpp"
#include "tda/core/filtration.hpp"
#include "tda/algorithms/vietoris_rips_impl.hpp"

#include <vector>
#include <memory>

namespace tda::algorithms {

/**
 * @brief Vietoris-Rips complex computation and persistent homology
 * 
 * This class implements the Vietoris-Rips complex construction algorithm
 * with GUDHI integration for efficient TDA computations. It provides:
 * 
 * - Vietoris-Rips complex construction from point clouds
 * - Persistent homology computation
 * - Betti number calculation
 * - SIMD-optimized distance computations
 * - Comprehensive statistics and analysis
 * 
 * The implementation uses the PIMPL idiom for efficient memory management
 * and clean separation of concerns.
 */
class VietorisRips {
public:
    /**
     * @brief Default constructor
     */
    VietorisRips();
    
    /**
     * @brief Destructor
     */
    ~VietorisRips();
    
    // Disable copy constructor and assignment
    VietorisRips(const VietorisRips&) = delete;
    VietorisRips& operator=(const VietorisRips&) = delete;
    
    // Enable move constructor and assignment
    VietorisRips(VietorisRips&&) noexcept;
    VietorisRips& operator=(VietorisRips&&) noexcept;
    
    /**
     * @brief Initialize the Vietoris-Rips algorithm
     * 
     * @param points Point cloud data (vector of vectors)
     * @param threshold Distance threshold for complex construction
     * @param max_dimension Maximum dimension of simplices to compute
     * @param coefficient_field Coefficient field for homology (default: Z/2Z)
     * @return Result<void> Success or failure with error message
     */
    tda::core::Result<void> initialize(const std::vector<std::vector<double>>& points,
                           double threshold,
                           int max_dimension = 3,
                           int coefficient_field = 2);
    
    /**
     * @brief Compute the Vietoris-Rips complex
     * 
     * This method constructs the simplicial complex based on the
     * initialized parameters. Must be called after initialize().
     * 
     * @return Result<void> Success or failure with error message
     */
    tda::core::Result<void> computeComplex();
    
    /**
     * @brief Compute persistent homology
     * 
     * This method computes the persistent homology of the constructed
     * complex. Must be called after computeComplex().
     * 
     * @return Result<void> Success or failure with error message
     */
    tda::core::Result<void> computePersistence();
    
    /**
     * @brief Get all simplices in the complex
     * 
     * @return Result<std::vector<SimplexInfo>> Vector of simplex information
     */
    tda::core::Result<std::vector<tda::core::SimplexInfo>> getSimplices() const;
    
    /**
     * @brief Get persistence pairs (birth-death pairs)
     * 
     * @return Result<std::vector<PersistencePair>> Vector of persistence pairs
     */
    tda::core::Result<std::vector<tda::core::PersistencePair>> getPersistencePairs() const;
    
    /**
     * @brief Get Betti numbers for each dimension
     * 
     * @return Result<std::vector<int>> Vector of Betti numbers by dimension
     */
    tda::core::Result<std::vector<int>> getBettiNumbers() const;
    
    /**
     * @brief Get comprehensive statistics about the complex
     * 
     * @return Result<ComplexStatistics> Complex statistics
     */
    tda::core::Result<tda::core::ComplexStatistics> getStatistics() const;
    
    /**
     * @brief Compute distances from a query point to all points in the cloud
     * 
     * This method provides SIMD-optimized batch distance computation
     * for efficient processing of large point clouds.
     * 
     * @param points Point cloud
     * @param query_point Query point
     * @return std::vector<double> Vector of distances
     */
    static std::vector<double> computeDistancesBatch(
        const std::vector<std::vector<double>>& points,
        const std::vector<double>& query_point);

private:
    std::unique_ptr<VietorisRipsImpl> impl_;
};

} // namespace tda::algorithms

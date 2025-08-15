#pragma once

#include "tda/core/types.hpp"
#include "tda/core/simplex.hpp"
#include "tda/core/point_cloud.hpp"
#include "tda/core/filtration.hpp"
#include "tda/algorithms/alpha_complex_impl.hpp"

#include <vector>
#include <memory>

namespace tda::algorithms {

/**
 * @brief Alpha Complex computation and persistent homology
 * 
 * This class implements the Alpha Complex construction algorithm
 * with GUDHI integration for efficient TDA computations. It provides:
 * 
 * - Alpha Complex construction from point clouds
 * - Persistent homology computation
 * - Betti number calculation
 * - Support for 2D, 3D, and dynamic dimension point clouds
 * - Comprehensive statistics and analysis
 * 
 * The implementation uses the PIMPL idiom for efficient memory management
 * and clean separation of concerns.
 */
class AlphaComplex {
public:
    /**
     * @brief Default constructor
     */
    AlphaComplex();
    
    /**
     * @brief Destructor
     */
    ~AlphaComplex();
    
    // Disable copy constructor and assignment
    AlphaComplex(const AlphaComplex&) = delete;
    AlphaComplex& operator=(const AlphaComplex&) = delete;
    
    // Enable move constructor and assignment
    AlphaComplex(AlphaComplex&&) noexcept;
    AlphaComplex& operator=(AlphaComplex&&) noexcept;
    
    /**
     * @brief Initialize the Alpha Complex algorithm
     * 
     * @param points Point cloud data (vector of vectors)
     * @param max_dimension Maximum dimension of simplices to compute
     * @param coefficient_field Coefficient field for homology (default: Z/2Z)
     * @return Result<void> Success or failure with error message
     */
    tda::core::Result<void> initialize(const std::vector<std::vector<double>>& points,
                           int max_dimension = 3,
                           int coefficient_field = 2);
    
    /**
     * @brief Compute the Alpha Complex
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

private:
    std::unique_ptr<AlphaComplexImpl> impl_;
};

} // namespace tda::algorithms

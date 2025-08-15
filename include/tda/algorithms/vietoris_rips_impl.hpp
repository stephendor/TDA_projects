#pragma once

#include "tda/core/types.hpp"
#include "tda/core/simplex.hpp"
#include "tda/core/point_cloud.hpp"
#include "tda/core/filtration.hpp"

// GUDHI includes
#include <gudhi/Simplex_tree.h>
#include <gudhi/Rips_complex.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/distance_functions.h>

// Standard library includes
#include <vector>
#include <memory>
#include <algorithm>
// #include <execution>  // Removed - not currently used
#include <immintrin.h> // For SIMD operations

// OpenMP support
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tda::algorithms {

class VietorisRipsImpl {
private:
    using Simplex_tree = Gudhi::Simplex_tree<>;
    using Filtration_value = Simplex_tree::Filtration_value;
    using Rips_complex = Gudhi::rips_complex::Rips_complex<double>;  // CRITICAL FIX: Use double like working GUDHI example
    using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
    using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;
    
    std::unique_ptr<Simplex_tree> simplex_tree_;
    std::unique_ptr<Rips_complex> rips_complex_;
    std::unique_ptr<Persistent_cohomology> persistent_cohomology_;
    
    std::vector<std::vector<double>> points_;
    double threshold_;
    int max_dimension_;
    int coefficient_field_;

public:
    VietorisRipsImpl() = default;
    
    ~VietorisRipsImpl() = default;
    
    // Disable copy constructor and assignment
    VietorisRipsImpl(const VietorisRipsImpl&) = delete;
    VietorisRipsImpl& operator=(const VietorisRipsImpl&) = delete;
    
    // Enable move constructor and assignment
    VietorisRipsImpl(VietorisRipsImpl&&) = default;
    VietorisRipsImpl& operator=(VietorisRipsImpl&&) = default;

    tda::core::Result<void> initialize(const std::vector<std::vector<double>>& points, 
                           double threshold, 
                           int max_dimension = 3,
                           int coefficient_field = 2);
    
    tda::core::Result<void> computeComplex();
    
    tda::core::Result<void> computePersistence();
    
    tda::core::Result<std::vector<tda::core::SimplexInfo>> getSimplices() const;
    
    tda::core::Result<std::vector<tda::core::PersistencePair>> getPersistencePairs() const;
    
    tda::core::Result<std::vector<int>> getBettiNumbers() const;
    
    tda::core::Result<tda::core::ComplexStatistics> getStatistics() const;
    
    static std::vector<double> computeDistancesBatch(const std::vector<std::vector<double>>& points, 
                                                   const std::vector<double>& query_point);

private:
    // CRITICAL FIX: Add cleanup method for RAII safety
    void cleanup();
};

} // namespace tda::algorithms

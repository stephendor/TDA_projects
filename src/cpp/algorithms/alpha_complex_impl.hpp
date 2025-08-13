#pragma once

#include "tda/core/types.hpp"

// GUDHI includes
#include <gudhi/Simplex_tree.h>
#include <gudhi/Alpha_complex_3d.h>
#include <gudhi/Alpha_complex.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/distance_functions.h>

// Standard library includes
#include <vector>
#include <memory>
#include <algorithm>
// #include <execution>  // Removed - not currently used
#include <immintrin.h> // For SIMD operations

namespace tda::algorithms {

class AlphaComplexImpl {
private:
    using Simplex_tree = Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
    using Filtration_value = double; // Use double to support NaN
    using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::EXACT>;
    using Alpha_complex_2d = Gudhi::alpha_complex::Alpha_complex<CGAL::Epeck_d<CGAL::Dimension_tag<2>>, false>;
    using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
    using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;
    
    std::unique_ptr<Simplex_tree> simplex_tree_;
    std::unique_ptr<Alpha_complex_2d> alpha_complex_;
    std::unique_ptr<Persistent_cohomology> persistent_cohomology_;
    
    std::vector<std::vector<double>> points_;
    int max_dimension_;
    int coefficient_field_;
    bool is_3d_;

public:
    AlphaComplexImpl() = default;
    
    ~AlphaComplexImpl() = default;
    
    // Disable copy constructor and assignment
    AlphaComplexImpl(const AlphaComplexImpl&) = delete;
    AlphaComplexImpl& operator=(const AlphaComplexImpl&) = delete;
    
    // Enable move constructor and assignment
    AlphaComplexImpl(AlphaComplexImpl&&) = default;
    AlphaComplexImpl& operator=(AlphaComplexImpl&&) = default;

    tda::core::Result<void> initialize(const std::vector<std::vector<double>>& points, 
                           int max_dimension = 3,
                           int coefficient_field = 2);
    
    tda::core::Result<void> computeComplex();
    
    tda::core::Result<void> computePersistence();
    
    tda::core::Result<std::vector<tda::core::SimplexInfo>> getSimplices() const;
    
    tda::core::Result<std::vector<tda::core::PersistencePair>> getPersistencePairs() const;
    
    tda::core::Result<std::vector<int>> getBettiNumbers() const;
    
    tda::core::Result<tda::core::ComplexStatistics> getStatistics() const;
};

} // namespace tda::algorithms

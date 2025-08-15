#pragma once

#include "tda/core/types.hpp"
#include "tda/core/simplex.hpp"
#include "tda/core/point_cloud.hpp"
#include "tda/core/filtration.hpp"

// GUDHI includes
#include <gudhi/Alpha_complex_3d.h>
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>

// CGAL includes for Alpha Complex
#include <CGAL/Epick_d.h>
#include <CGAL/Dimension.h>

// Standard library includes
#include <vector>
#include <memory>
#include <algorithm>

namespace tda::algorithms {

class AlphaComplexImpl {
private:
    using Simplex_tree = Gudhi::Simplex_tree<>;
    using Filtration_value = Simplex_tree::Filtration_value;
    using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
    using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;
    
    // 3D Alpha Complex
    using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::EXACT>;
    
    // Dynamic dimension Alpha Complex
    using Kernel = CGAL::Epick_d<CGAL::Dynamic_dimension_tag>;
    using Alpha_complex = Gudhi::alpha_complex::Alpha_complex<Kernel, false>;
    
    // 2D Alpha Complex
    using Kernel_2d = CGAL::Epick_d<CGAL::Dimension_tag<2>>;
    using Alpha_complex_2d = Gudhi::alpha_complex::Alpha_complex<Kernel_2d, false>;
    
    std::unique_ptr<Simplex_tree> simplex_tree_;
    std::unique_ptr<Persistent_cohomology> persistent_cohomology_;
    
    // Alpha complex objects for different dimensions
    std::unique_ptr<Alpha_complex_3d> alpha_complex_3d_;
    std::unique_ptr<Alpha_complex> alpha_complex_dynamic_;
    std::unique_ptr<Alpha_complex_2d> alpha_complex_2d_;
    
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

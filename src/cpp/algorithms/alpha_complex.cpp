#include "tda/algorithms/alpha_complex.hpp"
#include "tda/algorithms/alpha_complex_impl.hpp"

namespace tda::algorithms {

// AlphaComplexImpl constructor - using default from header

// AlphaComplexImpl method implementations
tda::core::Result<void> AlphaComplexImpl::initialize(const std::vector<std::vector<double>>& points, 
                           int max_dimension,
                           int coefficient_field) {
    try {
        points_ = points;
        max_dimension_ = max_dimension;
        coefficient_field_ = coefficient_field;
        
        // Validate inputs
        if (points_.empty()) {
            return tda::core::Result<void>::failure("Point cloud cannot be empty");
        }
        
        if (max_dimension_ < 0) {
            return tda::core::Result<void>::failure("Max dimension cannot be negative");
        }
        
        // Check if we have 3D points
        if (points_[0].size() == 3) {
            is_3d_ = true;
        } else if (points_[0].size() == 2) {
            is_3d_ = false;
        } else {
            return tda::core::Result<void>::failure("Only 2D and 3D point clouds are supported");
        }
        
        // Validate all points have the same dimension
        for (const auto& point : points_) {
            if (point.size() != points_[0].size()) {
                return tda::core::Result<void>::failure("All points must have the same dimension");
            }
        }
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure("Initialization failed: " + std::string(e.what()));
    }
}

tda::core::Result<void> AlphaComplexImpl::computeComplex() {
    try {
        // Create simplex tree
        simplex_tree_ = std::make_unique<Simplex_tree>();
        
        if (is_3d_) {
            // For 3D points, use the specialized 3D Alpha complex
            // Convert points to the format expected by GUDHI
            std::vector<Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::EXACT>::Point_3> gudhi_points;
            gudhi_points.reserve(points_.size());
            
            for (const auto& point : points_) {
                gudhi_points.emplace_back(point[0], point[1], point[2]);
            }
            
            // Create 3D Alpha complex
            auto alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::EXACT>(gudhi_points);
            
            // Create the simplex tree
            alpha_complex_3d.create_complex(*simplex_tree_);
        } else {
            // For 2D points, use the 2D Alpha complex with Epick_d kernel
            using Point_2 = CGAL::Epick_d<CGAL::Dimension_tag<2>>::Point_d;
            std::vector<Point_2> cgal_points;
            cgal_points.reserve(points_.size());
            
            for (const auto& point : points_) {
                // Create CGAL point with 2D coordinates
                cgal_points.emplace_back(point[0], point[1]);
            }
            
            // Create 2D Alpha complex
            auto alpha_complex_2d = Gudhi::alpha_complex::Alpha_complex<CGAL::Epick_d<CGAL::Dimension_tag<2>>>(cgal_points);
            
            // Create the simplex tree
            alpha_complex_2d.create_complex(*simplex_tree_);
        }
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure("Complex computation failed: " + std::string(e.what()));
    }
}

tda::core::Result<void> AlphaComplexImpl::computePersistence() {
    try {
        if (!simplex_tree_) {
            return tda::core::Result<void>::failure("Complex must be computed before persistence");
        }
        
        // Create persistent cohomology object
        persistent_cohomology_ = std::make_unique<Persistent_cohomology>(*simplex_tree_);
        
        // Compute persistence
        persistent_cohomology_->init_coefficients(coefficient_field_);
        persistent_cohomology_->compute_persistent_cohomology();
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure("Persistence computation failed: " + std::string(e.what()));
    }
}

tda::core::Result<std::vector<tda::core::SimplexInfo>> AlphaComplexImpl::getSimplices() const {
    try {
        if (!simplex_tree_) {
            return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure("Complex not computed");
        }
        
        std::vector<tda::core::SimplexInfo> simplices;
        
        // Iterate through all simplices in the simplex tree
        for (auto simplex : simplex_tree_->filtration_simplex_range()) {
            tda::core::SimplexInfo info;
            info.dimension = simplex_tree_->dimension(simplex);
            info.filtration_value = simplex_tree_->filtration(simplex);
            
            // Get vertices of the simplex
            for (auto vertex : simplex_tree_->simplex_vertex_range(simplex)) {
                info.vertices.push_back(vertex);
            }
            
            simplices.push_back(std::move(info));
        }
        
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::success(std::move(simplices));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure("Failed to get simplices: " + std::string(e.what()));
    }
}

tda::core::Result<std::vector<tda::core::PersistencePair>> AlphaComplexImpl::getPersistencePairs() const {
    try {
        if (!persistent_cohomology_) {
            return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure("Persistence not computed");
        }
        
        std::vector<tda::core::PersistencePair> pairs;
        
        // Extract persistence pairs from GUDHI's persistent cohomology
        // GUDHI uses persistent_pairs_ member variable, not a method
        // The pairs are now tuples with (birth_simplex_handle, death_simplex_handle, dimension)
        for (auto pcoh_tuple : persistent_cohomology_->persistent_pairs_) {
            tda::core::PersistencePair pair;
            
            // Extract values from tuple: (birth_simplex, death_simplex, dimension)
            auto birth_simplex_handle = std::get<0>(pcoh_tuple);
            auto death_simplex_handle = std::get<1>(pcoh_tuple);
            int dimension = std::get<2>(pcoh_tuple);
            
            // Set pair properties using the correct field names
            pair.dimension = dimension;
            pair.birth = simplex_tree_->filtration(birth_simplex_handle);
            pair.death = simplex_tree_->filtration(death_simplex_handle);
            pair.birth_simplex = simplex_tree_->key(birth_simplex_handle);
            pair.death_simplex = simplex_tree_->key(death_simplex_handle);
            
            pairs.push_back(std::move(pair));
        }
        
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::success(std::move(pairs));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure("Failed to get persistence pairs: " + std::string(e.what()));
    }
}

tda::core::Result<tda::core::ComplexStatistics> AlphaComplexImpl::getStatistics() const {
    try {
        if (!simplex_tree_) {
            return tda::core::Result<tda::core::ComplexStatistics>::failure("Complex not computed");
        }
        
        tda::core::ComplexStatistics stats;
        stats.num_points = points_.size();
        stats.num_simplices = simplex_tree_->num_simplices();
        stats.max_dimension = simplex_tree_->dimension();
        
        return tda::core::Result<tda::core::ComplexStatistics>::success(stats);
    } catch (const std::exception& e) {
        return tda::core::Result<tda::core::ComplexStatistics>::failure("Failed to get statistics: " + std::string(e.what()));
    }
}

// AlphaComplex wrapper class implementation
AlphaComplex::AlphaComplex() : impl_(std::make_unique<AlphaComplexImpl>()) {}

AlphaComplex::~AlphaComplex() = default;

tda::core::Result<void> AlphaComplex::initialize(const std::vector<std::vector<double>>& points, 
                                               int max_dimension, 
                                               int coefficient_field) {
    return impl_->initialize(points, max_dimension, coefficient_field);
}

tda::core::Result<void> AlphaComplex::computeComplex() {
    return impl_->computeComplex();
}

tda::core::Result<void> AlphaComplex::computePersistence() {
    return impl_->computePersistence();
}

tda::core::Result<std::vector<tda::core::SimplexInfo>> AlphaComplex::getSimplices() const {
    return impl_->getSimplices();
}

tda::core::Result<std::vector<tda::core::PersistencePair>> AlphaComplex::getPersistencePairs() const {
    return impl_->getPersistencePairs();
}

tda::core::Result<tda::core::ComplexStatistics> AlphaComplex::getStatistics() const {
    return impl_->getStatistics();
}

} // namespace tda::algorithms

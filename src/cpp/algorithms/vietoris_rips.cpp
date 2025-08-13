#include "tda/algorithms/vietoris_rips.hpp"
#include "tda/core/types.hpp"
#include "tda/utils/simd_utils.hpp"
#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <stdexcept>

namespace tda::algorithms {

// Implementation of VietorisRipsImpl class
tda::core::Result<void> VietorisRipsImpl::initialize(const std::vector<std::vector<double>>& points, 
                                       double threshold, 
                                       int max_dimension,
                                       int coefficient_field) {
    try {
        // CRITICAL FIX: Clear existing resources before re-initializing
        cleanup();
        
        points_ = points;
        threshold_ = threshold;
        max_dimension_ = max_dimension;
        coefficient_field_ = coefficient_field;
        
        // CRITICAL FIX: Validate input parameters
        if (points.empty()) {
            return tda::core::Result<void>::failure("Empty point cloud provided");
        }
        if (threshold <= 0.0) {
            return tda::core::Result<void>::failure("Threshold must be positive");
        }
        if (max_dimension < 0) {
            return tda::core::Result<void>::failure("Max dimension must be non-negative");
        }
        
        // Initialize GUDHI components with RAII safety
        try {
            simplex_tree_ = std::make_unique<Simplex_tree>();
            rips_complex_ = std::make_unique<Rips_complex>(points_, threshold_, Gudhi::Euclidean_distance());
            persistent_cohomology_ = std::make_unique<Persistent_cohomology>(*simplex_tree_);
        } catch (const std::exception& e) {
            // CRITICAL FIX: Clean up on initialization failure
            cleanup();
            return tda::core::Result<void>::failure("Failed to initialize GUDHI components: " + std::string(e.what()));
        }
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        cleanup(); // Ensure cleanup on any exception
        return tda::core::Result<void>::failure("Failed to initialize Vietoris-Rips: " + std::string(e.what()));
    }
}

tda::core::Result<void> VietorisRipsImpl::computeComplex() {
    try {
        if (!rips_complex_) {
            return tda::core::Result<void>::failure("Vietoris-Rips not initialized");
        }
        
        // Create the simplicial complex
        rips_complex_->create_complex(*simplex_tree_, max_dimension_);
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure("Failed to compute complex: " + std::string(e.what()));
    }
}

tda::core::Result<void> VietorisRipsImpl::computePersistence() {
    try {
        if (!simplex_tree_ || !persistent_cohomology_) {
            return tda::core::Result<void>::failure("Complex not computed yet");
        }
        
        // Compute persistent cohomology
        persistent_cohomology_->init_coefficients(coefficient_field_);
        persistent_cohomology_->compute_persistent_cohomology();
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure("Failed to compute persistence: " + std::string(e.what()));
    }
}

tda::core::Result<std::vector<tda::core::SimplexInfo>> VietorisRipsImpl::getSimplices() const {
    try {
        if (!simplex_tree_) {
            return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure("Complex not computed yet");
        }
        
        std::vector<tda::core::SimplexInfo> simplices;
        for (auto simplex_handle : simplex_tree_->complex_simplex_range()) {
            tda::core::SimplexInfo info;
            info.dimension = simplex_tree_->dimension(simplex_handle);
            info.filtration_value = simplex_tree_->filtration(simplex_handle);
            
            // Extract vertices
            for (auto vertex : simplex_tree_->simplex_vertex_range(simplex_handle)) {
                info.vertices.push_back(vertex);
            }
            
            simplices.push_back(std::move(info));
        }
        
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::success(std::move(simplices));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure("Failed to get simplices: " + std::string(e.what()));
    }
}

tda::core::Result<std::vector<tda::core::PersistencePair>> VietorisRipsImpl::getPersistencePairs() const {
    try {
        if (!persistent_cohomology_) {
            return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure("Persistence not computed yet");
        }
        
        std::vector<tda::core::PersistencePair> pairs;
        for (auto pair : persistent_cohomology_->get_persistent_pairs()) {
            tda::core::PersistencePair persistence_pair;
            auto birth_handle = std::get<0>(pair);
            auto death_handle = std::get<1>(pair);
            persistence_pair.birth = simplex_tree_->filtration(birth_handle);
            persistence_pair.death = simplex_tree_->filtration(death_handle);
            persistence_pair.dimension = simplex_tree_->dimension(birth_handle);
            pairs.push_back(std::move(persistence_pair));
        }
        
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::success(std::move(pairs));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure("Failed to get persistence pairs: " + std::string(e.what()));
    }
}

tda::core::Result<std::vector<int>> VietorisRipsImpl::getBettiNumbers() const {
    try {
        if (!persistent_cohomology_) {
            return tda::core::Result<std::vector<int>>::failure("Persistence not computed yet");
        }
        
        std::vector<int> betti_numbers;
        for (int dim = 0; dim <= max_dimension_; ++dim) {
            int betti = persistent_cohomology_->betti_number(dim);
            betti_numbers.push_back(betti);
        }
        
        return tda::core::Result<std::vector<int>>::success(std::move(betti_numbers));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<int>>::failure("Failed to get Betti numbers: " + std::string(e.what()));
    }
}

tda::core::Result<tda::core::ComplexStatistics> VietorisRipsImpl::getStatistics() const {
    try {
        if (!simplex_tree_) {
            return tda::core::Result<tda::core::ComplexStatistics>::failure("Complex not computed yet");
        }
        
        tda::core::ComplexStatistics stats;
        size_t total_simplices = simplex_tree_->num_simplices();
        
        // Safety check: if we have more simplices than points^3, something is wrong
        if (total_simplices > points_.size() * points_.size() * points_.size()) {
            return tda::core::Result<tda::core::ComplexStatistics>::failure(
                "Simplex count too large: " + std::to_string(total_simplices) + 
                " for " + std::to_string(points_.size()) + " points. Threshold may be too large.");
        }
        
        stats.num_simplices = total_simplices;
        stats.max_dimension = simplex_tree_->dimension();
        stats.num_points = points_.size();
        stats.threshold = threshold_;
        
        // Initialize simplex_count_by_dim vector with proper size
        int max_dim = simplex_tree_->dimension();
        stats.simplex_count_by_dim.resize(max_dim + 1, 0);
        
        // Count simplices by dimension with safety check
        size_t counted_simplices = 0;
        for (auto simplex_handle : simplex_tree_->complex_simplex_range()) {
            if (counted_simplices >= total_simplices) {
                break; // Safety check to prevent infinite loop
            }
            
            int dim = simplex_tree_->dimension(simplex_handle);
            if (dim >= 0 && dim < static_cast<int>(stats.simplex_count_by_dim.size())) {
                stats.simplex_count_by_dim[dim]++;
            }
            counted_simplices++;
        }
        
        return tda::core::Result<tda::core::ComplexStatistics>::success(std::move(stats));
    } catch (const std::exception& e) {
        return tda::core::Result<tda::core::ComplexStatistics>::failure("Failed to get statistics: " + std::string(e.what()));
    }
}

std::vector<double> VietorisRipsImpl::computeDistancesBatch(const std::vector<std::vector<double>>& points, 
                                                           const std::vector<double>& query_point) {
    std::vector<double> distances;
    distances.reserve(points.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i) {
        double dist = 0.0;
        for (size_t j = 0; j < query_point.size() && j < points[i].size(); ++j) {
            double diff = query_point[j] - points[i][j];
            dist += diff * diff;
        }
        distances.push_back(std::sqrt(dist));
    }
    
    return distances;
}

// CRITICAL FIX: Implement cleanup method for RAII safety
void VietorisRipsImpl::cleanup() {
    // Release GUDHI resources in reverse order of creation
    persistent_cohomology_.reset();
    rips_complex_.reset();
    simplex_tree_.reset();
    
    // Clear point data
    points_.clear();
    
    // Reset configuration
    threshold_ = 0.0;
    max_dimension_ = 0;
    coefficient_field_ = 2;
}

} // namespace tda::algorithms
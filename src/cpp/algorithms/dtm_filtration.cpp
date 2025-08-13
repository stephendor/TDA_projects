#include "tda/algorithms/dtm_filtration.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace tda::algorithms {

DTMFiltration::DTMFiltration(const Config& config) 
    : config_(config) {
    // Set default distance function if none provided
    if (!config_.distanceFunc) {
        config_.distanceFunc = euclideanDistance;
    }
}

tda::core::Result<void> DTMFiltration::initialize(const PointContainer& points) {
    try {
        // CRITICAL FIX: Add comprehensive input validation
        if (points.empty()) {
            return tda::core::Result<void>::failure("Empty point cloud provided");
        }
        
        if (config_.k <= 0 || config_.k > points.size()) {
            return tda::core::Result<void>::failure("Invalid k value: " + std::to_string(config_.k) + 
                                                  " (must be > 0 and <= " + std::to_string(points.size()) + ")");
        }
        
        if (config_.power <= 0.0) {
            return tda::core::Result<void>::failure("Invalid power value: " + std::to_string(config_.power) + 
                                                  " (must be > 0)");
        }
        
        // Validate point dimensions
        size_t expected_dim = points[0].size();
        for (size_t i = 1; i < points.size(); ++i) {
            if (points[i].size() != expected_dim) {
                return tda::core::Result<void>::failure("Inconsistent point dimensions: expected " + 
                                                      std::to_string(expected_dim) + ", got " + 
                                                      std::to_string(points[i].size()) + " at index " + 
                                                      std::to_string(i));
            }
        }
        
        points_ = points;
        
        // Initialize spatial index for efficient neighbor search
        spatialIndex_ = std::make_unique<tda::spatial::KDTree>(20); // Default max depth
        if (!spatialIndex_->build(points_)) {
            return tda::core::Result<void>::failure("Failed to build spatial index");
        }
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure("Failed to initialize DTM filtration: " + std::string(e.what()));
    }
}

tda::core::Result<std::vector<double>> DTMFiltration::computeDTMFunction() {
    if (points_.empty() || !spatialIndex_) {
        return tda::core::Result<std::vector<double>>::failure("Not initialized");
    }
    
    try {
        dtmValues_.clear();
        dtmValues_.reserve(points_.size());
        
        // Compute DTM value for each point
        for (size_t i = 0; i < points_.size(); ++i) {
            double dtmValue = computeDTMForPoint(i);
            dtmValues_.push_back(dtmValue);
        }
        
        // Normalize DTM values if requested
        if (config_.normalize) {
            normalizeDTMValues();
        }
        
        return tda::core::Result<std::vector<double>>::success(dtmValues_);
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<double>>::failure(std::string("DTM computation failed: ") + e.what());
    }
}

double DTMFiltration::computeDTMForPoint(size_t pointIndex) {
    if (pointIndex >= points_.size()) {
        return std::numeric_limits<double>::max();
    }
    
    const Point& queryPoint = points_[pointIndex];
    
    // Find k nearest neighbors (excluding the point itself)
    size_t k = std::min(config_.k + 1, points_.size()); // +1 to account for self
    auto neighbors = spatialIndex_->kNearestNeighbors(queryPoint, k);
    
    // Filter out the point itself and collect distances
    std::vector<double> distances;
    distances.reserve(neighbors.size());
    
    for (const auto& [idx, dist] : neighbors) {
        if (idx != pointIndex) {
            distances.push_back(dist);
        }
    }
    
    // If we don't have enough neighbors, return a large value
    if (distances.size() < config_.k) {
        return std::numeric_limits<double>::max();
    }
    
    // Sort distances and take the k closest
    std::sort(distances.begin(), distances.end());
    distances.resize(config_.k);
    
    // Compute DTM value: average of k-th power of distances, raised to 1/power
    double sum = 0.0;
    for (double dist : distances) {
        sum += std::pow(dist, config_.power);
    }
    
    // CRITICAL FIX: Handle case where power is 0 to prevent division by zero
    double dtmValue;
    if (std::abs(config_.power) < 1e-10) {
        // Power is effectively 0, use geometric mean instead
        double product = 1.0;
        for (double dist : distances) {
            product *= dist;
        }
        dtmValue = std::pow(product, 1.0 / distances.size());
    } else {
        // Normal case: raise to 1/power
        dtmValue = std::pow(sum / config_.k, 1.0 / config_.power);
    }
    return dtmValue;
}

void DTMFiltration::normalizeDTMValues() {
    if (dtmValues_.empty()) {
        return;
    }
    
    // Find min and max values
    auto [minIt, maxIt] = std::minmax_element(dtmValues_.begin(), dtmValues_.end());
    double minVal = *minIt;
    double maxVal = *maxIt;
    
    // Avoid division by zero
    if (std::abs(maxVal - minVal) < 1e-10) {
        // All values are the same, set to 0.5
        std::fill(dtmValues_.begin(), dtmValues_.end(), 0.5);
        return;
    }
    
    // Normalize to [0, 1]
    for (double& value : dtmValues_) {
        value = (value - minVal) / (maxVal - minVal);
    }
}

tda::core::Result<void> DTMFiltration::buildFiltration(int maxDimension) {
    if (dtmValues_.empty()) {
        return tda::core::Result<void>::failure("DTM function not computed");
    }
    
    if (maxDimension < 0) {
        return tda::core::Result<void>::failure("Max dimension must be non-negative");
    }
    
    try {
        // Lazily initialize simplex tree and persistent cohomology
        if (!simplex_tree_) {
            simplex_tree_ = std::make_unique<Simplex_tree>();
        }
        if (!persistent_cohomology_) {
            persistent_cohomology_ = std::make_unique<Persistent_cohomology>(*simplex_tree_);
        }

        buildFiltrationFromDTM(maxDimension);
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure(std::string("Filtration build failed: ") + e.what());
    }
}

void DTMFiltration::buildFiltrationFromDTM(int maxDimension) {
    if (!simplex_tree_) {
        throw std::runtime_error("Simplex tree not initialized");
    }
    
    simplex_tree_->clear();
    
    // Add vertices (0-simplices) with DTM-based filtration values
    for (size_t i = 0; i < points_.size(); ++i) {
        double filtrationValue = dtmValues_[i];
        // Use pooled simplex to build a temporary vertex list (arity 1)
        tda::core::Simplex* tmp = simplex_pool_.acquire(1);
        const std::vector<tda::core::Index>* verts_ptr = nullptr;
        if (tmp) {
            auto& verts = tmp->vertices();
            verts.clear();
            verts.reserve(1);
            verts.push_back(static_cast<tda::core::Index>(i));
            tmp->setFiltrationValue(filtrationValue);
            verts_ptr = &verts;
        }
        if (verts_ptr) {
            simplex_tree_->insert_simplex({static_cast<int>((*verts_ptr)[0])}, filtrationValue);
            simplex_pool_.release(tmp, 1);
        } else {
            simplex_tree_->insert_simplex({static_cast<int>(i)}, filtrationValue);
        }
    }
    
    // Build higher-dimensional simplices using Vietoris-Rips approach
    // but with DTM-modified filtration values
    for (size_t i = 0; i < points_.size(); ++i) {
        for (size_t j = i + 1; j < points_.size(); ++j) {
            // Compute edge filtration value based on DTM values
            double edgeFiltration = std::max(dtmValues_[i], dtmValues_[j]);
            
            // Add edge if it meets the dimension constraint
            if (maxDimension >= 1) {
                // Use pooled simplex (arity 2)
                tda::core::Simplex* tmp = simplex_pool_.acquire(2);
                const std::vector<tda::core::Index>* verts_ptr = nullptr;
                if (tmp) {
                    auto& verts = tmp->vertices();
                    verts.clear();
                    verts.reserve(2);
                    verts.push_back(static_cast<tda::core::Index>(i));
                    verts.push_back(static_cast<tda::core::Index>(j));
                    tmp->setFiltrationValue(edgeFiltration);
                    verts_ptr = &verts;
                }
                if (verts_ptr) {
                    simplex_tree_->insert_simplex({static_cast<int>((*verts_ptr)[0]), static_cast<int>((*verts_ptr)[1])}, edgeFiltration);
                    simplex_pool_.release(tmp, 2);
                } else {
                    simplex_tree_->insert_simplex({static_cast<int>(i), static_cast<int>(j)}, edgeFiltration);
                }
            }
        }
    }
    
    // Add triangles (2-simplices) if requested
    if (maxDimension >= 2) {
        for (size_t i = 0; i < points_.size(); ++i) {
            for (size_t j = i + 1; j < points_.size(); ++j) {
                for (size_t k = j + 1; k < points_.size(); ++k) {
                    // Compute triangle filtration value
                    double triangleFiltration = std::max({dtmValues_[i], dtmValues_[j], dtmValues_[k]});
                    // Use pooled simplex (arity 3)
                    tda::core::Simplex* tmp = simplex_pool_.acquire(3);
                    const std::vector<tda::core::Index>* verts_ptr = nullptr;
                    if (tmp) {
                        auto& verts = tmp->vertices();
                        verts.clear();
                        verts.reserve(3);
                        verts.push_back(static_cast<tda::core::Index>(i));
                        verts.push_back(static_cast<tda::core::Index>(j));
                        verts.push_back(static_cast<tda::core::Index>(k));
                        tmp->setFiltrationValue(triangleFiltration);
                        verts_ptr = &verts;
                    }
                    if (verts_ptr) {
                        simplex_tree_->insert_simplex({static_cast<int>((*verts_ptr)[0]), static_cast<int>((*verts_ptr)[1]), static_cast<int>((*verts_ptr)[2])}, triangleFiltration);
                        simplex_pool_.release(tmp, 3);
                    } else {
                        simplex_tree_->insert_simplex({static_cast<int>(i), static_cast<int>(j), static_cast<int>(k)}, triangleFiltration);
                    }
                }
            }
        }
    }
    
    // Add tetrahedra (3-simplices) if requested
    if (maxDimension >= 3) {
        for (size_t i = 0; i < points_.size(); ++i) {
            for (size_t j = i + 1; j < points_.size(); ++j) {
                for (size_t k = j + 1; k < points_.size(); ++k) {
                    for (size_t l = k + 1; l < points_.size(); ++l) {
                        // Compute tetrahedron filtration value
                        double tetrahedronFiltration = std::max({dtmValues_[i], dtmValues_[j], dtmValues_[k], dtmValues_[l]});
                        // Use pooled simplex (arity 4)
                        tda::core::Simplex* tmp = simplex_pool_.acquire(4);
                        const std::vector<tda::core::Index>* verts_ptr = nullptr;
                        if (tmp) {
                            auto& verts = tmp->vertices();
                            verts.clear();
                            verts.reserve(4);
                            verts.push_back(static_cast<tda::core::Index>(i));
                            verts.push_back(static_cast<tda::core::Index>(j));
                            verts.push_back(static_cast<tda::core::Index>(k));
                            verts.push_back(static_cast<tda::core::Index>(l));
                            tmp->setFiltrationValue(tetrahedronFiltration);
                            verts_ptr = &verts;
                        }
                        if (verts_ptr) {
                            simplex_tree_->insert_simplex({static_cast<int>((*verts_ptr)[0]), static_cast<int>((*verts_ptr)[1]), static_cast<int>((*verts_ptr)[2]), static_cast<int>((*verts_ptr)[3])}, tetrahedronFiltration);
                            simplex_pool_.release(tmp, 4);
                        } else {
                            simplex_tree_->insert_simplex({static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), static_cast<int>(l)}, tetrahedronFiltration);
                        }
                    }
                }
            }
        }
    }
}

tda::core::Result<void> DTMFiltration::computePersistence(int coefficientField) {
    if (!simplex_tree_ || !persistent_cohomology_) {
        return tda::core::Result<void>::failure("Filtration not built");
    }
    
    try {
        // Initialize coefficients
        persistent_cohomology_->init_coefficients(coefficientField);
        
        // Compute persistent cohomology
        persistent_cohomology_->compute_persistent_cohomology(coefficientField);
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure(std::string("Persistence computation failed: ") + e.what());
    }
}

std::vector<double> DTMFiltration::getDTMValues() const {
    return dtmValues_;
}

tda::core::Result<std::vector<tda::core::PersistencePair>> DTMFiltration::getPersistencePairs() const {
    if (!persistent_cohomology_) {
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure("Persistence not computed");
    }
    
    try {
        std::vector<tda::core::PersistencePair> pairs;
        
        // Get persistent pairs from GUDHI
        auto persistent_pairs = persistent_cohomology_->persistent_pairs_;
        
        for (const auto& pair : persistent_pairs) {
            tda::core::PersistencePair tdaPair;
            
            // Extract birth and death simplex handles
            auto birth_simplex_handle = std::get<0>(pair);
            auto death_simplex_handle = std::get<1>(pair);
            int dimension = std::get<2>(pair);
            
            // Set pair properties
            tdaPair.dimension = dimension;
            tdaPair.birth = simplex_tree_->filtration(birth_simplex_handle);
            tdaPair.death = simplex_tree_->filtration(death_simplex_handle);
            tdaPair.birth_simplex = simplex_tree_->key(birth_simplex_handle);
            tdaPair.death_simplex = simplex_tree_->key(death_simplex_handle);
            
            pairs.push_back(std::move(tdaPair));
        }
        
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::success(pairs);
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure(std::string("Failed to get persistence pairs: ") + e.what());
    }
}

tda::core::Result<std::vector<tda::core::SimplexInfo>> DTMFiltration::getSimplices() const {
    if (!simplex_tree_) {
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure("Filtration not built");
    }
    
    try {
        std::vector<tda::core::SimplexInfo> simplices;
        
        for (auto simplex : simplex_tree_->filtration_simplex_range()) {
            tda::core::SimplexInfo simplexInfo;
            
            simplexInfo.dimension = simplex_tree_->dimension(simplex);
            simplexInfo.filtration_value = simplex_tree_->filtration(simplex);
            
            // Extract vertices
            std::vector<int> vertices;
            for (auto vertex : simplex_tree_->simplex_vertex_range(simplex)) {
                vertices.push_back(static_cast<int>(vertex));
            }
            simplexInfo.vertices = vertices;
            
            simplices.push_back(std::move(simplexInfo));
        }
        
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::success(simplices);
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure(std::string("Failed to get simplices: ") + e.what());
    }
}

tda::core::Result<tda::core::ComplexStatistics> DTMFiltration::getStatistics() const {
    if (!simplex_tree_) {
        return tda::core::Result<tda::core::ComplexStatistics>::failure("Filtration not built");
    }
    
    try {
        tda::core::ComplexStatistics stats;
        
        stats.num_points = points_.size();
        stats.num_simplices = simplex_tree_->num_simplices();
        stats.max_dimension = simplex_tree_->dimension();
        
        // Count simplices by dimension
        for (auto simplex : simplex_tree_->filtration_simplex_range()) {
            int dim = simplex_tree_->dimension(simplex);
            if (dim < static_cast<int>(stats.simplex_count_by_dim.size())) {
                stats.simplex_count_by_dim[dim]++;
            }
        }
        
        return tda::core::Result<tda::core::ComplexStatistics>::success(stats);
    } catch (const std::exception& e) {
        return tda::core::Result<tda::core::ComplexStatistics>::failure(std::string("Failed to get statistics: ") + e.what());
    }
}

const DTMFiltration::Config& DTMFiltration::getConfig() const {
    return config_;
}

void DTMFiltration::updateConfig(const Config& newConfig) {
    config_ = newConfig;
    
    // Set default distance function if none provided
    if (!config_.distanceFunc) {
        config_.distanceFunc = euclideanDistance;
    }
    
    // Clear computed data since configuration changed
    clear();
}

void DTMFiltration::clear() {
    points_.clear();
    dtmValues_.clear();
    spatialIndex_.reset();
    simplex_tree_.reset();
    persistent_cohomology_.reset();
}

bool DTMFiltration::validateInput() const {
    if (config_.k == 0) {
        return false;
    }
    
    if (config_.power <= 0.0) {
        return false;
    }
    
    return true;
}

double DTMFiltration::euclideanDistance(const Point& a, const Point& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::max();
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

} // namespace tda::algorithms

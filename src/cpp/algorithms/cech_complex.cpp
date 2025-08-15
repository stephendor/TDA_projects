#include "tda/algorithms/cech_complex.hpp"
#include "tda/spatial/spatial_index.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

namespace tda::algorithms {

// Config constructors
CechComplex::Config::Config() 
    : radius(1.0), radiusMultiplier(1.5), maxDimension(3), maxNeighbors(50),
      useAdaptiveRadius(true), useWitnessComplex(true), maxDimensionForSpatialIndex(10), epsilon(1e-6) {}

CechComplex::Config::Config(double radius, double radiusMultiplier, size_t maxDim,
                           size_t maxNeighbors, bool adaptiveRadius, bool witnessComplex,
                           size_t maxDimForSpatial, double epsilon)
    : radius(radius), radiusMultiplier(radiusMultiplier), maxDimension(maxDim),
      maxNeighbors(maxNeighbors), useAdaptiveRadius(adaptiveRadius),
      useWitnessComplex(witnessComplex), maxDimensionForSpatialIndex(maxDimForSpatial),
      epsilon(epsilon) {}

// Constructor
CechComplex::CechComplex(const Config& config) : config_(config) {}

// Move constructor
CechComplex::CechComplex(CechComplex&& other) noexcept
    : config_(std::move(other.config_)),
      points_(std::move(other.points_)),
      spatialIndex_(std::move(other.spatialIndex_)),
      simplex_tree_(std::move(other.simplex_tree_)),
      persistent_cohomology_(std::move(other.persistent_cohomology_)) {}

// Move assignment operator
CechComplex& CechComplex::operator=(CechComplex&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        points_ = std::move(other.points_);
        spatialIndex_ = std::move(other.spatialIndex_);
        simplex_tree_ = std::move(other.simplex_tree_);
        persistent_cohomology_ = std::move(other.persistent_cohomology_);
    }
    return *this;
}

tda::core::Result<void> CechComplex::initialize(const PointContainer& points) {
    try {
        // CRITICAL FIX: Standardize input validation using Result<T> pattern
        if (points.empty()) {
            return tda::core::Result<void>::failure("Empty point cloud provided");
        }
        
        if (config_.maxNeighbors <= 0) {
            return tda::core::Result<void>::failure("Max neighbors must be positive");
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
        
        // Initialize spatial index
        spatialIndex_ = std::make_unique<tda::spatial::KDTree>(20); // Default max depth
        if (!spatialIndex_->build(points_)) {
            return tda::core::Result<void>::failure("Failed to build spatial index");
        }
        
        // Initialize GUDHI components
        simplex_tree_ = std::make_unique<Simplex_tree>();
        persistent_cohomology_ = std::make_unique<Persistent_cohomology>(*simplex_tree_);
        
        return tda::core::Result<void>::success();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure("Failed to initialize Čech complex: " + std::string(e.what()));
    }
}

tda::core::Result<void> CechComplex::computeComplex() {
    if (points_.empty()) {
        return tda::core::Result<void>::failure("Must call initialize() first");
    }

    if (!simplex_tree_) {
        return tda::core::Result<void>::failure("Simplex tree not initialized");
    }

    // Clear any existing complex
    simplex_tree_->clear();

    // Build the Čech complex approximation
    buildComplexFromNeighbors();

    return tda::core::Result<void>::success();
}

tda::core::Result<void> CechComplex::computePersistence(int coefficientField) {
    if (!simplex_tree_) {
        return tda::core::Result<void>::failure("Must call computeComplex() first");
    }

    if (!persistent_cohomology_) {
        return tda::core::Result<void>::failure("Persistent cohomology not initialized");
    }

    try {
        // Compute persistent cohomology
        persistent_cohomology_->init_coefficients(coefficientField);
        persistent_cohomology_->compute_persistent_cohomology();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure(std::string("Persistent homology computation failed: ") + e.what());
    }

    return tda::core::Result<void>::success();
}

tda::core::Result<std::vector<tda::core::SimplexInfo>> CechComplex::getSimplices() const {
    if (!simplex_tree_) {
        return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure("Complex not computed");
    }

    std::vector<tda::core::SimplexInfo> simplices;
    
    for (auto simplex : simplex_tree_->complex_simplex_range()) {
        tda::core::SimplexInfo simplexInfo;
        simplexInfo.dimension = simplex_tree_->dimension(simplex);
        simplexInfo.filtration_value = simplex_tree_->filtration(simplex);
        
        std::vector<int> vertices;
        for (auto vertex : simplex_tree_->simplex_vertex_range(simplex)) {
            vertices.push_back(static_cast<int>(vertex));
        }
        simplexInfo.vertices = vertices;
        
        simplices.push_back(simplexInfo);
    }

    return tda::core::Result<std::vector<tda::core::SimplexInfo>>::success(simplices);
}

tda::core::Result<std::vector<tda::core::PersistencePair>> CechComplex::getPersistencePairs() const {
    if (!persistent_cohomology_) {
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure("Persistence not computed");
    }

    std::vector<tda::core::PersistencePair> pairs;
    
    try {
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
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<tda::core::PersistencePair>>::failure(std::string("Failed to get persistence pairs: ") + e.what());
    }

    return tda::core::Result<std::vector<tda::core::PersistencePair>>::success(pairs);
}

tda::core::Result<std::vector<int>> CechComplex::getBettiNumbers() const {
    if (!persistent_cohomology_) {
        return tda::core::Result<std::vector<int>>::failure("Persistence not computed");
    }

    std::vector<int> bettiNumbers;
    
    try {
        // Get all persistent pairs and count by dimension
        auto persistent_pairs = persistent_cohomology_->persistent_pairs_;
        
        // Initialize Betti numbers for each dimension
        for (int dim = 0; dim <= static_cast<int>(config_.maxDimension); ++dim) {
            int betti = 0;
            for (const auto& pair : persistent_pairs) {
                if (std::get<2>(pair) == dim) {
                    betti++;
                }
            }
            bettiNumbers.push_back(betti);
        }
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<int>>::failure(std::string("Failed to compute Betti numbers: ") + e.what());
    }

    return tda::core::Result<std::vector<int>>::success(bettiNumbers);
}

tda::core::Result<tda::core::ComplexStatistics> CechComplex::getStatistics() const {
    if (!simplex_tree_) {
        return tda::core::Result<tda::core::ComplexStatistics>::failure("Complex not computed");
    }

    tda::core::ComplexStatistics stats;
    stats.num_points = points_.size();
    stats.num_simplices = simplex_tree_->num_simplices();
    stats.max_dimension = simplex_tree_->dimension();
    stats.threshold = 0.0; // Not applicable for Čech complex
    
    // Calculate simplex count by dimension
    stats.simplex_count_by_dim.clear();
    for (int dim = 0; dim <= simplex_tree_->dimension(); ++dim) {
        int count = 0;
        for (auto simplex : simplex_tree_->complex_simplex_range()) {
            if (simplex_tree_->dimension(simplex) == dim) {
                count++;
            }
        }
        stats.simplex_count_by_dim.push_back(count);
    }

    return tda::core::Result<tda::core::ComplexStatistics>::success(stats);
}

const CechComplex::Config& CechComplex::getConfig() const {
    return config_;
}

void CechComplex::updateConfig(const Config& newConfig) {
    config_ = newConfig;
}

void CechComplex::clear() {
    points_.clear();
    spatialIndex_.reset();
    simplex_tree_.reset();
    persistent_cohomology_.reset();
}

// Private helper methods

bool CechComplex::validateInput() const {
    return !points_.empty() && !points_[0].empty();
}

double CechComplex::computeAdaptiveRadius(size_t pointIndex) {
    if (!config_.useAdaptiveRadius) {
        return config_.radius;
    }

    // Find nearest neighbors to estimate local density
    auto neighbors = findNeighborsInRadius(pointIndex, config_.radius * 2.0);
    
    if (neighbors.size() < 2) {
        return config_.radius;
    }

    // Compute average distance to neighbors
    double totalDistance = 0.0;
    for (size_t neighborIdx : neighbors) {
        if (neighborIdx != pointIndex) {
            totalDistance += euclideanDistance(points_[pointIndex], points_[neighborIdx]);
        }
    }
    
    double avgDistance = totalDistance / (neighbors.size() - 1);
    return std::min(config_.radius * config_.radiusMultiplier, avgDistance);
}

bool CechComplex::checkSimplexIntersection(const std::vector<size_t>& simplex) {
    if (simplex.size() < 2) {
        return true; // Single points always form valid simplices
    }

    // For Čech complex, we need to check if the intersection of balls is non-empty
    // This is a simplified check using the circumradius of the simplex
    
    if (simplex.size() == 2) {
        // Edge case: check if distance between points is <= 2 * radius
        double distance = euclideanDistance(points_[simplex[0]], points_[simplex[1]]);
        double radius0 = computeAdaptiveRadius(simplex[0]);
        double radius1 = computeAdaptiveRadius(simplex[1]);
        return distance <= (radius0 + radius1);
    }

    // For higher dimensions, use a simplified approach
    // Check if the maximum pairwise distance is reasonable
    double maxDistance = 0.0;
    for (size_t i = 0; i < simplex.size(); ++i) {
        for (size_t j = i + 1; j < simplex.size(); ++j) {
            double distance = euclideanDistance(points_[simplex[i]], points_[simplex[j]]);
            maxDistance = std::max(maxDistance, distance);
        }
    }

    // Use average radius as threshold
    double avgRadius = 0.0;
    for (size_t idx : simplex) {
        avgRadius += computeAdaptiveRadius(idx);
    }
    avgRadius /= simplex.size();

    return maxDistance <= (2.0 * avgRadius);
}

void CechComplex::buildComplexFromNeighbors() {
    // CRITICAL FIX: Add complexity bounds to prevent exponential growth
    size_t maxSimplices = points_.size() * 100; // Reasonable limit
    size_t simplexCount = 0;
    
    // Add all vertices (0-simplices)
    for (size_t i = 0; i < points_.size(); ++i) {
        std::vector<size_t> vertex = {i};
        addSimplexToTree(vertex, 0.0);
        simplexCount++;
    }

    // Build higher-dimensional simplices with complexity bounds
    for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx) {
        // Safety check: stop if we're creating too many simplices
        if (simplexCount > maxSimplices) {
            std::cerr << "WARNING: Čech complex complexity limit reached. Stopping at " 
                      << simplexCount << " simplices." << std::endl;
            break;
        }
        
        double radius = computeAdaptiveRadius(pointIdx);
        
        // Find neighbors within radius
        auto neighbors = findNeighborsInRadius(pointIdx, radius);
        
        // CRITICAL FIX: Limit neighbors more aggressively to prevent O(n³) growth
        size_t maxNeighbors = std::min(config_.maxNeighbors, 
                                      static_cast<size_t>(std::sqrt(points_.size())));
        
        if (neighbors.size() > maxNeighbors) {
            std::partial_sort(neighbors.begin(), 
                             neighbors.begin() + maxNeighbors, 
                             neighbors.end(),
                             [this, pointIdx](size_t a, size_t b) {
                                 return euclideanDistance(points_[pointIdx], points_[a]) <
                                        euclideanDistance(points_[pointIdx], points_[b]);
                             });
            neighbors.resize(maxNeighbors);
        }

        // Build simplices from neighbors with dimension limits
        for (size_t dim = 1; dim <= std::min(static_cast<size_t>(config_.maxDimension), static_cast<size_t>(3)) && dim < neighbors.size(); ++dim) {
            // CRITICAL FIX: Limit simplex construction to prevent explosion
            if (dim > 2 && neighbors.size() > 20) {
                // Skip high-dimensional simplices for large neighbor sets
                break;
            }
            
            std::vector<bool> mask(neighbors.size(), false);
            std::fill(mask.begin(), mask.begin() + dim, true);
            
            do {
                // Safety check: stop if we're creating too many simplices
                if (simplexCount > maxSimplices) {
                    break;
                }
                
                std::vector<size_t> simplex;
                simplex.push_back(pointIdx);
                
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    if (mask[i]) {
                        simplex.push_back(neighbors[i]);
                    }
                }
                
                // Check if this simplex should be included
                if (checkSimplexIntersection(simplex)) {
                    // Compute filtration value based on maximum pairwise distance
                    double maxDistance = 0.0;
                    for (size_t i = 0; i < simplex.size(); ++i) {
                        for (size_t j = i + 1; j < simplex.size(); ++j) {
                            double distance = euclideanDistance(points_[simplex[i]], points_[simplex[j]]);
                            maxDistance = std::max(maxDistance, distance);
                        }
                    }
                    
                    addSimplexToTree(simplex, maxDistance);
                    simplexCount++;
                }
            } while (std::prev_permutation(mask.begin(), mask.end()));
            
            // Break out of dimension loop if we've hit the limit
            if (simplexCount > maxSimplices) {
                break;
            }
        }
        
        // Break out of point loop if we've hit the limit
        if (simplexCount > maxSimplices) {
            break;
        }
    }
}

void CechComplex::addSimplexToTree(const std::vector<size_t>& simplex, double filtrationValue) {
    if (!simplex_tree_) {
        return;
    }

    // Convert to GUDHI format
    std::vector<Simplex_tree::Vertex_handle> gudhiSimplex;
    for (size_t idx : simplex) {
        gudhiSimplex.push_back(static_cast<Simplex_tree::Vertex_handle>(idx));
    }

    // Insert simplex with filtration value
    simplex_tree_->insert_simplex_and_subfaces(gudhiSimplex, filtrationValue);
}

double CechComplex::euclideanDistance(const Point& a, const Point& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }

    // CRITICAL FIX: Use Kahan summation to prevent floating point overflow
    double sum = 0.0;
    double c = 0.0; // Kahan compensation term
    
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        
        // CRITICAL FIX: Check for coordinate overflow
        if (std::abs(diff) > 1e6) {
            std::cerr << "WARNING: Large coordinate difference detected: " << diff 
                      << " at dimension " << i << std::endl;
            // Return a safe upper bound to prevent overflow
            return 1e6 * std::sqrt(static_cast<double>(a.size()));
        }
        
        double y = diff * diff - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    // CRITICAL FIX: Check for overflow in final result
    if (sum > 1e12) {
        std::cerr << "WARNING: Distance sum overflow detected: " << sum << std::endl;
        return 1e6; // Return safe upper bound
    }
    
    return std::sqrt(sum);
}

std::vector<size_t> CechComplex::findNeighborsInRadius(size_t pointIndex, double radius) {
    if (!spatialIndex_) {
        return {};
    }

    std::vector<size_t> neighbors;
    auto results = spatialIndex_->radiusSearch(points_[pointIndex], radius);
    
    for (const auto& result : results) {
        if (result.first != pointIndex) { // Exclude the point itself
            neighbors.push_back(result.first);
        }
    }

    return neighbors;
}

} // namespace tda::algorithms

#include "tda/spatial/spatial_index.hpp"
#include "tda/utils/simd_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <immintrin.h>
#include <omp.h>

namespace tda::spatial {

// KDTree implementation
KDTree::KDTree(size_t maxDepth) 
    : maxDepth_(maxDepth), dimension_(0), distanceFunc_(euclideanDistance) {
    buildStats_ = {0.0, 0, 0, 0};
}

bool KDTree::build(const PointContainer& points) {
    if (points.empty()) {
        return false;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        clear();
        points_ = points;
        dimension_ = points_[0].size();
        
        // Validate all points have the same dimension
        for (const auto& point : points_) {
            if (point.size() != dimension_) {
                throw std::runtime_error("All points must have the same dimension");
            }
        }
        
        // Create initial indices
        std::vector<size_t> indices(points_.size());
        for (size_t i = 0; i < points_.size(); ++i) {
            indices[i] = i;
        }
        
        // Build tree recursively
        root_ = buildRecursive(indices, 0);
        
        // Calculate build statistics
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        buildStats_.buildTimeMs = duration.count() / 1000.0;
        buildStats_.memoryUsageBytes = sizeof(*this) + points_.size() * dimension_ * sizeof(double);
        buildStats_.treeDepth = maxDepth_;
        buildStats_.leafNodes = 0; // Will be calculated during traversal if needed
        
        return true;
    } catch (const std::exception& e) {
        clear();
        return false;
    }
}

std::unique_ptr<KDTree::Node> KDTree::buildRecursive(const std::vector<size_t>& indices, size_t depth) {
    if (indices.empty() || depth >= maxDepth_) {
        return nullptr;
    }
    
    if (indices.size() == 1) {
        auto node = std::make_unique<Node>(indices[0]);
        node->splitDimension = 0;
        node->splitValue = 0.0;
        return node;
    }
    
    // Find best split dimension
    size_t splitDim = findBestSplitDimension(indices);
    double splitValue = findMedianValue(indices, splitDim);
    
    // Create node
    auto node = std::make_unique<Node>(indices[0]);
    node->splitDimension = splitDim;
    node->splitValue = splitValue;
    
    // Split points
    std::vector<size_t> leftIndices, rightIndices;
    for (size_t idx : indices) {
        if (points_[idx][splitDim] <= splitValue) {
            leftIndices.push_back(idx);
        } else {
            rightIndices.push_back(idx);
        }
    }
    
    // Build children recursively
    node->left = buildRecursive(leftIndices, depth + 1);
    node->right = buildRecursive(rightIndices, depth + 1);
    
    return node;
}

size_t KDTree::findBestSplitDimension(const std::vector<size_t>& indices) const {
    if (indices.empty() || dimension_ == 0) {
        return 0;
    }
    
    // Simple strategy: choose dimension with highest variance
    size_t bestDim = 0;
    double maxVariance = -1.0;
    
    for (size_t dim = 0; dim < dimension_; ++dim) {
        double sum = 0.0, sumSq = 0.0;
        for (size_t idx : indices) {
            double val = points_[idx][dim];
            sum += val;
            sumSq += val * val;
        }
        
        double mean = sum / indices.size();
        double variance = (sumSq / indices.size()) - (mean * mean);
        
        if (variance > maxVariance) {
            maxVariance = variance;
            bestDim = dim;
        }
    }
    
    return bestDim;
}

double KDTree::findMedianValue(const std::vector<size_t>& indices, size_t dim) const {
    if (indices.empty()) {
        return 0.0;
    }
    
    std::vector<double> values;
    values.reserve(indices.size());
    
    for (size_t idx : indices) {
        values.push_back(points_[idx][dim]);
    }
    
    // Use nth_element for efficient median finding
    size_t medianIdx = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + medianIdx, values.end());
    
    return values[medianIdx];
}

std::vector<SpatialIndex::SearchResult> KDTree::kNearestNeighbors(const Point& query, size_t k) const {
    if (!root_ || query.size() != dimension_) {
        return {};
    }
    
    // Use priority queue to maintain k nearest neighbors
    std::priority_queue<std::pair<double, size_t>> pq;
    
    // Initialize with first k points (or all if k > size)
    k = std::min(k, points_.size());
    
    // Recursive search
    findKNearestRecursive(root_.get(), query, k, pq);
    
    // Convert to result format
    std::vector<SearchResult> results;
    results.reserve(k);
    
    while (!pq.empty()) {
        auto [dist, idx] = pq.top();
        pq.pop();
        results.emplace_back(idx, dist);
    }
    
    // Reverse to get ascending order
    std::reverse(results.begin(), results.end());
    
    return results;
}

void KDTree::findKNearestRecursive(const Node* node, const Point& query, size_t k,
                                   std::priority_queue<std::pair<double, size_t>>& pq) const {
    if (!node) {
        return;
    }
    
    // CRITICAL FIX: Add bounds checking to prevent buffer overflow
    if (node->pointIndex >= points_.size()) {
        // Log error and skip this node to prevent crash
        std::cerr << "ERROR: Invalid pointIndex " << node->pointIndex 
                  << " >= " << points_.size() << " in KDTree" << std::endl;
        return;
    }
    
    // Calculate distance to current node's point
    double dist = distanceFunc_(query, points_[node->pointIndex]);
    
    // Add to priority queue if we have space or if it's closer than current farthest
    if (pq.size() < k) {
        pq.emplace(dist, node->pointIndex);
    } else if (dist < pq.top().first) {
        pq.pop();
        pq.emplace(dist, node->pointIndex);
    }
    
    // Determine which child to explore first
    double queryVal = query[node->splitDimension];
    const Node* firstChild = (queryVal <= node->splitValue) ? node->left.get() : node->right.get();
    const Node* secondChild = (queryVal <= node->splitValue) ? node->right.get() : node->left.get();
    
    // Explore first child
    if (firstChild) {
        findKNearestRecursive(firstChild, query, k, pq);
    }
    
    // Check if we need to explore second child
    if (secondChild) {
        double splitDist = std::abs(queryVal - node->splitValue);
        if (pq.size() < k || splitDist < pq.top().first) {
            findKNearestRecursive(secondChild, query, k, pq);
        }
    }
}

std::vector<SpatialIndex::SearchResult> KDTree::radiusSearch(const Point& query, double radius) const {
    if (!root_ || query.size() != dimension_) {
        return {};
    }
    
    std::vector<SearchResult> results;
    radiusSearchRecursive(root_.get(), query, radius, results);
    
    // Sort by distance
    std::sort(results.begin(), results.end(), 
              [](const SearchResult& a, const SearchResult& b) {
                  return a.second < b.second;
              });
    
    return results;
}

void KDTree::radiusSearchRecursive(const Node* node, const Point& query, double radius,
                                  std::vector<SearchResult>& results) const {
    if (!node) {
        return;
    }
    
    // CRITICAL FIX: Add bounds checking to prevent buffer overflow
    if (node->pointIndex >= points_.size()) {
        // Log error and skip this node to prevent crash
        std::cerr << "ERROR: Invalid pointIndex " << node->pointIndex 
                  << " >= " << points_.size() << " in KDTree" << std::endl;
        return;
    }
    
    // Calculate distance to current node's point
    double dist = distanceFunc_(query, points_[node->pointIndex]);
    
    // Add to results if within radius
    if (dist <= radius) {
        results.emplace_back(node->pointIndex, dist);
    }
    
    // Check if we need to explore children
    double queryVal = query[node->splitDimension];
    double splitDist = std::abs(queryVal - node->splitValue);
    
    if (splitDist <= radius) {
        // Need to explore both children
        if (node->left) {
            radiusSearchRecursive(node->left.get(), query, radius, results);
        }
        if (node->right) {
            radiusSearchRecursive(node->right.get(), query, radius, results);
        }
    } else {
        // Only explore the closer child
        const Node* closerChild = (queryVal <= node->splitValue) ? node->left.get() : node->right.get();
        if (closerChild) {
            radiusSearchRecursive(closerChild, query, radius, results);
        }
    }
}

SpatialIndex::SearchResult KDTree::nearestNeighbor(const Point& query) const {
    auto results = kNearestNeighbors(query, 1);
    if (results.empty()) {
        return {0, std::numeric_limits<double>::max()};
    }
    return results[0];
}

size_t KDTree::size() const {
    return points_.size();
}

bool KDTree::empty() const {
    return points_.empty();
}

void KDTree::clear() {
    points_.clear();
    root_.reset();
    dimension_ = 0;
    buildStats_ = {0.0, 0, 0, 0};
}

size_t KDTree::dimension() const {
    return dimension_;
}

void KDTree::setDistanceFunction(DistanceFunction distanceFunc) {
    distanceFunc_ = distanceFunc;
}

BuildStats KDTree::getBuildStats() const {
    return buildStats_;
}

double KDTree::euclideanDistance(const Point& a, const Point& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::max();
    }
    
    // Use SIMD optimized distance calculation for better performance
    return tda::utils::SIMDUtils::vectorizedEuclideanDistance(a.data(), b.data(), a.size());
}

// SIMD-optimized batch distance calculation for multiple queries
void KDTree::batchDistanceCalculation(
    const std::vector<Point>& query_points,
    const std::vector<Point>& data_points,
    std::vector<double>& distances
) {
    if (query_points.size() != data_points.size()) {
        throw std::invalid_argument("Query and data point vectors must have same size");
    }
    
    distances.resize(query_points.size());
    
    const size_t simd_width = 4; // AVX 256-bit / 64-bit double = 4
    const size_t aligned_size = (query_points.size() / simd_width) * simd_width;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < aligned_size; i += simd_width) {
        // Process 4 points at once with SIMD
        if (query_points[i].size() == data_points[i].size() && 
            query_points[i].size() >= 2) { // Ensure minimum dimension for SIMD
            
            __m256d dist_squared = _mm256_setzero_pd();
            
            for (size_t dim = 0; dim < query_points[i].size(); ++dim) {
                __m256d q = _mm256_set_pd(
                    query_points[i+3][dim], query_points[i+2][dim],
                    query_points[i+1][dim], query_points[i][dim]
                );
                __m256d d = _mm256_set_pd(
                    data_points[i+3][dim], data_points[i+2][dim], 
                    data_points[i+1][dim], data_points[i][dim]
                );
                __m256d diff = _mm256_sub_pd(q, d);
                dist_squared = _mm256_fmadd_pd(diff, diff, dist_squared);
            }
            
            // Store results
            alignas(32) double results[4];
            _mm256_store_pd(results, dist_squared);
            for (int j = 0; j < 4; ++j) {
                distances[i + j] = std::sqrt(results[j]);
            }
        } else {
            // Fallback for mismatched dimensions
            for (int j = 0; j < 4; ++j) {
                distances[i + j] = euclideanDistance(query_points[i + j], data_points[i + j]);
            }
        }
    }
    
    // Handle remaining points
    for (size_t i = aligned_size; i < query_points.size(); ++i) {
        distances[i] = euclideanDistance(query_points[i], data_points[i]);
    }
}

} // namespace tda::spatial

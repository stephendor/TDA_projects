#include "tda/spatial/spatial_index.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace tda::spatial {

// BallTree implementation
BallTree::BallTree(size_t maxLeafSize) 
    : maxLeafSize_(maxLeafSize), dimension_(0), distanceFunc_(euclideanDistance) {
    buildStats_ = {0.0, 0, 0, 0};
}

bool BallTree::build(const PointContainer& points) {
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
        root_ = buildRecursive(indices);
        
        // Calculate build statistics
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        buildStats_.buildTimeMs = duration.count() / 1000.0;
        buildStats_.memoryUsageBytes = sizeof(*this) + points_.size() * dimension_ * sizeof(double);
        buildStats_.treeDepth = 0; // Will be calculated during traversal if needed
        buildStats_.leafNodes = 0; // Will be calculated during traversal if needed
        
        return true;
    } catch (const std::exception& e) {
        clear();
        return false;
    }
}

std::unique_ptr<BallTree::BallNode> BallTree::buildRecursive(const std::vector<size_t>& indices) {
    if (indices.empty()) {
        return nullptr;
    }
    
    auto node = std::make_unique<BallNode>();
    
    if (indices.size() <= maxLeafSize_) {
        // Leaf node
        node->pointIndices = indices;
        node->center = computeCentroid(indices);
        node->radius = computeRadius(indices, node->center);
        return node;
    }
    
    // Split points and create internal node
    auto [leftIndices, rightIndices] = splitPoints(indices);
    
    node->center = computeCentroid(indices);
    node->radius = computeRadius(indices, node->center);
    
    // Build children recursively
    node->left = buildRecursive(leftIndices);
    node->right = buildRecursive(rightIndices);
    
    return node;
}

BallTree::Point BallTree::computeCentroid(const std::vector<size_t>& indices) const {
    if (indices.empty()) {
        return Point(dimension_, 0.0);
    }
    
    Point centroid(dimension_, 0.0);
    
    for (size_t idx : indices) {
        for (size_t dim = 0; dim < dimension_; ++dim) {
            centroid[dim] += points_[idx][dim];
        }
    }
    
    for (size_t dim = 0; dim < dimension_; ++dim) {
        centroid[dim] /= indices.size();
    }
    
    return centroid;
}

double BallTree::computeRadius(const std::vector<size_t>& indices, const Point& center) const {
    if (indices.empty()) {
        return 0.0;
    }
    
    double maxRadius = 0.0;
    
    for (size_t idx : indices) {
        double dist = distanceFunc_(center, points_[idx]);
        maxRadius = std::max(maxRadius, dist);
    }
    
    return maxRadius;
}

std::pair<std::vector<size_t>, std::vector<size_t>> BallTree::splitPoints(const std::vector<size_t>& indices) const {
    if (indices.size() < 2) {
        return {{indices}, {}};
    }
    
    // Find the two points farthest apart
    size_t idx1 = indices[0], idx2 = indices[1];
    double maxDist = distanceFunc_(points_[idx1], points_[idx2]);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = i + 1; j < indices.size(); ++j) {
            double dist = distanceFunc_(points_[indices[i]], points_[indices[j]]);
            if (dist > maxDist) {
                maxDist = dist;
                idx1 = indices[i];
                idx2 = indices[j];
            }
        }
    }
    
    // Split points based on distance to these two points
    std::vector<size_t> leftIndices, rightIndices;
    
    for (size_t idx : indices) {
        double dist1 = distanceFunc_(points_[idx], points_[idx1]);
        double dist2 = distanceFunc_(points_[idx], points_[idx2]);
        
        if (dist1 <= dist2) {
            leftIndices.push_back(idx);
        } else {
            rightIndices.push_back(idx);
        }
    }
    
    // Ensure both groups have at least one point
    if (leftIndices.empty()) {
        leftIndices.push_back(rightIndices.back());
        rightIndices.pop_back();
    } else if (rightIndices.empty()) {
        rightIndices.push_back(leftIndices.back());
        leftIndices.pop_back();
    }
    
    return {leftIndices, rightIndices};
}

std::vector<SpatialIndex::SearchResult> BallTree::kNearestNeighbors(const Point& query, size_t k) const {
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

void BallTree::findKNearestRecursive(const BallNode* node, const Point& query, size_t k,
                                     std::priority_queue<std::pair<double, size_t>>& pq) const {
    if (!node) {
        return;
    }
    
    // Check if this node could contain closer points
    double distToCenter = distanceFunc_(query, node->center);
    double minDist = std::max(0.0, distToCenter - node->radius);
    
    // If the minimum possible distance is greater than our current k-th best, skip this node
    if (!pq.empty() && pq.size() >= k && minDist >= pq.top().first) {
        return;
    }
    
    if (node->left == nullptr && node->right == nullptr) {
        // Leaf node - check all points
        for (size_t idx : node->pointIndices) {
            double dist = distanceFunc_(query, points_[idx]);
            
            if (pq.size() < k) {
                pq.emplace(dist, idx);
            } else if (dist < pq.top().first) {
                pq.pop();
                pq.emplace(dist, idx);
            }
        }
    } else {
        // Internal node - explore children
        if (node->left) {
            findKNearestRecursive(node->left.get(), query, k, pq);
        }
        if (node->right) {
            findKNearestRecursive(node->right.get(), query, k, pq);
        }
    }
}

std::vector<SpatialIndex::SearchResult> BallTree::radiusSearch(const Point& query, double radius) const {
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

void BallTree::radiusSearchRecursive(const BallNode* node, const Point& query, double radius,
                                    std::vector<SearchResult>& results) const {
    if (!node) {
        return;
    }
    
    // Check if this node could contain points within radius
    double distToCenter = distanceFunc_(query, node->center);
    double minDist = std::max(0.0, distToCenter - node->radius);
    
    // If the minimum possible distance is greater than radius, skip this node
    if (minDist > radius) {
        return;
    }
    
    if (node->left == nullptr && node->right == nullptr) {
        // Leaf node - check all points
        for (size_t idx : node->pointIndices) {
            double dist = distanceFunc_(query, points_[idx]);
            if (dist <= radius) {
                results.emplace_back(idx, dist);
            }
        }
    } else {
        // Internal node - explore children
        if (node->left) {
            radiusSearchRecursive(node->left.get(), query, radius, results);
        }
        if (node->right) {
            radiusSearchRecursive(node->right.get(), query, radius, results);
        }
    }
}

SpatialIndex::SearchResult BallTree::nearestNeighbor(const Point& query) const {
    auto results = kNearestNeighbors(query, 1);
    if (results.empty()) {
        return {0, std::numeric_limits<double>::max()};
    }
    return results[0];
}

size_t BallTree::size() const {
    return points_.size();
}

bool BallTree::empty() const {
    return points_.empty();
}

void BallTree::clear() {
    points_.clear();
    root_.reset();
    dimension_ = 0;
    buildStats_ = {0.0, 0, 0, 0};
}

size_t BallTree::dimension() const {
    return dimension_;
}

void BallTree::setDistanceFunction(DistanceFunction distanceFunc) {
    distanceFunc_ = distanceFunc;
}

BallTree::BuildStats BallTree::getBuildStats() const {
    return buildStats_;
}

double BallTree::euclideanDistance(const Point& a, const Point& b) {
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

} // namespace tda::spatial

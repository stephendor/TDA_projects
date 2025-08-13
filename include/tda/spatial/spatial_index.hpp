#pragma once

#include "tda/core/types.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <cstddef>
#include <queue> // Added for priority_queue

namespace tda::spatial {

/**
 * @brief Build statistics structure for spatial indices
 */
struct BuildStats {
    double buildTimeMs;
    size_t memoryUsageBytes;
    size_t treeDepth;
    size_t leafNodes;
};

/**
 * @brief Abstract base class for spatial indexing structures
 * 
 * Provides a common interface for different spatial indexing implementations
 * (KD-trees, ball trees, etc.) to support efficient nearest neighbor search.
 */
class SpatialIndex {
public:
    using Point = std::vector<double>;
    using PointContainer = std::vector<Point>;
    using DistanceFunction = std::function<double(const Point&, const Point&)>;
    using SearchResult = std::pair<size_t, double>; // (index, distance)
    
    virtual ~SpatialIndex() = default;
    
    /**
     * @brief Build the spatial index from a point cloud
     * @param points The point cloud to index
     * @return true if successful, false otherwise
     */
    virtual bool build(const PointContainer& points) = 0;
    
    /**
     * @brief Find the k nearest neighbors to a query point
     * @param query The query point
     * @param k Number of neighbors to find
     * @return Vector of (index, distance) pairs, sorted by distance
     */
    virtual std::vector<SearchResult> kNearestNeighbors(const Point& query, size_t k) const = 0;
    
    /**
     * @brief Find all points within a given radius of a query point
     * @param query The query point
     * @param radius Search radius
     * @return Vector of (index, distance) pairs within the radius
     */
    virtual std::vector<SearchResult> radiusSearch(const Point& query, double radius) const = 0;
    
    /**
     * @brief Find the single nearest neighbor to a query point
     * @param query The query point
     * @return (index, distance) pair of the nearest neighbor
     */
    virtual SearchResult nearestNeighbor(const Point& query) const = 0;
    
    /**
     * @brief Get the number of points in the index
     * @return Number of indexed points
     */
    virtual size_t size() const = 0;
    
    /**
     * @brief Check if the index is empty
     * @return true if empty, false otherwise
     */
    virtual bool empty() const = 0;
    
    /**
     * @brief Clear the index
     */
    virtual void clear() = 0;
    
    /**
     * @brief Get the dimension of the indexed points
     * @return Point dimension
     */
    virtual size_t dimension() const = 0;
};

/**
 * @brief KD-tree implementation for efficient nearest neighbor search
 * 
 * Optimized for low-dimensional spaces (2D, 3D) with O(log n) average
 * search time for balanced trees.
 */
class KDTree : public SpatialIndex {
public:
    explicit KDTree(size_t maxDepth = 20);
    ~KDTree() override = default;
    
    bool build(const PointContainer& points) override;
    std::vector<SearchResult> kNearestNeighbors(const Point& query, size_t k) const override;
    std::vector<SearchResult> radiusSearch(const Point& query, double radius) const override;
    SearchResult nearestNeighbor(const Point& query) const override;
    size_t size() const override;
    bool empty() const override;
    void clear() override;
    size_t dimension() const override;
    
    /**
     * @brief Set the distance function to use
     * @param distanceFunc Custom distance function
     */
    void setDistanceFunction(DistanceFunction distanceFunc);
    
    /**
     * @brief Get build statistics
     * @return Build time and memory usage information
     */
    BuildStats getBuildStats() const;
    
    /**
     * @brief SIMD-optimized batch distance calculation for multiple query points
     * @param query_points Vector of query points
     * @param data_points Vector of data points (same size as query_points)
     * @param distances Output vector for computed distances
     */
    static void batchDistanceCalculation(
        const std::vector<Point>& query_points,
        const std::vector<Point>& data_points,
        std::vector<double>& distances
    );

private:
    struct Node {
        size_t pointIndex;
        size_t splitDimension;
        double splitValue;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        
        Node(size_t idx) : pointIndex(idx), splitDimension(0), splitValue(0.0) {}
    };
    
    PointContainer points_;
    std::unique_ptr<Node> root_;
    size_t maxDepth_;
    size_t dimension_;
    DistanceFunction distanceFunc_;
    BuildStats buildStats_;
    
    std::unique_ptr<Node> buildRecursive(const std::vector<size_t>& indices, size_t depth);
    void findKNearestRecursive(const Node* node, const Point& query, size_t k, 
                               std::priority_queue<std::pair<double, size_t>>& pq) const;
    void radiusSearchRecursive(const Node* node, const Point& query, double radius,
                              std::vector<SearchResult>& results) const;
    size_t findBestSplitDimension(const std::vector<size_t>& indices) const;
    double findMedianValue(const std::vector<size_t>& indices, size_t dim) const;
    
    // Default distance function (Euclidean)
    static double euclideanDistance(const Point& a, const Point& b);
};

/**
 * @brief Ball Tree implementation for efficient nearest neighbor search
 * 
 * Better suited for high-dimensional spaces compared to KD-trees, with
 * O(log n) average search time and better performance scaling.
 */
class BallTree : public SpatialIndex {
public:
    explicit BallTree(size_t maxLeafSize = 20);
    ~BallTree() override = default;
    
    bool build(const PointContainer& points) override;
    std::vector<SearchResult> kNearestNeighbors(const Point& query, size_t k) const override;
    std::vector<SearchResult> radiusSearch(const Point& query, double radius) const override;
    SearchResult nearestNeighbor(const Point& query) const override;
    size_t size() const override;
    bool empty() const override;
    size_t dimension() const override;
    
    void clear();
    BuildStats getStatistics() const;
    void setDistanceFunction(DistanceFunction distFunc) { distanceFunc_ = distFunc; }

private:
    struct BallNode {
        Point center;
        double radius = 0.0;
        std::vector<size_t> pointIndices;  // For leaf nodes
        std::unique_ptr<BallNode> left;
        std::unique_ptr<BallNode> right;
        
        bool isLeaf() const { return !pointIndices.empty(); }
    };
    
    PointContainer points_;
    std::unique_ptr<BallNode> root_;
    size_t maxLeafSize_;
    size_t dimension_;
    DistanceFunction distanceFunc_;
    BuildStats buildStats_;
    
    std::unique_ptr<BallNode> buildRecursive(const std::vector<size_t>& indices);
    void findKNearestRecursive(const BallNode* node, const Point& query, size_t k, 
                               std::priority_queue<std::pair<double, size_t>>& pq) const;
    void radiusSearchRecursive(const BallNode* node, const Point& query, double radius,
                              std::vector<SearchResult>& results) const;
    Point computeCentroid(const std::vector<size_t>& indices) const;
    double computeRadius(const std::vector<size_t>& indices, const Point& center) const;
    std::pair<std::vector<size_t>, std::vector<size_t>> splitPoints(const std::vector<size_t>& indices) const;
    
    // Default distance function (Euclidean)
    static double euclideanDistance(const Point& a, const Point& b);
};

/**
 * @brief Factory function to create the most appropriate spatial index
 * 
 * Automatically selects between KD-tree and ball tree based on data characteristics
 * @param points The point cloud to index
 * @param maxDimension Threshold for choosing ball tree over KD-tree
 * @return Unique pointer to the created spatial index
 */
std::unique_ptr<SpatialIndex> createSpatialIndex(const std::vector<std::vector<double>>& points, size_t maxDimension = 10);

} // namespace tda::spatial

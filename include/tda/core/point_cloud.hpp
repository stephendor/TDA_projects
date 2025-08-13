#pragma once

#include "types.hpp"
#include <vector>
#include <memory>

namespace tda::core {

/**
 * @brief Point cloud data structure for TDA computations
 * 
 * Represents a collection of points in n-dimensional space.
 * Provides efficient storage and access methods for topological analysis.
 */
class PointCloud {
public:
    using Point = std::vector<double>;
    using PointContainer = std::vector<Point>;
    
    PointCloud() = default;
    explicit PointCloud(PointContainer points);
    
    // Copy constructor and assignment
    PointCloud(const PointCloud&) = default;
    PointCloud& operator=(const PointCloud&) = default;
    
    // Move constructor and assignment
    PointCloud(PointCloud&&) = default;
    PointCloud& operator=(PointCloud&&) = default;
    
    ~PointCloud() = default;
    
    // Access methods
    const Point& operator[](size_t index) const;
    Point& operator[](size_t index);
    
    // Size and capacity
    size_t size() const noexcept;
    size_t dimension() const noexcept;
    bool empty() const noexcept;
    
    // Iterators
    PointContainer::const_iterator begin() const noexcept;
    PointContainer::const_iterator end() const noexcept;
    PointContainer::iterator begin() noexcept;
    PointContainer::iterator end() noexcept;
    
    // Data access
    const PointContainer& points() const noexcept;
    PointContainer& points() noexcept;
    
    // Validation
    bool isValid() const;
    
    // Utility methods
    void addPoint(const Point& point);
    void addPoint(Point&& point);
    void clear();
    void reserve(size_t capacity);
    
    // Distance computation
    double compute_distance(size_t i, size_t j) const;

private:
    PointContainer points_;
    bool valid_ = true;
};

} // namespace tda::core

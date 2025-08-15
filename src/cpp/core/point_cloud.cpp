#include "../../../include/tda/core/point_cloud.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tda::core {

PointCloud::PointCloud(PointContainer points) 
    : points_(std::move(points)), valid_(true) {
    if (!points_.empty()) {
        // Validate that all points have the same dimension
        size_t dim = points_[0].size();
        for (const auto& point : points_) {
            if (point.size() != dim) {
                valid_ = false;
                break;
            }
        }
    }
}

const PointCloud::Point& PointCloud::operator[](size_t index) const {
    if (index >= points_.size()) {
        throw std::out_of_range("PointCloud index out of range");
    }
    return points_[index];
}

PointCloud::Point& PointCloud::operator[](size_t index) {
    if (index >= points_.size()) {
        throw std::out_of_range("PointCloud index out of range");
    }
    return points_[index];
}

size_t PointCloud::size() const noexcept {
    return points_.size();
}

size_t PointCloud::dimension() const noexcept {
    if (points_.empty()) return 0;
    return points_[0].size();
}

bool PointCloud::empty() const noexcept {
    return points_.empty();
}

PointCloud::PointContainer::const_iterator PointCloud::begin() const noexcept {
    return points_.begin();
}

PointCloud::PointContainer::iterator PointCloud::begin() noexcept {
    return points_.begin();
}

PointCloud::PointContainer::const_iterator PointCloud::end() const noexcept {
    return points_.end();
}

PointCloud::PointContainer::iterator PointCloud::end() noexcept {
    return points_.end();
}

const PointCloud::PointContainer& PointCloud::points() const noexcept {
    return points_;
}

PointCloud::PointContainer& PointCloud::points() noexcept {
    return points_;
}

bool PointCloud::isValid() const {
    return valid_;
}

void PointCloud::addPoint(const Point& point) {
    if (!points_.empty() && point.size() != dimension()) {
        throw std::invalid_argument("Point dimension mismatch");
    }
    points_.push_back(point);
}

void PointCloud::addPoint(Point&& point) {
    if (!points_.empty() && point.size() != dimension()) {
        throw std::invalid_argument("Point dimension mismatch");
    }
    points_.push_back(std::move(point));
}

void PointCloud::clear() {
    points_.clear();
}

void PointCloud::reserve(size_t capacity) {
    points_.reserve(capacity);
}

double PointCloud::compute_distance(size_t i, size_t j) const {
    if (i >= points_.size() || j >= points_.size()) {
        return 0.0;
    }
    
    const auto& p1 = points_[i];
    const auto& p2 = points_[j];
    
    double sum = 0.0;
    for (size_t k = 0; k < p1.size(); ++k) {
        double diff = p1[k] - p2[k];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

} // namespace tda::core
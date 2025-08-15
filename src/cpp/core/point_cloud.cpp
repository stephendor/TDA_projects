#include "../../../include/tda/core/types.hpp"
#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <stdexcept>

namespace tda::core {

class PointCloud {
private:
    std::vector<DynamicVector> points_;
    std::size_t dimension_;
    std::size_t capacity_;
    
    // Cache for computed properties
    mutable std::optional<std::vector<double>> distances_cache_;
    mutable std::optional<std::vector<double>> angles_cache_;
    mutable std::optional<std::vector<double>> norms_cache_;
    
public:
    // Type aliases
    using point_type = DynamicVector;
    using size_type = std::size_t;
    using iterator = std::vector<DynamicVector>::iterator;
    using const_iterator = std::vector<DynamicVector>::const_iterator;
    
    // Constructors
    PointCloud() = default;
    
    explicit PointCloud(size_type dimension) 
        : dimension_(dimension), capacity_(0) {}
    
    PointCloud(size_type dimension, size_type capacity)
        : dimension_(dimension), capacity_(capacity) {
        points_.reserve(capacity);
    }
    
    // Copy constructor
    PointCloud(const PointCloud& other)
        : points_(other.points_), dimension_(other.dimension_), capacity_(other.capacity_) {}
    
    // Move constructor
    PointCloud(PointCloud&& other) noexcept
        : points_(std::move(other.points_)), dimension_(other.dimension_), capacity_(other.capacity_) {
        other.dimension_ = 0;
        other.capacity_ = 0;
    }
    
    // Destructor
    ~PointCloud() = default;
    
    // Assignment operators
    PointCloud& operator=(const PointCloud& other) {
        if (this != &other) {
            points_ = other.points_;
            dimension_ = other.dimension_;
            capacity_ = other.capacity_;
            clear_cache();
        }
        return *this;
    }
    
    PointCloud& operator=(PointCloud&& other) noexcept {
        if (this != &other) {
            points_ = std::move(other.points_);
            dimension_ = other.dimension_;
            capacity_ = other.capacity_;
            other.dimension_ = 0;
            other.capacity_ = 0;
            clear_cache();
        }
        return *this;
    }
    
    // Element access
    point_type& operator[](size_type index) {
        return points_[index];
    }
    
    const point_type& operator[](size_type index) const {
        return points_[index];
    }
    
    point_type& at(size_type index) {
        if (index >= points_.size()) {
            throw std::out_of_range("PointCloud index out of range");
        }
        return points_[index];
    }
    
    const point_type& at(size_type index) const {
        if (index >= points_.size()) {
            throw std::out_of_range("PointCloud index out of range");
        }
        return points_[index];
    }
    
    // Capacity
    size_type size() const noexcept { return points_.size(); }
    size_type dimension() const noexcept { return dimension_; }
    size_type capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return points_.empty(); }
    
    // Iterators
    iterator begin() noexcept { return points_.begin(); }
    const_iterator begin() const noexcept { return points_.begin(); }
    const_iterator cbegin() const noexcept { return points_.cbegin(); }
    
    iterator end() noexcept { return points_.end(); }
    const_iterator end() const noexcept { return points_.end(); }
    const_iterator cend() const noexcept { return points_.cend(); }
    
    // Data access
    std::vector<DynamicVector>& data() noexcept { return points_; }
    const std::vector<DynamicVector>& data() const noexcept { return points_; }
    
    // Point management
    void add_point(const point_type& point) {
        if (point.size() != dimension_) {
            throw std::invalid_argument("Point dimension mismatch");
        }
        points_.push_back(point);
        clear_cache();
    }
    
    void add_point(point_type&& point) {
        if (point.size() != dimension_) {
            throw std::invalid_argument("Point dimension mismatch");
        }
        points_.push_back(std::move(point));
        clear_cache();
    }
    
    template<typename Container>
    void add_points(const Container& container) {
        // Concept to check if Container is iterable and contains elements convertible to point_type
        static_assert(requires(const Container& c) {
            { c.begin() } -> std::input_iterator;
            { c.end() } -> std::input_iterator;
            requires std::convertible_to<decltype(*c.begin()), point_type>;
        }, "Container must be iterable and contain convertible elements");
        
        for (const auto& point : container) {
            add_point(point);
        }
    }
    
    void remove_point(size_type index) {
        if (index >= points_.size()) {
            throw std::out_of_range("PointCloud index out of range");
        }
        points_.erase(points_.begin() + index);
        clear_cache();
    }
    
    void clear() noexcept {
        points_.clear();
        clear_cache();
    }
    
    // Resize operations
    void resize(size_type new_size) {
        points_.resize(new_size);
        clear_cache();
    }
    
    void resize(size_type new_size, const point_type& value) {
        points_.resize(new_size, value);
        clear_cache();
    }
    
    void reserve(size_type new_capacity) {
        capacity_ = new_capacity;
        points_.reserve(new_capacity);
    }
    
    // Geometric computations
    double compute_distance(size_type i, size_type j) const {
        if (i >= points_.size() || j >= points_.size()) {
            throw std::out_of_range("PointCloud index out of range");
        }
        
        const auto& p1 = points_[i];
        const auto& p2 = points_[j];
        
        double sum = 0.0;
        for (size_type k = 0; k < dimension_; ++k) {
            double diff = p1[k] - p2[k];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    std::vector<double> compute_distances() const {
        if (!distances_cache_.has_value()) {
            size_type n = points_.size();
            size_type total_pairs = n * (n - 1) / 2;
            std::vector<double> distances;
            distances.reserve(total_pairs);
            
            for (size_type i = 0; i < n; ++i) {
                for (size_type j = i + 1; j < n; ++j) {
                    distances.push_back(compute_distance(i, j));
                }
            }
            
            distances_cache_ = std::move(distances);
        }
        return *distances_cache_;
    }
    
    std::vector<double> compute_angles() const {
        if (!angles_cache_.has_value()) {
            size_type n = points_.size();
            std::vector<double> angles;
            angles.reserve(n * (n - 1) * (n - 2) / 6);
            
            for (size_type i = 0; i < n; ++i) {
                for (size_type j = i + 1; j < n; ++j) {
                    for (size_type k = j + 1; k < n; ++k) {
                        angles.push_back(compute_angle(i, j, k));
                    }
                }
            }
            
            angles_cache_ = std::move(angles);
        }
        return *angles_cache_;
    }
    
    std::vector<double> compute_norms() const {
        if (!norms_cache_.has_value()) {
            std::vector<double> norms;
            norms.reserve(points_.size());
            
            // Use regular transform instead of parallel execution for now
            std::transform(points_.begin(), points_.end(), 
                          std::back_inserter(norms),
                          [this](const auto& point) {
                              return compute_norm(point);
                          });
            
            norms_cache_ = std::move(norms);
        }
        return *norms_cache_;
    }
    
    // Statistical computations
    point_type compute_centroid() const {
        if (points_.empty()) {
            return point_type{};
        }
        
        point_type centroid(dimension_, 0.0);
        for (const auto& point : points_) {
            for (size_type i = 0; i < dimension_; ++i) {
                centroid[i] += point[i];
            }
        }
        
        for (double& coord : centroid) {
            coord /= static_cast<double>(points_.size());
        }
        
        return centroid;
    }
    
    point_type compute_variance() const {
        if (points_.size() < 2) {
            return point_type(dimension_, 0.0);
        }
        
        point_type centroid = compute_centroid();
        point_type variance(dimension_, 0.0);
        
        for (const auto& point : points_) {
            for (size_type i = 0; i < dimension_; ++i) {
                double diff = point[i] - centroid[i];
                variance[i] += diff * diff;
            }
        }
        
        for (double& var : variance) {
            var /= static_cast<double>(points_.size() - 1);
        }
        
        return variance;
    }
    
    // Utility methods
    void normalize() {
        point_type centroid = compute_centroid();
        point_type std_dev = compute_variance();
        
        for (auto& point : points_) {
            for (size_type i = 0; i < dimension_; ++i) {
                if (std_dev[i] > 0) {
                    point[i] = (point[i] - centroid[i]) / std::sqrt(std_dev[i]);
                }
            }
        }
        clear_cache();
    }
    
    void center() {
        point_type centroid = compute_centroid();
        
        for (auto& point : points_) {
            for (size_type i = 0; i < dimension_; ++i) {
                point[i] -= centroid[i];
            }
        }
        clear_cache();
    }
    
    // Serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(points_, dimension_, capacity_);
    }
    
private:
    double compute_angle(size_type i, size_type j, size_type k) const {
        const auto& p1 = points_[i];
        const auto& p2 = points_[j];
        const auto& p3 = points_[k];
        
        // Compute vectors
        point_type v1(dimension_), v2(dimension_);
        for (size_type d = 0; d < dimension_; ++d) {
            v1[d] = p2[d] - p1[d];
            v2[d] = p3[d] - p1[d];
        }
        
        // Compute dot product and norms
        double dot_product = 0.0;
        double norm1 = 0.0, norm2 = 0.0;
        
        for (size_type d = 0; d < dimension_; ++d) {
            dot_product += v1[d] * v2[d];
            norm1 += v1[d] * v1[d];
            norm2 += v2[d] * v2[d];
        }
        
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }
        
        double cos_angle = dot_product / (norm1 * norm2);
        cos_angle = std::clamp(cos_angle, -1.0, 1.0);
        return std::acos(cos_angle);
    }
    
    double compute_norm(const point_type& point) const {
        double sum = 0.0;
        for (double coord : point) {
            sum += coord * coord;
        }
        return std::sqrt(sum);
    }
    
    void clear_cache() const noexcept {
        distances_cache_.reset();
        angles_cache_.reset();
        norms_cache_.reset();
    }
};

} // namespace tda::core
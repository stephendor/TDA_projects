#include "../../../include/tda/vector_stack/vector_stack.hpp"
#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <stdexcept>

namespace tda::vector_stack {

// Mathematical operations
template<typename T>
VectorStack<T> VectorStack<T>::operator+(const VectorStack& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("VectorStack dimensions must match for addition");
    }
    
    VectorStack result(rows_, cols_);
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  other.data_.begin(), result.data_.begin(), std::plus<T>());
    return result;
}

template<typename T>
VectorStack<T> VectorStack<T>::operator-(const VectorStack& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("VectorStack dimensions must match for subtraction");
    }
    
    VectorStack result(rows_, cols_);
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  other.data_.begin(), result.data_.begin(), std::minus<T>());
    return result;
}

template<typename T>
VectorStack<T> VectorStack<T>::operator*(const T& scalar) const {
    VectorStack result(rows_, cols_);
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  result.data_.begin(), [scalar](T val) { return val * scalar; });
    return result;
}

template<typename T>
VectorStack<T> VectorStack<T>::operator/(const T& scalar) const {
    if (scalar == T{0}) {
        throw std::invalid_argument("Division by zero");
    }
    
    VectorStack result(rows_, cols_);
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  result.data_.begin(), [scalar](T val) { return val / scalar; });
    return result;
}

template<typename T>
VectorStack<T>& VectorStack<T>::operator+=(const VectorStack& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("VectorStack dimensions must match for addition");
    }
    
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  other.data_.begin(), data_.begin(), std::plus<T>());
    return *this;
}

template<typename T>
VectorStack<T>& VectorStack<T>::operator-=(const VectorStack& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("VectorStack dimensions must match for subtraction");
    }
    
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  other.data_.begin(), data_.begin(), std::minus<T>());
    return *this;
}

template<typename T>
VectorStack<T>& VectorStack<T>::operator*=(const T& scalar) {
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  data_.begin(), [scalar](T val) { return val * scalar; });
    return *this;
}

template<typename T>
VectorStack<T>& VectorStack<T>::operator/=(const T& scalar) {
    if (scalar == T{0}) {
        throw std::invalid_argument("Division by zero");
    }
    
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), 
                  data_.begin(), [scalar](T val) { return val / scalar; });
    return *this;
}

// Comparison operators
template<typename T>
bool VectorStack<T>::operator==(const VectorStack& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }
    return data_ == other.data_;
}

template<typename T>
bool VectorStack<T>::operator!=(const VectorStack& other) const {
    return !(*this == other);
}

// Utility methods
template<typename T>
T VectorStack<T>::sum() const {
    return std::reduce(std::execution::par_unseq, data_.begin(), data_.end(), T{0});
}

template<typename T>
T VectorStack<T>::min() const {
    if (data_.empty()) {
        throw std::runtime_error("Cannot compute min of empty VectorStack");
    }
    return *std::min_element(std::execution::par_unseq, data_.begin(), data_.end());
}

template<typename T>
T VectorStack<T>::max() const {
    if (data_.empty()) {
        throw std::runtime_error("Cannot compute max of empty VectorStack");
    }
    return *std::max_element(std::execution::par_unseq, data_.begin(), data_.end());
}

template<typename T>
T VectorStack<T>::mean() const {
    if (data_.empty()) {
        throw std::runtime_error("Cannot compute mean of empty VectorStack");
    }
    return sum() / static_cast<T>(data_.size());
}

template<typename T>
T VectorStack<T>::variance() const {
    if (data_.size() < 2) {
        return T{0};
    }
    
    T mean_val = mean();
    T sum_sq = std::reduce(std::execution::par_unseq, data_.begin(), data_.end(), T{0},
                           [mean_val](T acc, T val) { 
                               T diff = val - mean_val; 
                               return acc + diff * diff; 
                           });
    
    return sum_sq / static_cast<T>(data_.size() - 1);
}

template<typename T>
T VectorStack<T>::std_dev() const {
    return std::sqrt(variance());
}

// TDA-specific methods
template<typename T>
std::vector<T> VectorStack<T>::compute_distances() const {
    std::vector<T> distances;
    size_t n = rows_;
    size_t total_pairs = n * (n - 1) / 2;
    distances.reserve(total_pairs);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            T sum = T{0};
            for (size_t k = 0; k < cols_; ++k) {
                T diff = (*this)(i, k) - (*this)(j, k);
                sum += diff * diff;
            }
            distances.push_back(std::sqrt(sum));
        }
    }
    
    return distances;
}

template<typename T>
std::vector<T> VectorStack<T>::compute_angles() const {
    std::vector<T> angles;
    size_t n = rows_;
    size_t total_triangles = n * (n - 1) * (n - 2) / 6;
    angles.reserve(total_triangles);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            for (size_t k = j + 1; k < n; ++k) {
                // Compute vectors from point i to points j and k
                std::vector<T> v1(cols_), v2(cols_);
                T norm1 = T{0}, norm2 = T{0}, dot_product = T{0};
                
                for (size_t d = 0; d < cols_; ++d) {
                    v1[d] = (*this)(j, d) - (*this)(i, d);
                    v2[d] = (*this)(k, d) - (*this)(i, d);
                    norm1 += v1[d] * v1[d];
                    norm2 += v2[d] * v2[d];
                    dot_product += v1[d] * v2[d];
                }
                
                norm1 = std::sqrt(norm1);
                norm2 = std::sqrt(norm2);
                
                if (norm1 > T{0} && norm2 > T{0}) {
                    T cos_angle = dot_product / (norm1 * norm2);
                    cos_angle = std::clamp(cos_angle, T{-1}, T{1});
                    angles.push_back(std::acos(cos_angle));
                } else {
                    angles.push_back(T{0});
                }
            }
        }
    }
    
    return angles;
}

template<typename T>
std::vector<T> VectorStack<T>::compute_norms() const {
    std::vector<T> norms;
    norms.reserve(rows_);
    
    for (size_t i = 0; i < rows_; ++i) {
        T sum = T{0};
        for (size_t j = 0; j < cols_; ++j) {
            T val = (*this)(i, j);
            sum += val * val;
        }
        norms.push_back(std::sqrt(sum));
    }
    
    return norms;
}

    // Memory optimization
    template<typename T>
    void VectorStack<T>::optimize_memory() {
        // Align data to SIMD boundaries
        size_t aligned_size = ((data_.size() * sizeof(T) + core::SIMD_ALIGNMENT - 1) / 
                               core::SIMD_ALIGNMENT) * core::SIMD_ALIGNMENT;
    
    if (data_.capacity() * sizeof(T) < aligned_size) {
        data_.reserve(aligned_size / sizeof(T));
    }
}

template<typename T>
void VectorStack<T>::shrink_to_fit() {
    data_.shrink_to_fit();
}

// Serialization
template<typename T>
template<typename Archive>
void VectorStack<T>::serialize(Archive& ar) {
    ar(rows_, cols_, capacity_, data_);
}

// Explicit template instantiations
template class VectorStack<float>;
template class VectorStack<double>;
template class VectorStack<int>;

} // namespace tda::vector_stack
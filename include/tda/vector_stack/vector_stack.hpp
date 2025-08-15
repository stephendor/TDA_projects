#pragma once

#include "../core/types.hpp"
#include <vector>
#include <memory>
#include <concepts>
#include <ranges>
#include <span>
#include <optional>

namespace tda::vector_stack {

// Forward declarations
class PersistenceDiagram;
class Barcode;
class BettiNumbers;

// Vector Stack - Core data structure for TDA operations
template<typename T = double>
class VectorStack {
    static_assert(std::is_arithmetic_v<T>, "VectorStack requires arithmetic type");
    
private:
    std::vector<T> data_;
    std::size_t rows_;
    std::size_t cols_;
    std::size_t capacity_;
    
    // Memory pool for efficient allocation
    std::unique_ptr<T[]> memory_pool_;
    
public:
    // Type aliases
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using reverse_iterator = typename std::vector<T>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;
    
    // Constructors
    VectorStack() = default;
    
    explicit VectorStack(size_type rows, size_type cols)
        : rows_(rows), cols_(cols), capacity_(rows * cols) {
        data_.reserve(capacity_);
        data_.resize(capacity_);
    }
    
    VectorStack(size_type rows, size_type cols, const T& value)
        : rows_(rows), cols_(cols), capacity_(rows * cols) {
        data_.reserve(capacity_);
        data_.resize(capacity_, value);
    }
    
    // Copy constructor
    VectorStack(const VectorStack& other)
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_), capacity_(other.capacity_) {}
    
    // Move constructor
    VectorStack(VectorStack&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_), capacity_(other.capacity_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.capacity_ = 0;
    }
    
    // Destructor
    ~VectorStack() = default;
    
    // Assignment operators
    VectorStack& operator=(const VectorStack& other) {
        if (this != &other) {
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            capacity_ = other.capacity_;
        }
        return *this;
    }
    
    VectorStack& operator=(VectorStack&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            capacity_ = other.capacity_;
            other.rows_ = 0;
            other.cols_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Element access
    reference operator()(size_type row, size_type col) {
        return data_[row * cols_ + col];
    }
    
    const_reference operator()(size_type row, size_type col) const {
        return data_[row * cols_ + col];
    }
    
    reference at(size_type row, size_type col) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("VectorStack index out of range");
        }
        return data_[row * cols_ + col];
    }
    
    const_reference at(size_type row, size_type col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("VectorStack index out of range");
        }
        return data_[row * cols_ + col];
    }
    
    // Row and column access
    std::span<T> row(size_type row_idx) {
        if (row_idx >= rows_) {
            throw std::out_of_range("Row index out of range");
        }
        return std::span<T>(data_.data() + row_idx * cols_, cols_);
    }
    
    std::span<const T> row(size_type row_idx) const {
        if (row_idx >= rows_) {
            throw std::out_of_range("Row index out of range");
        }
        return std::span<const T>(data_.data() + row_idx * cols_, cols_);
    }
    
    std::span<T> column(size_type col_idx) {
        if (col_idx >= cols_) {
            throw std::out_of_range("Column index out of range");
        }
        std::vector<T> col_data;
        col_data.reserve(rows_);
        for (size_type i = 0; i < rows_; ++i) {
            col_data.push_back(data_[i * cols_ + col_idx]);
        }
        return std::span<T>(col_data);
    }
    
    // Capacity
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type size() const noexcept { return data_.size(); }
    size_type capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return data_.empty(); }
    
    // Iterators
    iterator begin() noexcept { return data_.begin(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    
    iterator end() noexcept { return data_.end(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cend() const noexcept { return data_.cend(); }
    
    reverse_iterator rbegin() noexcept { return data_.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return data_.rbegin(); }
    const_reverse_iterator crbegin() const noexcept { return data_.crbegin(); }
    
    reverse_iterator rend() noexcept { return data_.rend(); }
    const_reverse_iterator rend() const noexcept { return data_.rend(); }
    const_reverse_iterator crend() const noexcept { return data_.crend(); }
    
    // Data access
    pointer data() noexcept { return data_.data(); }
    const_pointer data() const noexcept { return data_.data(); }
    
    // Resize operations
    void resize(size_type new_rows, size_type new_cols) {
        rows_ = new_rows;
        cols_ = new_cols;
        capacity_ = new_rows * new_cols;
        data_.resize(capacity_);
    }
    
    void resize(size_type new_rows, size_type new_cols, const T& value) {
        rows_ = new_rows;
        cols_ = new_cols;
        capacity_ = new_rows * new_cols;
        data_.resize(capacity_, value);
    }
    
    void reserve(size_type new_capacity) {
        capacity_ = new_capacity;
        data_.reserve(new_capacity);
    }
    
    // Clear
    void clear() noexcept {
        data_.clear();
        rows_ = 0;
        cols_ = 0;
        capacity_ = 0;
    }
    
    // Swap
    void swap(VectorStack& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(capacity_, other.capacity_);
    }
    
    // Mathematical operations
    VectorStack operator+(const VectorStack& other) const;
    VectorStack operator-(const VectorStack& other) const;
    VectorStack operator*(const T& scalar) const;
    VectorStack operator/(const T& scalar) const;
    
    VectorStack& operator+=(const VectorStack& other);
    VectorStack& operator-=(const VectorStack& other);
    VectorStack& operator*=(const T& scalar);
    VectorStack& operator/=(const T& scalar);
    
    // Comparison operators
    bool operator==(const VectorStack& other) const;
    bool operator!=(const VectorStack& other) const;
    
    // Utility methods
    T sum() const;
    T min() const;
    T max() const;
    T mean() const;
    T variance() const;
    T std_dev() const;
    
    // TDA-specific methods
    std::vector<T> compute_distances() const;
    std::vector<T> compute_angles() const;
    std::vector<T> compute_norms() const;
    
    // Memory optimization
    void optimize_memory();
    void shrink_to_fit();
    
    // Serialization
    template<typename Archive>
    void serialize(Archive& ar);
};

// Free functions
template<typename T>
VectorStack<T> operator*(const T& scalar, const VectorStack<T>& vs) {
    return vs * scalar;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const VectorStack<T>& vs) {
    os << "VectorStack(" << vs.rows() << "x" << vs.cols() << ")\n";
    for (size_t i = 0; i < vs.rows(); ++i) {
        for (size_t j = 0; j < vs.cols(); ++j) {
            os << vs(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}

// Type aliases for common use cases
using VectorStackF = VectorStack<float>;
using VectorStackD = VectorStack<double>;
using VectorStackI = VectorStack<int>;

} // namespace tda::vector_stack

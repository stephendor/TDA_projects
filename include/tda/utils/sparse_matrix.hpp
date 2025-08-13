#pragma once

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstddef>
#include <cassert>

namespace tda::utils {

/**
 * @brief Sparse matrix representation for efficient memory usage
 * 
 * This class provides a sparse matrix implementation that only stores
 * non-zero elements, significantly reducing memory usage for sparse data.
 */
template<typename T>
class SparseMatrix {
private:
    size_t rows_;
    size_t cols_;
    T default_value_;
    std::unordered_map<size_t, std::unordered_map<size_t, T>> data_;
    
    // Helper function to compute hash for 2D coordinates
    static size_t hash2D(size_t row, size_t col) {
        return row * 0x100000000 + col;
    }
    
public:
    /**
     * @brief Constructor
     * @param rows Number of rows
     * @param cols Number of columns
     * @param default_value Default value for unset elements
     */
    SparseMatrix(size_t rows, size_t cols, T default_value = T{})
        : rows_(rows), cols_(cols), default_value_(default_value) {}
    
    /**
     * @brief Get matrix dimensions
     * @return Pair of (rows, cols)
     */
    std::pair<size_t, size_t> dimensions() const {
        return {rows_, cols_};
    }
    
    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    size_t rows() const { return rows_; }
    
    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    size_t cols() const { return cols_; }
    
    /**
     * @brief Set element value
     * @param row Row index
     * @param col Column index
     * @param value Value to set
     */
    void set(size_t row, size_t col, T value) {
        assert(row < rows_ && col < cols_);
        
        if (value == default_value_) {
            // Remove element if it equals default value
            auto row_it = data_.find(row);
            if (row_it != data_.end()) {
                row_it->second.erase(col);
                if (row_it->second.empty()) {
                    data_.erase(row_it);
                }
            }
        } else {
            data_[row][col] = value;
        }
    }
    
    /**
     * @brief Get element value
     * @param row Row index
     * @param col Column index
     * @return Element value
     */
    T get(size_t row, size_t col) const {
        assert(row < rows_ && col < cols_);
        
        auto row_it = data_.find(row);
        if (row_it == data_.end()) {
            return default_value_;
        }
        
        auto col_it = row_it->second.find(col);
        if (col_it == row_it->second.end()) {
            return default_value_;
        }
        
        return col_it->second;
    }
    
    /**
     * @brief Check if element is set (non-default)
     * @param row Row index
     * @param col Column index
     * @return True if element is set
     */
    bool isSet(size_t row, size_t col) const {
        assert(row < rows_ && col < cols_);
        
        auto row_it = data_.find(row);
        if (row_it == data_.end()) {
            return false;
        }
        
        return row_it->second.find(col) != row_it->second.end();
    }
    
    /**
     * @brief Get number of non-zero elements
     * @return Count of non-zero elements
     */
    size_t nnz() const {
        size_t count = 0;
        for (const auto& row : data_) {
            count += row.second.size();
        }
        return count;
    }
    
    /**
     * @brief Get memory usage estimate
     * @return Estimated memory usage in bytes
     */
    size_t memoryUsage() const {
        size_t overhead = sizeof(SparseMatrix) + data_.size() * sizeof(std::unordered_map<size_t, T>);
        size_t data = nnz() * (sizeof(size_t) + sizeof(T));
        return overhead + data;
    }
    
    /**
     * @brief Get sparsity ratio
     * @return Ratio of non-zero elements to total elements
     */
    double sparsity() const {
        return static_cast<double>(nnz()) / (rows_ * cols_);
    }
    
    /**
     * @brief Clear all data
     */
    void clear() {
        data_.clear();
    }
    
    /**
     * @brief Get row data
     * @param row Row index
     * @return Vector of (col, value) pairs for non-zero elements
     */
    std::vector<std::pair<size_t, T>> getRow(size_t row) const {
        assert(row < rows_);
        
        auto row_it = data_.find(row);
        if (row_it == data_.end()) {
            return {};
        }
        
        std::vector<std::pair<size_t, T>> result;
        result.reserve(row_it->second.size());
        
        for (const auto& [col, value] : row_it->second) {
            result.emplace_back(col, value);
        }
        
        // Sort by column index for consistent ordering
        std::sort(result.begin(), result.end());
        return result;
    }
    
    /**
     * @brief Get column data
     * @param col Column index
     * @return Vector of (row, value) pairs for non-zero elements
     */
    std::vector<std::pair<size_t, T>> getColumn(size_t col) const {
        assert(col < cols_);
        
        std::vector<std::pair<size_t, T>> result;
        
        for (const auto& [row, row_data] : data_) {
            auto col_it = row_data.find(col);
            if (col_it != row_data.end()) {
                result.emplace_back(row, col_it->second);
            }
        }
        
        // Sort by row index for consistent ordering
        std::sort(result.begin(), result.end());
        return result;
    }
    
    /**
     * @brief Get all non-zero elements
     * @return Vector of (row, col, value) tuples
     */
    std::vector<std::tuple<size_t, size_t, T>> getNonZeroElements() const {
        std::vector<std::tuple<size_t, size_t, T>> result;
        result.reserve(nnz());
        
        for (const auto& [row, row_data] : data_) {
            for (const auto& [col, value] : row_data) {
                result.emplace_back(row, col, value);
            }
        }
        
        return result;
    }
    
    /**
     * @brief Apply function to all non-zero elements
     * @param func Function to apply: void(size_t row, size_t col, T& value)
     */
    template<typename Func>
    void forEachNonZero(Func&& func) {
        for (auto& [row, row_data] : data_) {
            for (auto& [col, value] : row_data) {
                func(row, col, value);
            }
        }
    }
    
    /**
     * @brief Apply function to all non-zero elements (const version)
     * @param func Function to apply: void(size_t row, size_t col, const T& value)
     */
    template<typename Func>
    void forEachNonZero(Func&& func) const {
        for (const auto& [row, row_data] : data_) {
            for (const auto& [col, value] : row_data) {
                func(row, col, value);
            }
        }
    }
};

/**
 * @brief Symmetric sparse matrix for distance matrices
 * 
 * This class provides a memory-efficient representation for symmetric
 * matrices like distance matrices, storing only the upper triangular part.
 */
template<typename T>
class SymmetricSparseMatrix {
private:
    size_t size_;
    T default_value_;
    std::unordered_map<size_t, std::unordered_map<size_t, T>> data_;
    
public:
    /**
     * @brief Constructor
     * @param size Matrix size (NxN)
     * @param default_value Default value for unset elements
     */
    explicit SymmetricSparseMatrix(size_t size, T default_value = T{})
        : size_(size), default_value_(default_value) {}
    
    /**
     * @brief Get matrix size
     * @return Matrix size
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Set element value (automatically sets symmetric element)
     * @param i Row index
     * @param j Column index
     * @param value Value to set
     */
    void set(size_t i, size_t j, T value) {
        assert(i < size_ && j < size_);
        
        if (value == default_value_) {
            // Remove both symmetric elements
            removeElement(i, j);
            removeElement(j, i);
        } else {
            // Store in upper triangular part (i <= j)
            if (i <= j) {
                data_[i][j] = value;
            } else {
                data_[j][i] = value;
            }
        }
    }
    
    /**
     * @brief Get element value
     * @param i Row index
     * @param j Column index
     * @return Element value
     */
    T get(size_t i, size_t j) const {
        assert(i < size_ && j < size_);
        
        // Check upper triangular part
        if (i <= j) {
            auto row_it = data_.find(i);
            if (row_it != data_.end()) {
                auto col_it = row_it->second.find(j);
                if (col_it != row_it->second.end()) {
                    return col_it->second;
                }
            }
        } else {
            // Check lower triangular part
            auto row_it = data_.find(j);
            if (row_it != data_.end()) {
                auto col_it = row_it->second.find(i);
                if (col_it != row_it->second.end()) {
                    return col_it->second;
                }
            }
        }
        
        return default_value_;
    }
    
    /**
     * @brief Check if element is set
     * @param i Row index
     * @param j Column index
     * @return True if element is set
     */
    bool isSet(size_t i, size_t j) const {
        assert(i < size_ && j < size_);
        
        if (i <= j) {
            auto row_it = data_.find(i);
            if (row_it != data_.end()) {
                return row_it->second.find(j) != row_it->second.end();
            }
        } else {
            auto row_it = data_.find(j);
            if (row_it != data_.end()) {
                return row_it->second.find(i) != row_it->second.end();
            }
        }
        
        return false;
    }
    
    /**
     * @brief Get number of non-zero elements
     * @return Count of non-zero elements
     */
    size_t nnz() const {
        size_t count = 0;
        for (const auto& row : data_) {
            count += row.second.size();
        }
        return count;
    }
    
    /**
     * @brief Get memory usage estimate
     * @return Estimated memory usage in bytes
     */
    size_t memoryUsage() const {
        size_t overhead = sizeof(SymmetricSparseMatrix) + data_.size() * sizeof(std::unordered_map<size_t, T>);
        size_t data = nnz() * (sizeof(size_t) + sizeof(T));
        return overhead + data;
    }
    
    /**
     * @brief Get sparsity ratio
     * @return Ratio of non-zero elements to total elements
     */
    double sparsity() const {
        return static_cast<double>(nnz()) / (size_ * size_);
    }
    
    /**
     * @brief Clear all data
     */
    void clear() {
        data_.clear();
    }
    
    /**
     * @brief Get row data
     * @param row Row index
     * @return Vector of (col, value) pairs for non-zero elements
     */
    std::vector<std::pair<size_t, T>> getRow(size_t row) const {
        assert(row < size_);
        
        std::vector<std::pair<size_t, T>> result;
        
        // Check upper triangular part (row <= col)
        auto row_it = data_.find(row);
        if (row_it != data_.end()) {
            for (const auto& [col, value] : row_it->second) {
                if (col >= row) { // Only upper triangular
                    result.emplace_back(col, value);
                }
            }
        }
        
        // Check lower triangular part (col < row)
        for (const auto& [other_row, row_data] : data_) {
            if (other_row < row) {
                auto col_it = row_data.find(row);
                if (col_it != row_data.end()) {
                    result.emplace_back(other_row, col_it->second);
                }
            }
        }
        
        // Sort by column index for consistent ordering
        std::sort(result.begin(), result.end());
        return result;
    }
    
    /**
     * @brief Get all non-zero elements
     * @return Vector of (row, col, value) tuples
     */
    std::vector<std::tuple<size_t, size_t, T>> getNonZeroElements() const {
        std::vector<std::tuple<size_t, size_t, T>> result;
        result.reserve(nnz());
        
        for (const auto& [row, row_data] : data_) {
            for (const auto& [col, value] : row_data) {
                result.emplace_back(row, col, value);
            }
        }
        
        return result;
    }

private:
    /**
     * @brief Remove element from matrix
     * @param i Row index
     * @param j Column index
     */
    void removeElement(size_t i, size_t j) {
        auto row_it = data_.find(i);
        if (row_it != data_.end()) {
            row_it->second.erase(j);
            if (row_it->second.empty()) {
                data_.erase(row_it);
            }
        }
    }
};

} // namespace tda::utils


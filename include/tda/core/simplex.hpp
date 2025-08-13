#pragma once

#include "types.hpp"
#include <vector>
#include <memory>

namespace tda::core {

/**
 * @brief Simplex data structure for TDA computations
 * 
 * Represents a k-simplex (0-simplex = vertex, 1-simplex = edge,
 * 2-simplex = triangle, etc.) with associated metadata.
 */
class Simplex {
public:
    using VertexList = std::vector<core::Index>;
    
    Simplex() = default;
    Simplex(VertexList vertices, double filtration_value = 0.0);
    
    // Copy constructor and assignment
    Simplex(const Simplex&) = default;
    Simplex& operator=(const Simplex&) = default;
    
    // Move constructor and assignment
    Simplex(Simplex&&) = default;
    Simplex& operator=(Simplex&&) = default;
    
    ~Simplex() = default;
    
    // Access methods
    const VertexList& vertices() const noexcept;
    VertexList& vertices() noexcept;
    
    // Properties
    core::Dimension dimension() const noexcept;
    double filtrationValue() const noexcept;
    void setFiltrationValue(double value);
    
    // Vertex operations
    void addVertex(core::Index vertex);
    void removeVertex(core::Index vertex);
    bool containsVertex(core::Index vertex) const;
    
    // Utility methods
    bool isEmpty() const noexcept;
    size_t size() const noexcept;
    void clear();
    bool isValid() const noexcept;
    
    // Comparison operators
    bool operator==(const Simplex& other) const;
    bool operator!=(const Simplex& other) const;
    bool operator<(const Simplex& other) const;

private:
    VertexList vertices_;
    double filtration_value_ = 0.0;
};

} // namespace tda::core

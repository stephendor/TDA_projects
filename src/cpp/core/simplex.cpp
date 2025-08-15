#include "../../../include/tda/core/simplex.hpp"
#include <algorithm>
#include <stdexcept>

namespace tda::core {

Simplex::Simplex(VertexList vertices, double filtration_value)
    : vertices_(std::move(vertices)), filtration_value_(filtration_value) {
    // Sort vertices for consistent representation
    std::sort(vertices_.begin(), vertices_.end());
}

const Simplex::VertexList& Simplex::vertices() const noexcept {
    return vertices_;
}

Simplex::VertexList& Simplex::vertices() noexcept {
    return vertices_;
}

core::Dimension Simplex::dimension() const noexcept {
    if (vertices_.empty()) return 0;
    return static_cast<core::Dimension>(vertices_.size()) - 1;
}

double Simplex::filtrationValue() const noexcept {
    return filtration_value_;
}

void Simplex::setFiltrationValue(double value) {
    filtration_value_ = value;
}

void Simplex::addVertex(core::Index vertex) {
    if (std::find(vertices_.begin(), vertices_.end(), vertex) == vertices_.end()) {
        vertices_.push_back(vertex);
        std::sort(vertices_.begin(), vertices_.end());
    }
}

void Simplex::removeVertex(core::Index vertex) {
    auto it = std::find(vertices_.begin(), vertices_.end(), vertex);
    if (it != vertices_.end()) {
        vertices_.erase(it);
    }
}

bool Simplex::containsVertex(core::Index vertex) const {
    return std::find(vertices_.begin(), vertices_.end(), vertex) != vertices_.end();
}

bool Simplex::isEmpty() const noexcept {
    return vertices_.empty();
}

size_t Simplex::size() const noexcept {
    return vertices_.size();
}

void Simplex::clear() {
    vertices_.clear();
    filtration_value_ = 0.0;
}

bool Simplex::isValid() const noexcept {
    // A simplex is valid if it has at least one vertex
    // and all vertices are non-negative
    if (vertices_.empty()) return false;
    
    for (core::Index vertex : vertices_) {
        if (vertex < 0) return false;
    }
    
    return true;
}

bool Simplex::operator==(const Simplex& other) const {
    return vertices_ == other.vertices_ && 
           std::abs(filtration_value_ - other.filtration_value_) < 1e-10;
}

bool Simplex::operator!=(const Simplex& other) const {
    return !(*this == other);
}

bool Simplex::operator<(const Simplex& other) const {
    if (dimension() != other.dimension()) {
        return dimension() < other.dimension();
    }
    if (std::abs(filtration_value_ - other.filtration_value_) > 1e-10) {
        return filtration_value_ < other.filtration_value_;
    }
    return vertices_ < other.vertices_;
}

} // namespace tda::core
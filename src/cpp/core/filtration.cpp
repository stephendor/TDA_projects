#include "../../../include/tda/core/filtration.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tda::core {

Filtration::Filtration(SimplexContainer simplices)
    : simplices_(std::move(simplices)), sorted_(false), valid_(true) {
    // Validate simplices
    for (const auto& simplex : simplices_) {
        if (!simplex.isValid()) {
            valid_ = false;
            break;
        }
    }
}

const Simplex& Filtration::operator[](size_t index) const {
    if (index >= simplices_.size()) {
        throw std::out_of_range("Filtration index out of range");
    }
    return simplices_[index];
}

Simplex& Filtration::operator[](size_t index) {
    if (index >= simplices_.size()) {
        throw std::out_of_range("Filtration index out of range");
    }
    return simplices_[index];
}

size_t Filtration::size() const noexcept {
    return simplices_.size();
}

bool Filtration::empty() const noexcept {
    return simplices_.empty();
}

void Filtration::reserve(size_t capacity) {
    simplices_.reserve(capacity);
}

Filtration::ConstIterator Filtration::begin() const noexcept {
    return simplices_.begin();
}

Filtration::Iterator Filtration::begin() noexcept {
    return simplices_.begin();
}

Filtration::ConstIterator Filtration::end() const noexcept {
    return simplices_.end();
}

Filtration::Iterator Filtration::end() noexcept {
    return simplices_.end();
}

const Filtration::SimplexContainer& Filtration::simplices() const noexcept {
    return simplices_;
}

Filtration::SimplexContainer& Filtration::simplices() noexcept {
    return simplices_;
}

void Filtration::addSimplex(const Simplex& simplex) {
    if (!simplex.isValid()) {
        throw std::invalid_argument("Invalid simplex");
    }
    simplices_.push_back(simplex);
    sorted_ = false;
}

void Filtration::addSimplex(Simplex&& simplex) {
    if (!simplex.isValid()) {
        throw std::invalid_argument("Invalid simplex");
    }
    simplices_.push_back(std::move(simplex));
    sorted_ = false;
}

void Filtration::removeSimplex(size_t index) {
    if (index >= simplices_.size()) {
        throw std::out_of_range("Filtration index out of range");
    }
    simplices_.erase(simplices_.begin() + index);
    sorted_ = false;
}

void Filtration::clear() {
    simplices_.clear();
    sorted_ = false;
}

void Filtration::sortByFiltration() {
    std::sort(simplices_.begin(), simplices_.end(),
              [](const Simplex& a, const Simplex& b) {
                  return a.filtrationValue() < b.filtrationValue();
              });
    sorted_ = true;
}

void Filtration::sortByDimension() {
    std::sort(simplices_.begin(), simplices_.end(),
              [](const Simplex& a, const Simplex& b) {
                  if (a.dimension() != b.dimension()) {
                      return a.dimension() < b.dimension();
                  }
                  return a.filtrationValue() < b.filtrationValue();
              });
    sorted_ = true;
}

void Filtration::sortByDimensionAndFiltration() {
    sortByDimension();
}

std::vector<Simplex> Filtration::getSimplicesAtFiltration(double value) const {
    std::vector<Simplex> result;
    for (const auto& simplex : simplices_) {
        if (std::abs(simplex.filtrationValue() - value) < 1e-10) {
            result.push_back(simplex);
        }
    }
    return result;
}

std::vector<Simplex> Filtration::getSimplicesAtDimension(int dimension) const {
    std::vector<Simplex> result;
    for (const auto& simplex : simplices_) {
        if (simplex.dimension() == dimension) {
            result.push_back(simplex);
        }
    }
    return result;
}

double Filtration::maxFiltrationValue() const {
    if (simplices_.empty()) return 0.0;
    auto max_it = std::max_element(simplices_.begin(), simplices_.end(),
                                   [](const Simplex& a, const Simplex& b) {
                                       return a.filtrationValue() < b.filtrationValue();
                                   });
    return max_it->filtrationValue();
}

double Filtration::minFiltrationValue() const {
    if (simplices_.empty()) return 0.0;
    auto min_it = std::min_element(simplices_.begin(), simplices_.end(),
                                   [](const Simplex& a, const Simplex& b) {
                                       return a.filtrationValue() < b.filtrationValue();
                                   });
    return min_it->filtrationValue();
}

int Filtration::maxDimension() const {
    if (simplices_.empty()) return -1;
    auto max_it = std::max_element(simplices_.begin(), simplices_.end(),
                                   [](const Simplex& a, const Simplex& b) {
                                       return a.dimension() < b.dimension();
                                   });
    return max_it->dimension();
}

bool Filtration::isValid() const {
    return valid_;
}

bool Filtration::isSorted() const {
    return sorted_;
}

} // namespace tda::core
// Implementations for the public Filtration interface declared in include/tda/core/filtration.hpp
#include "../../../include/tda/core/filtration.hpp"
#include <algorithm>

namespace tda::core {

Filtration::Filtration(SimplexContainer simplices)
    : simplices_(std::move(simplices)) {}

const Simplex& Filtration::operator[](size_t index) const { return simplices_[index]; }
Simplex& Filtration::operator[](size_t index) { return simplices_[index]; }

size_t Filtration::size() const noexcept { return simplices_.size(); }
bool Filtration::empty() const noexcept { return simplices_.empty(); }
void Filtration::reserve(size_t capacity) { simplices_.reserve(capacity); }

Filtration::ConstIterator Filtration::begin() const noexcept { return simplices_.begin(); }
Filtration::ConstIterator Filtration::end() const noexcept { return simplices_.end(); }
Filtration::Iterator Filtration::begin() noexcept { return simplices_.begin(); }
Filtration::Iterator Filtration::end() noexcept { return simplices_.end(); }

const Filtration::SimplexContainer& Filtration::simplices() const noexcept { return simplices_; }
Filtration::SimplexContainer& Filtration::simplices() noexcept { return simplices_; }

void Filtration::addSimplex(const Simplex& simplex) {
    simplices_.push_back(simplex);
    sorted_ = false;
}
void Filtration::addSimplex(Simplex&& simplex) {
    simplices_.push_back(std::move(simplex));
    sorted_ = false;
}
void Filtration::removeSimplex(size_t index) {
    if (index >= simplices_.size()) return;
    simplices_.erase(simplices_.begin() + index);
}
void Filtration::clear() { simplices_.clear(); sorted_ = false; }

void Filtration::sortByFiltration() {
    std::sort(simplices_.begin(), simplices_.end(), [](const Simplex& a, const Simplex& b){
        return a.filtrationValue() < b.filtrationValue();
    });
    sorted_ = true;
}
void Filtration::sortByDimension() {
    std::sort(simplices_.begin(), simplices_.end(), [](const Simplex& a, const Simplex& b){
        return a.dimension() < b.dimension();
    });
    sorted_ = false;
}
void Filtration::sortByDimensionAndFiltration() {
    std::sort(simplices_.begin(), simplices_.end(), [](const Simplex& a, const Simplex& b){
        if (a.dimension() != b.dimension()) return a.dimension() < b.dimension();
        return a.filtrationValue() < b.filtrationValue();
    });
    sorted_ = true;
}

std::vector<Simplex> Filtration::getSimplicesAtFiltration(double value) const {
    std::vector<Simplex> out;
    for (const auto& s : simplices_) if (s.filtrationValue() == value) out.push_back(s);
    return out;
}
std::vector<Simplex> Filtration::getSimplicesAtDimension(int dimension) const {
    std::vector<Simplex> out;
    for (const auto& s : simplices_) if (static_cast<int>(s.dimension()) == dimension) out.push_back(s);
    return out;
}
double Filtration::maxFiltrationValue() const {
    double m = 0.0; for (const auto& s: simplices_) m = std::max(m, s.filtrationValue()); return m;
}
double Filtration::minFiltrationValue() const {
    if (simplices_.empty()) return 0.0; double m = simplices_.front().filtrationValue();
    for (const auto& s: simplices_) m = std::min(m, s.filtrationValue()); return m;
}
int Filtration::maxDimension() const {
    int m = 0; for (const auto& s: simplices_) m = std::max(m, static_cast<int>(s.dimension())); return m;
}

bool Filtration::isValid() const {
    for (const auto& s: simplices_) if (!s.isValid()) return false; return true;
}
bool Filtration::isSorted() const { return sorted_; }

} // namespace tda::core
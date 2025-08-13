#pragma once

#include "types.hpp"
#include "simplex.hpp"
#include <vector>
#include <memory>
#include <algorithm>

namespace tda::core {

/**
 * @brief Filtration data structure for TDA computations
 * 
 * Represents a filtered simplicial complex where simplices are
 * ordered by their filtration values. This is essential for
 * persistent homology computation.
 */
class Filtration {
public:
    using SimplexContainer = std::vector<Simplex>;
    using Iterator = SimplexContainer::iterator;
    using ConstIterator = SimplexContainer::const_iterator;
    
    Filtration() = default;
    explicit Filtration(SimplexContainer simplices);
    
    // Copy constructor and assignment
    Filtration(const Filtration&) = default;
    Filtration& operator=(const Filtration&) = default;
    
    // Move constructor and assignment
    Filtration(Filtration&&) = default;
    Filtration& operator=(Filtration&&) = default;
    
    ~Filtration() = default;
    
    // Access methods
    const Simplex& operator[](size_t index) const;
    Simplex& operator[](size_t index);
    
    // Size and capacity
    size_t size() const noexcept;
    bool empty() const noexcept;
    void reserve(size_t capacity);
    
    // Iterators
    ConstIterator begin() const noexcept;
    ConstIterator end() const noexcept;
    Iterator begin() noexcept;
    Iterator end() noexcept;
    
    // Data access
    const SimplexContainer& simplices() const noexcept;
    SimplexContainer& simplices() noexcept;
    
    // Filtration operations
    void addSimplex(const Simplex& simplex);
    void addSimplex(Simplex&& simplex);
    void removeSimplex(size_t index);
    void clear();
    
    // Sorting and ordering
    void sortByFiltration();
    void sortByDimension();
    void sortByDimensionAndFiltration();
    
    // Query methods
    std::vector<Simplex> getSimplicesAtFiltration(double value) const;
    std::vector<Simplex> getSimplicesAtDimension(int dimension) const;
    double maxFiltrationValue() const;
    double minFiltrationValue() const;
    int maxDimension() const;
    
    // Validation
    bool isValid() const;
    bool isSorted() const;

private:
    SimplexContainer simplices_;
    bool sorted_ = false;
    bool valid_ = true;
};

} // namespace tda::core

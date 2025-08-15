#pragma once

#include "types.hpp"
#include "filtration.hpp"
#include <vector>
#include <memory>
#include <utility>

namespace tda::core {

/**
 * @brief Persistent homology computation engine
 * 
 * Computes persistent homology from a filtered simplicial complex.
 * Provides methods to extract persistence pairs, Betti numbers,
 * and other topological invariants.
 */
class PersistentHomology {
public:
    using PersistencePair = tda::core::PersistencePair;
    using PersistenceContainer = std::vector<PersistencePair>;
    using BettiNumbers = std::vector<int>;
    
    PersistentHomology() = default;
    explicit PersistentHomology(const Filtration& filtration);
    
    // Copy constructor and assignment
    PersistentHomology(const PersistentHomology&) = default;
    PersistentHomology& operator=(const PersistentHomology&) = default;
    
    // Move constructor and assignment
    PersistentHomology(PersistentHomology&&) = default;
    PersistentHomology& operator=(PersistentHomology&&) = default;
    
    ~PersistentHomology() = default;
    
    // Computation methods
    Result<void> compute(int coefficient_field = 2);
    bool isComputed() const noexcept;
    
    // Results access
    Result<PersistenceContainer> getPersistencePairs() const;
    Result<BettiNumbers> getBettiNumbers() const;
    Result<std::vector<double>> getPersistenceValues() const;
    
    // Statistics
    size_t numPersistencePairs() const noexcept;
    int maxDimension() const noexcept;
    double maxPersistence() const noexcept;
    double minPersistence() const noexcept;
    
    // Utility methods
    void clear();
    void setFiltration(const Filtration& filtration);
    const Filtration& getFiltration() const;
    
    // Validation
    bool isValid() const;

private:
    Filtration filtration_;
    PersistenceContainer persistence_pairs_;
    BettiNumbers betti_numbers_;
    bool computed_ = false;
    bool valid_ = true;
    int coefficient_field_ = 2;
};

} // namespace tda::core

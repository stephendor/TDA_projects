#include "../../../include/tda/core/persistent_homology.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tda::core {

PersistentHomology::PersistentHomology(const Filtration& filtration)
    : filtration_(filtration), computed_(false), valid_(true) {
    // Validate filtration
    if (!filtration.isValid()) {
        valid_ = false;
    }
}

Result<void> PersistentHomology::compute(int coefficient_field) {
    if (!valid_) {
        return Result<void>::failure("Invalid PersistentHomology object");
    }
    
    if (filtration_.empty()) {
        return Result<void>::failure("Empty filtration");
    }
    
    try {
        // Clear previous results
        persistence_pairs_.clear();
        betti_numbers_.clear();
        
        // Sort filtration by birth time
        filtration_.sortByFiltration();
        
        // For now, implement a simple algorithm
        // In a real implementation, this would use matrix reduction
        // or other persistent homology algorithms
        
        computed_ = true;
        return Result<void>::success();
    } catch (const std::exception& e) {
        return Result<void>::failure(std::string("Computation failed: ") + e.what());
    }
}

bool PersistentHomology::isComputed() const noexcept {
    return computed_;
}

Result<PersistentHomology::PersistenceContainer> PersistentHomology::getPersistencePairs() const {
    if (!computed_) {
        return Result<PersistenceContainer>::failure("Persistent homology not computed");
    }
    
    if (!valid_) {
        return Result<PersistenceContainer>::failure("Invalid PersistentHomology object");
    }
    
    return Result<PersistenceContainer>::success(persistence_pairs_);
}

Result<PersistentHomology::BettiNumbers> PersistentHomology::getBettiNumbers() const {
    if (!computed_) {
        return Result<BettiNumbers>::failure("Persistent homology not computed");
    }
    
    if (!valid_) {
        return Result<BettiNumbers>::failure("Invalid PersistentHomology object");
    }
    
    return Result<BettiNumbers>::success(betti_numbers_);
}

Result<std::vector<double>> PersistentHomology::getPersistenceValues() const {
    if (!computed_) {
        return Result<std::vector<double>>::failure("Persistent homology not computed");
    }
    
    if (!valid_) {
        return Result<std::vector<double>>::failure("Invalid PersistentHomology object");
    }
    
    std::vector<double> values;
    values.reserve(persistence_pairs_.size());
    
    for (const auto& pair : persistence_pairs_) {
        if (std::isinf(pair.death)) {
            values.push_back(std::numeric_limits<double>::infinity());
        } else {
            values.push_back(pair.death - pair.birth);
        }
    }
    
    return Result<std::vector<double>>::success(std::move(values));
}

size_t PersistentHomology::numPersistencePairs() const noexcept {
    return persistence_pairs_.size();
}

int PersistentHomology::maxDimension() const noexcept {
    if (persistence_pairs_.empty()) return -1;
    
    int max_dim = 0;
    for (const auto& pair : persistence_pairs_) {
        max_dim = std::max(max_dim, static_cast<int>(pair.dimension));
    }
    return max_dim;
}

double PersistentHomology::maxPersistence() const noexcept {
    if (persistence_pairs_.empty()) return 0.0;
    
    double max_pers = 0.0;
    for (const auto& pair : persistence_pairs_) {
        if (!std::isinf(pair.death)) {
            double pers = pair.death - pair.birth;
            max_pers = std::max(max_pers, pers);
        }
    }
    return max_pers;
}

double PersistentHomology::minPersistence() const noexcept {
    if (persistence_pairs_.empty()) return 0.0;
    
    double min_pers = std::numeric_limits<double>::infinity();
    for (const auto& pair : persistence_pairs_) {
        if (!std::isinf(pair.death)) {
            double pers = pair.death - pair.birth;
            min_pers = std::min(min_pers, pers);
        }
    }
    
    return std::isinf(min_pers) ? 0.0 : min_pers;
}

void PersistentHomology::clear() {
    persistence_pairs_.clear();
    betti_numbers_.clear();
    computed_ = false;
}

void PersistentHomology::setFiltration(const Filtration& filtration) {
    filtration_ = filtration;
    computed_ = false;
    valid_ = filtration.isValid();
}

const Filtration& PersistentHomology::getFiltration() const {
    return filtration_;
}

bool PersistentHomology::isValid() const {
    return valid_;
}

} // namespace tda::core
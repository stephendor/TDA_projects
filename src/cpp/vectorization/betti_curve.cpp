#include <tda/vectorization/betti_curve.hpp>
#include <tda/vectorization/vectorizer_registry.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace tda {
namespace vectorization {

BettiCurve::BettiCurve() = default;

BettiCurve::BettiCurve(const Config& config) : config_(config) {}

BettiCurve::FeatureVector BettiCurve::vectorize(const PersistenceDiagram& diagram) const {
    // Determine filtration range if not specified
    double min_filtration = config_.min_filtration;
    double max_filtration = config_.max_filtration;
    
    if (max_filtration < 0) {
        // Auto-compute max filtration
        max_filtration = 0.0;
        for (const auto& pair : diagram) {
            max_filtration = std::max(max_filtration, pair.death);
        }
        // Add a small margin
        max_filtration *= 1.1;
    }
    
    // Generate grid points
    std::vector<double> grid_points(config_.resolution);
    const double step = (max_filtration - min_filtration) / (config_.resolution - 1);
    for (size_t i = 0; i < config_.resolution; ++i) {
        grid_points[i] = min_filtration + i * step;
    }
    
    // Pre-allocate result vector
    const size_t feature_dim = config_.resolution * (config_.max_dimension + 1);
    FeatureVector result(feature_dim, 0.0);
    
    // Calculate Betti numbers for each dimension and grid point
    for (int dim = 0; dim <= config_.max_dimension; ++dim) {
        const size_t offset = dim * config_.resolution;
        
        double max_betti = 0.0;
        for (size_t i = 0; i < config_.resolution; ++i) {
            const double filtration_value = grid_points[i];
            const int betti_number = calculateBettiNumber(diagram, filtration_value, dim);
            result[offset + i] = static_cast<double>(betti_number);
            max_betti = std::max(max_betti, result[offset + i]);
        }
        
        // Normalize if requested
        if (config_.normalize && max_betti > 0) {
            for (size_t i = 0; i < config_.resolution; ++i) {
                result[offset + i] /= max_betti;
            }
        }
    }
    
    return result;
}

std::string BettiCurve::getName() const {
    return "BettiCurve";
}

size_t BettiCurve::getDimension() const {
    return config_.resolution * (config_.max_dimension + 1);
}

std::string BettiCurve::toJSON() const {
    nlohmann::json json;
    json["type"] = "BettiCurve";
    json["config"] = {
        {"resolution", config_.resolution},
        {"max_dimension", config_.max_dimension},
        {"normalize", config_.normalize},
        {"include_dimension_prefix", config_.include_dimension_prefix},
        {"min_filtration", config_.min_filtration},
        {"max_filtration", config_.max_filtration}
    };
    return json.dump();
}

const BettiCurve::Config& BettiCurve::getConfig() const {
    return config_;
}

void BettiCurve::setConfig(const Config& config) {
    config_ = config;
}

int BettiCurve::calculateBettiNumber(
    const PersistenceDiagram& diagram,
    double filtration_value,
    int dimension) const {
    
    // Count how many homology features of the given dimension are active at this filtration value
    int count = 0;
    for (const auto& pair : diagram) {
        if (static_cast<int>(pair.dimension) != dimension) continue;
        
        // A feature is active if it has been born but not died yet
        if (pair.birth <= filtration_value && filtration_value < pair.death) {
            count++;
        }
    }
    
    return count;
}

// Factory function
std::unique_ptr<Vectorizer> createBettiCurve() {
    return std::make_unique<BettiCurve>();
}

// Register the BettiCurve vectorizer
REGISTER_VECTORIZER(BettiCurve, createBettiCurve);

} // namespace vectorization
} // namespace tda

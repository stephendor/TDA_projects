#include <tda/vectorization/persistence_landscape.hpp>
#include <tda/vectorization/vectorizer_registry.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <utility>
#include <nlohmann/json.hpp>

namespace tda {
namespace vectorization {

PersistenceLandscape::PersistenceLandscape() = default;

PersistenceLandscape::PersistenceLandscape(const Config& config) : config_(config) {}

PersistenceLandscape::FeatureVector PersistenceLandscape::vectorize(const PersistenceDiagram& diagram) const {
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
    const size_t feature_dim = config_.resolution * config_.num_landscapes * (config_.max_dimension + 1);
    FeatureVector result(feature_dim, 0.0);
    
    // Calculate landscapes for each dimension
    for (int dim = 0; dim <= config_.max_dimension; ++dim) {
        // Calculate each landscape level
        for (size_t level = 1; level <= config_.num_landscapes; ++level) {
            // Compute this landscape level
            auto landscape_values = computeLandscapeLevel(diagram, level, dim, grid_points);
            
            // Determine normalization factor if needed
            double max_value = 0.0;
            if (config_.normalize) {
                max_value = *std::max_element(landscape_values.begin(), landscape_values.end());
            }
            
            // Store values in result vector
            const size_t offset = (dim * config_.num_landscapes + (level - 1)) * config_.resolution;
            for (size_t i = 0; i < config_.resolution; ++i) {
                double value = landscape_values[i];
                if (config_.normalize && max_value > 0) {
                    value /= max_value;
                }
                result[offset + i] = value;
            }
        }
    }
    
    return result;
}

std::string PersistenceLandscape::getName() const {
    return "PersistenceLandscape";
}

size_t PersistenceLandscape::getDimension() const {
    return config_.resolution * config_.num_landscapes * (config_.max_dimension + 1);
}

std::string PersistenceLandscape::toJSON() const {
    nlohmann::json json;
    json["type"] = "PersistenceLandscape";
    json["config"] = {
        {"resolution", config_.resolution},
        {"num_landscapes", config_.num_landscapes},
        {"max_dimension", config_.max_dimension},
        {"normalize", config_.normalize},
        {"include_dimension_prefix", config_.include_dimension_prefix},
        {"min_filtration", config_.min_filtration},
        {"max_filtration", config_.max_filtration}
    };
    return json.dump();
}

const PersistenceLandscape::Config& PersistenceLandscape::getConfig() const {
    return config_;
}

void PersistenceLandscape::setConfig(const Config& config) {
    config_ = config;
}

std::vector<double> PersistenceLandscape::computeLandscapeLevel(
    const PersistenceDiagram& diagram,
    size_t level,
    int dimension,
    const std::vector<double>& grid_points) const {
    
    // Filter diagram for the desired dimension
    std::vector<std::pair<double, double>> filtered_pairs;
    for (const auto& pair : diagram) {
        if (pair.dimension == dimension) {
            filtered_pairs.emplace_back(pair.birth, pair.death);
        }
    }
    
    // If no pairs, return zero landscape
    if (filtered_pairs.empty()) {
        return std::vector<double>(grid_points.size(), 0.0);
    }
    
    // For each grid point, compute the landscape value
    std::vector<double> landscape_values(grid_points.size(), 0.0);
    
    for (size_t i = 0; i < grid_points.size(); ++i) {
        const double t = grid_points[i];
        
        // Compute landscape value for this point
        std::vector<double> values;
        values.reserve(filtered_pairs.size());
        
        for (const auto& [birth, death] : filtered_pairs) {
            const double persistence = death - birth;
            
            if (t <= birth) {
                values.push_back(0.0);
            } else if (birth < t && t <= (birth + death) / 2) {
                values.push_back(t - birth);
            } else if ((birth + death) / 2 < t && t < death) {
                values.push_back(death - t);
            } else {
                values.push_back(0.0);
            }
        }
        
        // Sort values in descending order
        std::sort(values.begin(), values.end(), std::greater<double>());
        
        // Get the k-th largest value (k-th landscape)
        if (level <= values.size()) {
            landscape_values[i] = values[level - 1];
        } else {
            landscape_values[i] = 0.0;
        }
    }
    
    return landscape_values;
}

// Factory function
std::unique_ptr<Vectorizer> createPersistenceLandscape() {
    return std::make_unique<PersistenceLandscape>();
}

// Register the PersistenceLandscape vectorizer
REGISTER_VECTORIZER(PersistenceLandscape, createPersistenceLandscape);

} // namespace vectorization
} // namespace tda

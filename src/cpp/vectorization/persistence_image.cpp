#include <tda/vectorization/persistence_image.hpp>
#include <tda/vectorization/vectorizer_registry.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <nlohmann/json.hpp>

namespace tda {
namespace vectorization {

PersistenceImage::PersistenceImage() = default;

PersistenceImage::PersistenceImage(const Config& config) : config_(config) {}

PersistenceImage::FeatureVector PersistenceImage::vectorize(const PersistenceDiagram& diagram) const {
    // Determine image size
    const size_t total_pixels = config_.resolution_x * config_.resolution_y * (config_.max_dimension + 1);
    FeatureVector result(total_pixels, 0.0);
    
    // Process each dimension
    for (int dim = 0; dim <= config_.max_dimension; ++dim) {
        // Filter diagram for the desired dimension
        std::vector<std::pair<double, double>> filtered_pairs;
        for (const auto& pair : diagram) {
            if (pair.dimension == dim) {
                // Transform from (birth, death) to (birth, persistence)
                // For stability and better visualization
                const double birth = pair.birth;
                const double persistence = pair.death - pair.birth;
                // Use min_persistence as threshold if provided
                const double threshold = config_.min_persistence;
                if (persistence >= threshold) {
                    filtered_pairs.emplace_back(birth, persistence);
                }
            }
        }
        
        // If no points in this dimension, continue
        if (filtered_pairs.empty()) {
            continue;
        }
        
        // Determine domain boundaries
    double min_birth = std::numeric_limits<double>::max();
    double max_birth = std::numeric_limits<double>::lowest();
    double max_persistence = 0.0;
        
        for (const auto& [birth, persistence] : filtered_pairs) {
            min_birth = std::min(min_birth, birth);
            max_birth = std::max(max_birth, birth);
            max_persistence = std::max(max_persistence, persistence);
        }
        
        // Ensure we have sensible domain boundaries
        if (min_birth == std::numeric_limits<double>::max()) {
            min_birth = 0.0;
        }
        
        // Honor configured ranges when provided, otherwise auto with margin
        const double margin_factor = 0.1;
        const double birth_range = max_birth - min_birth;
        if (config_.max_birth >= 0.0) {
            max_birth = config_.max_birth;
        } else {
            max_birth += birth_range * margin_factor;
        }
        if (config_.min_birth <= max_birth) {
            min_birth = (config_.max_birth >= 0.0 || config_.min_birth != 0.0) ? config_.min_birth : (min_birth - birth_range * margin_factor);
        }
        if (config_.max_persistence >= 0.0) {
            max_persistence = config_.max_persistence;
        } else {
            max_persistence *= (1.0 + margin_factor);
        }
        
        // Compute the image
        const size_t base_idx = dim * config_.resolution_x * config_.resolution_y;
        
        for (size_t i = 0; i < config_.resolution_y; ++i) {
            const double pers_i = max_persistence * (i + 0.5) / config_.resolution_y;
            
            for (size_t j = 0; j < config_.resolution_x; ++j) {
                const double birth_j = min_birth + (max_birth - min_birth) * (j + 0.5) / config_.resolution_x;
                
                // Compute pixel value by summing kernel contributions
                double pixel_value = 0.0;
                
                for (const auto& [birth, persistence] : filtered_pairs) {
                    // Apply Gaussian kernel (isotropic sigma)
                    const double birth_diff = (birth_j - birth) / config_.sigma;
                    const double pers_diff = (pers_i - persistence) / config_.sigma;
                    
                    // Weight
                    double weight = 1.0;
                    if (config_.weighting_function) {
                        weight = config_.weighting_function(birth, birth + persistence, persistence);
                    } else {
                        weight = persistence; // reasonable default
                    }
                    
                    // Add contribution from this point
                    const double kernel_value = weight * exp(-(birth_diff * birth_diff + pers_diff * pers_diff) / 2.0);
                    pixel_value += kernel_value;
                }
                
                // Store pixel value
                const size_t idx = base_idx + i * config_.resolution_x + j;
                result[idx] = pixel_value;
            }
        }
        
        // Normalize if requested
        if (config_.normalize) {
            const size_t dim_start = base_idx;
            const size_t dim_end = dim_start + config_.resolution_x * config_.resolution_y;
            
            double max_value = *std::max_element(result.begin() + dim_start, result.begin() + dim_end);
            
            if (max_value > 0) {
                for (size_t idx = dim_start; idx < dim_end; ++idx) {
                    result[idx] /= max_value;
                }
            }
        }
    }
    
    return result;
}

std::string PersistenceImage::getName() const {
    return "PersistenceImage";
}

size_t PersistenceImage::getDimension() const {
    return config_.resolution_x * config_.resolution_y * (config_.max_dimension + 1);
}

std::string PersistenceImage::toJSON() const {
    nlohmann::json json;
    json["type"] = "PersistenceImage";
    json["config"] = {
        {"resolution_x", config_.resolution_x},
        {"resolution_y", config_.resolution_y},
    {"sigma", config_.sigma},
        {"max_dimension", config_.max_dimension},
        {"normalize", config_.normalize},
    {"include_dimension_prefix", config_.include_dimension_prefix},
    {"min_birth", config_.min_birth},
    {"max_birth", config_.max_birth},
    {"min_persistence", config_.min_persistence},
    {"max_persistence", config_.max_persistence}
    };
    return json.dump();
}

const PersistenceImage::Config& PersistenceImage::getConfig() const {
    return config_;
}

void PersistenceImage::setConfig(const Config& config) {
    config_ = config;
}

// Factory function
std::unique_ptr<Vectorizer> createPersistenceImage() {
    return std::make_unique<PersistenceImage>();
}

// Register the PersistenceImage vectorizer
REGISTER_VECTORIZER(PersistenceImage, createPersistenceImage);

} // namespace vectorization
} // namespace tda

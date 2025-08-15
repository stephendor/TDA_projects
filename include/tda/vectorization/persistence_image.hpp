#pragma once

#include <tda/vectorization/vectorizer.hpp>
#include <string>
#include <functional>
#include <nlohmann/json.hpp>

namespace tda {
namespace vectorization {

/**
 * @brief Implements Persistence Image vectorization
 * 
 * Persistence images transform persistence diagrams into a fixed-size
 * discretized grid where each pixel value represents the density of 
 * persistence pairs in that region, weighted by a persistence-based
 * weighting function.
 */
class PersistenceImage : public Vectorizer {
public:
    /**
     * @brief Weighting function type
     * 
     * A function that assigns a weight to a persistence pair based on its birth,
     * death, and persistence values.
     */
    using WeightingFunction = std::function<double(double birth, double death, double persistence)>;
    
    /**
     * @brief Configuration for Persistence Image computation
     */
    struct Config {
        // Image resolution (width and height)
        size_t resolution_x = 50;
        size_t resolution_y = 50;
        
        // Standard deviation for the Gaussian kernel
        double sigma = 0.1;
        
        // Maximum homology dimension to consider (0 = connected components, 1 = loops, etc.)
        int max_dimension = 2;
        
        // Whether to normalize the image values to [0,1] range
        bool normalize = true;
        
        // Whether to include dimension in the feature vector as a prefix
        bool include_dimension_prefix = true;
        
        // Minimum birth value (auto-computed if not specified)
        double min_birth = 0.0;
        
        // Maximum birth value (auto-computed if not specified)
        double max_birth = -1.0;  // negative value means auto-compute
        
        // Minimum persistence value (auto-computed if not specified)
        double min_persistence = 0.0;
        
        // Maximum persistence value (auto-computed if not specified)
        double max_persistence = -1.0;  // negative value means auto-compute
        
        // Weighting function to use (linear by default)
        WeightingFunction weighting_function = nullptr;
    };
    
    /**
     * @brief Construct with default configuration
     */
    PersistenceImage();
    
    /**
     * @brief Construct with custom configuration
     * 
     * @param config The configuration to use
     */
    explicit PersistenceImage(const Config& config);
    
    /**
     * @brief Vectorize a persistence diagram into a persistence image
     * 
     * @param diagram The persistence diagram to vectorize
     * @return FeatureVector The resulting persistence image feature vector
     */
    FeatureVector vectorize(const PersistenceDiagram& diagram) const override;
    
    /**
     * @brief Get the name of the vectorization method
     * 
     * @return std::string "PersistenceImage"
     */
    std::string getName() const override;
    
    /**
     * @brief Get the dimension of the resulting feature vector
     * 
     * @return size_t resolution_x * resolution_y * (max_dimension + 1)
     */
    size_t getDimension() const override;
    
    /**
     * @brief Create a JSON representation of the vectorizer configuration
     * 
     * @return std::string JSON string with vectorizer parameters
     */
    std::string toJSON() const override;
    
    /**
     * @brief Get the current configuration
     * 
     * @return const Config& The current configuration
     */
    const Config& getConfig() const;
    
    /**
     * @brief Set a new configuration
     * 
     * @param config The new configuration to use
     */
    void setConfig(const Config& config);
    
    /**
     * @brief Create a linear weighting function
     * 
     * Weight = persistence / max_persistence
     * 
     * @param max_persistence The maximum persistence value to normalize by
     * @return WeightingFunction The linear weighting function
     */
    static WeightingFunction linearWeighting(double max_persistence = 1.0);
    
    /**
     * @brief Create a quadratic weighting function
     * 
     * Weight = (persistence / max_persistence)^2
     * 
     * @param max_persistence The maximum persistence value to normalize by
     * @return WeightingFunction The quadratic weighting function
     */
    static WeightingFunction quadraticWeighting(double max_persistence = 1.0);
    
    /**
     * @brief Create a sigmoidal weighting function
     * 
     * Weight = 1 / (1 + exp(-(persistence - midpoint) / steepness))
     * 
     * @param midpoint The persistence value at which the weight is 0.5
     * @param steepness Controls the steepness of the sigmoid
     * @return WeightingFunction The sigmoidal weighting function
     */
    static WeightingFunction sigmoidWeighting(double midpoint = 0.5, double steepness = 0.1);
    
private:
    Config config_;
    
    /**
     * @brief Default linear weighting function
     * 
     * @param birth Birth value
     * @param death Death value
     * @param persistence Persistence value (death - birth)
     * @return double The weight
     */
    static double defaultWeighting(double birth, double death, double persistence);
    
    /**
     * @brief Compute the Gaussian kernel value at a point
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param birth_x Birth X coordinate (transformed to image coordinates)
     * @param persistence_y Persistence Y coordinate (transformed to image coordinates)
     * @param sigma Standard deviation of the Gaussian
     * @return double The Gaussian kernel value
     */
    static double gaussianKernel(
        double x, 
        double y, 
        double birth_x, 
        double persistence_y, 
        double sigma);
};

// Factory function for PersistenceImage
std::unique_ptr<Vectorizer> createPersistenceImage();

} // namespace vectorization
} // namespace tda

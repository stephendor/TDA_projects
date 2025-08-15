#pragma once

#include <tda/vectorization/vectorizer.hpp>
#include <array>
#include <string>
#include <nlohmann/json.hpp>

namespace tda {
namespace vectorization {

/**
 * @brief Implements Betti Curve vectorization
 * 
 * Betti curves capture the evolution of homology groups across the filtration.
 * For each dimension k and filtration value t, the Betti curve represents
 * the k-th Betti number, which counts the number of k-dimensional holes
 * at filtration value t.
 */
class BettiCurve : public Vectorizer {
public:
    /**
     * @brief Configuration for Betti curve computation
     */
    struct Config {
        // Number of sample points along the filtration range
        size_t resolution = 100;
        
        // Maximum homology dimension to consider (0 = connected components, 1 = loops, etc.)
        int max_dimension = 2;
        
        // Whether to normalize the curve values to [0,1] range
        bool normalize = true;
        
        // Whether to include dimension in the feature vector as a prefix
        bool include_dimension_prefix = true;
        
        // Minimum filtration value (auto-computed if not specified)
        double min_filtration = 0.0;
        
        // Maximum filtration value (auto-computed if not specified)
        double max_filtration = -1.0;  // negative value means auto-compute
    };
    
    /**
     * @brief Construct with default configuration
     */
    BettiCurve();
    
    /**
     * @brief Construct with custom configuration
     * 
     * @param config The configuration to use
     */
    explicit BettiCurve(const Config& config);
    
    /**
     * @brief Vectorize a persistence diagram into a Betti curve
     * 
     * @param diagram The persistence diagram to vectorize
     * @return FeatureVector The resulting Betti curve feature vector
     */
    FeatureVector vectorize(const PersistenceDiagram& diagram) const override;
    
    /**
     * @brief Get the name of the vectorization method
     * 
     * @return std::string "BettiCurve"
     */
    std::string getName() const override;
    
    /**
     * @brief Get the dimension of the resulting feature vector
     * 
     * @return size_t resolution * (max_dimension + 1)
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
    
private:
    Config config_;
    
    /**
     * @brief Calculate Betti numbers at a specific filtration value
     * 
     * @param diagram The persistence diagram
     * @param filtration_value The filtration value
     * @param dimension The homology dimension
     * @return int The Betti number
     */
    int calculateBettiNumber(
        const PersistenceDiagram& diagram,
        double filtration_value,
        int dimension) const;
};

// Factory function for BettiCurve
std::unique_ptr<Vectorizer> createBettiCurve();

} // namespace vectorization
} // namespace tda

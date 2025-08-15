#pragma once

#include <tda/core/types.hpp>
#include <tda/core/persistent_homology.hpp>
#include <string>
#include <vector>
#include <memory>

namespace tda {
namespace vectorization {

/**
 * @brief Base class for all vectorization methods
 * 
 * This abstract class defines the interface for all persistence diagram
 * vectorization methods. Each derived class must implement the vectorize()
 * method to convert a persistence diagram into a vector representation.
 */
class Vectorizer {
public:
    using PersistenceDiagram = std::vector<tda::core::PersistencePair>;
    using FeatureVector = std::vector<double>;
    
    Vectorizer() = default;
    virtual ~Vectorizer() = default;
    
    // Delete copy constructors to prevent slicing
    Vectorizer(const Vectorizer&) = delete;
    Vectorizer& operator=(const Vectorizer&) = delete;
    
    // Allow move semantics
    Vectorizer(Vectorizer&&) = default;
    Vectorizer& operator=(Vectorizer&&) = default;
    
    /**
     * @brief Vectorize a persistence diagram
     * 
     * @param diagram The persistence diagram to vectorize
     * @return FeatureVector The resulting feature vector
     */
    virtual FeatureVector vectorize(const PersistenceDiagram& diagram) const = 0;
    
    /**
     * @brief Get the name of the vectorization method
     * 
     * @return std::string The name of the method
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get the dimension of the resulting feature vector
     * 
     * @return size_t The dimension of the feature vector
     */
    virtual size_t getDimension() const = 0;
    
    /**
     * @brief Create a JSON representation of the vectorizer configuration
     * 
     * @return std::string JSON string with vectorizer parameters
     */
    virtual std::string toJSON() const = 0;
};

// Factory function type for creating vectorizers
using VectorizerFactory = std::unique_ptr<Vectorizer> (*)();

} // namespace vectorization
} // namespace tda

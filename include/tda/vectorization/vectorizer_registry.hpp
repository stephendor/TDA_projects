#pragma once

#include <tda/vectorization/vectorizer.hpp>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tda {
namespace vectorization {

/**
 * @brief Registry for vectorization methods
 * 
 * This class manages the available vectorization methods and allows retrieving
 * them by name. It provides a central registry for all vectorization methods
 * in the system.
 */
class VectorizerRegistry {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return VectorizerRegistry& The singleton instance
     */
    static VectorizerRegistry& getInstance();
    
    /**
     * @brief Register a vectorization method
     * 
     * @param name The name of the vectorization method
     * @param factory Factory function to create an instance
     * @return bool True if registration succeeded, false if already registered
     */
    bool registerVectorizer(const std::string& name, VectorizerFactory factory);
    
    /**
     * @brief Create a vectorizer by name
     * 
     * @param name The name of the vectorization method
     * @return std::unique_ptr<Vectorizer> The created vectorizer, or nullptr if not found
     */
    std::unique_ptr<Vectorizer> createVectorizer(const std::string& name) const;
    
    /**
     * @brief Check if a vectorization method is registered
     * 
     * @param name The name of the vectorization method
     * @return bool True if registered, false otherwise
     */
    bool isRegistered(const std::string& name) const;
    
    /**
     * @brief Get a list of all registered vectorization methods
     * 
     * @return std::vector<std::string> List of registered method names
     */
    std::vector<std::string> getRegisteredNames() const;
    
private:
    // Private constructor for singleton
    VectorizerRegistry() = default;
    
    // Delete copy and move constructors
    VectorizerRegistry(const VectorizerRegistry&) = delete;
    VectorizerRegistry& operator=(const VectorizerRegistry&) = delete;
    VectorizerRegistry(VectorizerRegistry&&) = delete;
    VectorizerRegistry& operator=(VectorizerRegistry&&) = delete;
    
    std::unordered_map<std::string, VectorizerFactory> factories_;
};

/**
 * @brief Helper class for automatic vectorizer registration
 * 
 * This class allows vectorizers to be registered automatically when
 * the program starts, before main() is called.
 */
class VectorizerRegistrar {
public:
    /**
     * @brief Register a vectorizer
     * 
     * @param name The name of the vectorization method
     * @param factory Factory function to create an instance
     */
    VectorizerRegistrar(const std::string& name, VectorizerFactory factory);
};

// Macro for easy registration of vectorizers
#define REGISTER_VECTORIZER(name, factory) \
    static tda::vectorization::VectorizerRegistrar registrar_##name(#name, factory)

} // namespace vectorization
} // namespace tda

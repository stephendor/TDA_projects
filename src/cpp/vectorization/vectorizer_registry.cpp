#include <tda/vectorization/vectorizer_registry.hpp>
#include <tda/vectorization/betti_curve.hpp>
#include <tda/vectorization/persistence_landscape.hpp>
#include <tda/vectorization/persistence_image.hpp>
#include <iostream>
#include <stdexcept>

namespace tda {
namespace vectorization {

VectorizerRegistry& VectorizerRegistry::getInstance() {
    static VectorizerRegistry instance;
    return instance;
}

bool VectorizerRegistry::registerVectorizer(const std::string& name, VectorizerFactory factory) {
    auto [it, inserted] = factories_.emplace(name, factory);
    if (!inserted) {
        // Overwrite existing and warn
        std::cerr << "Warning: Vectorizer '" << name << "' already registered. Overwriting." << std::endl;
        it->second = factory;
        return false;
    }
    return true;
}

std::unique_ptr<Vectorizer> VectorizerRegistry::createVectorizer(const std::string& name) const {
    auto it = factories_.find(name);
    if (it == factories_.end()) {
        throw std::runtime_error("Vectorizer '" + name + "' not found in registry");
    }
    return it->second();
}

std::vector<std::string> VectorizerRegistry::getRegisteredNames() const {
    std::vector<std::string> names;
    names.reserve(factories_.size());
    
    for (const auto& [name, _] : factories_) {
        names.push_back(name);
    }
    
    return names;
}

// Implement the registrar that vectorizers use for static registration
VectorizerRegistrar::VectorizerRegistrar(const std::string& name, VectorizerFactory factory) {
    VectorizerRegistry::getInstance().registerVectorizer(name, factory);
}

} // namespace vectorization
} // namespace tda

#pragma once

#include <tda/storage/storage_service.hpp>
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace tda { namespace storage {

class MongoDBStorage : public StorageService {
public:
    using PersistenceDiagram = StorageService::PersistenceDiagram;
    using FeatureVector = StorageService::FeatureVector;

    struct ConnectionConfig {
        std::string host{"localhost"};
        int port{27017};
        std::string database;
        std::string user;
        std::string password;
    };

    MongoDBStorage();
    explicit MongoDBStorage(const ConnectionConfig& config);
    ~MongoDBStorage();

    bool connect();
    void disconnect();
    bool isConnected() const;

    // StorageService interface
    bool storeDiagram(const std::string& id, const PersistenceDiagram& diagram, const DiagramMetadata& metadata) override;
    bool storeVectorization(const std::string& diagram_id, const std::string& vectorizer_id, const FeatureVector& vector, const VectorMetadata& metadata) override;

    std::optional<PersistenceDiagram> getDiagram(const std::string& id) override;
    std::optional<FeatureVector> getVectorization(const std::string& diagram_id, const std::string& vectorizer_id) override;
    std::vector<std::string> listDiagrams(const QueryFilter& filter) override;
    bool deleteDiagram(const std::string& id) override;

private:
    ConnectionConfig config_{};
    bool connected_{false};
};

} } // namespace tda::storage

#include "tda/spatial/spatial_index.hpp"
#include <memory>

namespace tda::spatial {

std::unique_ptr<SpatialIndex> createSpatialIndex(const std::vector<std::vector<double>>& points, size_t maxDimension) {
    if (points.empty()) {
        return nullptr;
    }
    
    // For now, always use KD-tree since BallTree implementation was removed
    // TODO: Re-implement BallTree in Phase 2B
    return std::make_unique<KDTree>();
}

} // namespace tda::spatial

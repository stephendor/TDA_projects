#include "../../../include/tda/algorithms/vietoris_rips.hpp"
#include "../../../include/tda/algorithms/vietoris_rips_impl.hpp"

namespace tda::algorithms {

// Forward declaration of the implementation class
class VietorisRipsImpl;

VietorisRips::VietorisRips() : impl_(std::make_unique<VietorisRipsImpl>()) {}

VietorisRips::~VietorisRips() = default;

VietorisRips::VietorisRips(VietorisRips&& other) noexcept = default;

VietorisRips& VietorisRips::operator=(VietorisRips&& other) noexcept = default;

tda::core::Result<void> VietorisRips::initialize(const std::vector<std::vector<double>>& points,
                                   double threshold,
                                   int max_dimension,
                                   int coefficient_field) {
    return impl_->initialize(points, threshold, max_dimension, coefficient_field);
}

tda::core::Result<void> VietorisRips::computeComplex() {
    return impl_->computeComplex();
}

tda::core::Result<void> VietorisRips::computePersistence() {
    return impl_->computePersistence();
}

tda::core::Result<std::vector<tda::core::SimplexInfo>> VietorisRips::getSimplices() const {
    return impl_->getSimplices();
}

tda::core::Result<std::vector<tda::core::PersistencePair>> VietorisRips::getPersistencePairs() const {
    return impl_->getPersistencePairs();
}

tda::core::Result<std::vector<int>> VietorisRips::getBettiNumbers() const {
    return impl_->getBettiNumbers();
}

tda::core::Result<tda::core::ComplexStatistics> VietorisRips::getStatistics() const {
    return impl_->getStatistics();
}

std::vector<double> VietorisRips::computeDistancesBatch(const std::vector<std::vector<double>>& points,
                                                       const std::vector<double>& query_point) {
    return VietorisRipsImpl::computeDistancesBatch(points, query_point);
}

} // namespace tda::algorithms

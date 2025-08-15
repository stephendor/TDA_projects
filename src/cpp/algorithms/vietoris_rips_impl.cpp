#include <tda/algorithms/vietoris_rips_impl.hpp>
#include <stdexcept>
#include <numeric>
#include <cmath>

namespace tda::algorithms {

using ResultVoid = tda::core::Result<void>;
using SimplexInfo = tda::core::SimplexInfo;
using PersistencePair = tda::core::PersistencePair;
using ComplexStatistics = tda::core::ComplexStatistics;

ResultVoid VietorisRipsImpl::initialize(const std::vector<std::vector<double>>& points,
                                        double threshold,
                                        int max_dimension,
                                        int coefficient_field) {
    points_ = points;
    threshold_ = threshold;
    max_dimension_ = max_dimension;
    coefficient_field_ = coefficient_field;
    cleanup();
    return ResultVoid::success();
}

ResultVoid VietorisRipsImpl::computeComplex() {
    if (points_.empty()) {
        return ResultVoid::failure("VietorisRipsImpl: initialize() must be called with non-empty points before computeComplex().");
    }
    if (threshold_ <= 0.0) {
        return ResultVoid::failure("VietorisRipsImpl: threshold must be > 0.");
    }

    try {
        // Reset any previous state
        cleanup();

        // Build Rips complex from point cloud with Euclidean distance
        rips_complex_ = std::make_unique<Rips_complex>(points_, threshold_, Gudhi::Euclidean_distance());

        // Create simplex tree up to max_dimension_
        simplex_tree_ = std::make_unique<Simplex_tree>();
        rips_complex_->create_complex(*simplex_tree_, max_dimension_);

        // Ensure filtration is non-decreasing and valid
        simplex_tree_->initialize_filtration();

        return ResultVoid::success();
    } catch (const std::exception& e) {
        cleanup();
        return ResultVoid::failure(std::string("VietorisRipsImpl::computeComplex failed: ") + e.what());
    }
}

ResultVoid VietorisRipsImpl::computePersistence() {
    if (!simplex_tree_) {
        return ResultVoid::failure("VietorisRipsImpl: complex not computed. Call computeComplex() first.");
    }
    try {
        // Create persistent cohomology object and compute
        persistent_cohomology_ = std::make_unique<Persistent_cohomology>(*simplex_tree_);
        persistent_cohomology_->init_coefficients(coefficient_field_);
    // Compute with default min persistence to keep API compatibility across versions
    persistent_cohomology_->compute_persistent_cohomology();
        return ResultVoid::success();
    } catch (const std::exception& e) {
        persistent_cohomology_.reset();
        return ResultVoid::failure(std::string("VietorisRipsImpl::computePersistence failed: ") + e.what());
    }
}

tda::core::Result<std::vector<SimplexInfo>> VietorisRipsImpl::getSimplices() const {
    if (!simplex_tree_) {
        return tda::core::Result<std::vector<SimplexInfo>>::failure("Simplices not computed: complex not computed yet.");
    }
    std::vector<SimplexInfo> simplices;
    try {
        simplices.reserve(simplex_tree_->num_simplices());
        for (auto simplex : simplex_tree_->filtration_simplex_range()) {
            int dim = static_cast<int>(simplex_tree_->dimension(simplex));
            double filt = static_cast<double>(simplex_tree_->filtration(simplex));
            std::vector<int> verts;
            verts.reserve(static_cast<std::size_t>(dim + 1));
            for (auto v : simplex_tree_->simplex_vertex_range(simplex)) {
                verts.push_back(static_cast<int>(v));
            }
            simplices.emplace_back(dim, filt, std::move(verts));
        }
        return tda::core::Result<std::vector<SimplexInfo>>::success(std::move(simplices));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<SimplexInfo>>::failure(std::string("Failed to extract simplices: ") + e.what());
    }
}

tda::core::Result<std::vector<PersistencePair>> VietorisRipsImpl::getPersistencePairs() const {
    if (!persistent_cohomology_) {
        return tda::core::Result<std::vector<PersistencePair>>::failure("Persistence not computed: call computePersistence() first.");
    }
    try {
        std::vector<PersistencePair> pairs;
        auto persistent_pairs = persistent_cohomology_->get_persistent_pairs();
        pairs.reserve(persistent_pairs.size());
        for (const auto& pr : persistent_pairs) {
            auto birth_handle = std::get<0>(pr);
            auto death_handle = std::get<1>(pr);
            double birth = simplex_tree_->filtration(birth_handle);
            double death = simplex_tree_->filtration(death_handle);
            int dim = simplex_tree_->dimension(birth_handle);
            pairs.emplace_back(static_cast<tda::core::Dimension>(dim), static_cast<tda::core::Birth>(birth), static_cast<tda::core::Death>(death));
        }
        return tda::core::Result<std::vector<PersistencePair>>::success(std::move(pairs));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<PersistencePair>>::failure(std::string("Failed to extract persistence pairs: ") + e.what());
    }
}

tda::core::Result<std::vector<int>> VietorisRipsImpl::getBettiNumbers() const {
    if (!persistent_cohomology_) {
        return tda::core::Result<std::vector<int>>::failure("Betti numbers not computed: call computePersistence() first.");
    }
    try {
        int max_dim = std::max(0, max_dimension_);
        std::vector<int> betti(static_cast<std::size_t>(max_dim + 1), 0);
        for (int d = 0; d <= max_dim; ++d) {
            betti[static_cast<std::size_t>(d)] = persistent_cohomology_->betti_number(d);
        }
        return tda::core::Result<std::vector<int>>::success(std::move(betti));
    } catch (const std::exception& e) {
        return tda::core::Result<std::vector<int>>::failure(std::string("Failed to compute Betti numbers: ") + e.what());
    }
}

tda::core::Result<ComplexStatistics> VietorisRipsImpl::getStatistics() const {
    if (!simplex_tree_) {
        ComplexStatistics stats;
        stats.num_points = points_.size();
        stats.num_simplices = 0;
        stats.max_dimension = max_dimension_;
        stats.threshold = threshold_;
        return tda::core::Result<ComplexStatistics>::success(stats);
    }

    ComplexStatistics stats;
    stats.num_points = points_.size();
    stats.num_simplices = simplex_tree_->num_simplices();
    stats.max_dimension = simplex_tree_->dimension();
    stats.threshold = threshold_;
    stats.simplex_count_by_dim.assign(static_cast<std::size_t>(stats.max_dimension + 1), 0);
    for (auto simplex : simplex_tree_->filtration_simplex_range()) {
        int dim = static_cast<int>(simplex_tree_->dimension(simplex));
        if (dim >= 0 && dim <= stats.max_dimension) {
            stats.simplex_count_by_dim[static_cast<std::size_t>(dim)] += 1;
        }
    }
    return tda::core::Result<ComplexStatistics>::success(std::move(stats));
}

std::vector<double> VietorisRipsImpl::computeDistancesBatch(const std::vector<std::vector<double>>& points,
                                                           const std::vector<double>& query_point) {
    std::vector<double> distances;
    distances.reserve(points.size());
    for (const auto& p : points) {
        double sum = 0.0;
        size_t d = std::min(p.size(), query_point.size());
        for (size_t i = 0; i < d; ++i) {
            double diff = p[i] - query_point[i];
            sum += diff * diff;
        }
        distances.push_back(std::sqrt(sum));
    }
    return distances;
}

void VietorisRipsImpl::cleanup() {
    simplex_tree_.reset();
    rips_complex_.reset();
    persistent_cohomology_.reset();
}

} // namespace tda::algorithms

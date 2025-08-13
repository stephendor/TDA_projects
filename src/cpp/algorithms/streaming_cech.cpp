#include "tda/algorithms/streaming_cech.hpp"
#include "tda/utils/streaming_distance_matrix.hpp"
#include <algorithm>
#include <cmath>

namespace tda::algorithms {

static inline double max_pairwise_distance(const std::vector<std::vector<double>>& pts,
                                           const std::vector<size_t>& simplex) {
    double m = 0.0;
    for (size_t i = 0; i < simplex.size(); ++i) {
        for (size_t j = i + 1; j < simplex.size(); ++j) {
            const auto& a = pts[simplex[i]];
            const auto& b = pts[simplex[j]];
            double s = 0.0;
            for (size_t k = 0; k < a.size(); ++k) {
                double d = a[k] - b[k];
                s += d * d;
            }
            m = std::max(m, std::sqrt(s));
        }
    }
    return m;
}

// Lifecycle

tda::core::Result<void> StreamingCechComplex::initialize(const PointContainer& points) {
    if (points.empty()) {
        return tda::core::Result<void>::failure("Empty point cloud provided");
    }
    // Validate dimensions
    const size_t dim = points.front().size();
    for (const auto& p : points) {
        if (p.size() != dim) {
            return tda::core::Result<void>::failure("Inconsistent point dimensions");
        }
    }
    points_ = points;
    neighbors_.assign(points_.size(), {});

    // Initialize GUDHI structures
    simplex_tree_ = std::make_unique<Simplex_tree>();
    persistent_cohomology_ = std::make_unique<Persistent_cohomology>(*simplex_tree_);
    return tda::core::Result<void>::success();
}

tda::core::Result<void> StreamingCechComplex::computeComplex() {
    if (points_.empty() || !simplex_tree_) {
        return tda::core::Result<void>::failure("Must call initialize() first");
    }
    simplex_tree_->clear();

    // Build neighbor adjacency using streaming distances under global radius threshold
    buildNeighborsStreaming();

    // Add vertices
    for (size_t i = 0; i < points_.size(); ++i) {
        std::vector<size_t> v = {i};
        addSimplexToTree(v, 0.0);
    }

    // Build higher-dimensional simplices up to config_.maxDimension using bounded neighbors
    const size_t N = points_.size();
    const size_t max_dim = std::min<size_t>(config_.maxDimension, 3);

    for (size_t i = 0; i < N; ++i) {
        auto& nbrs = neighbors_[i];
        // Cap neighbor list
        if (nbrs.size() > config_.maxNeighbors) nbrs.resize(config_.maxNeighbors);

        // 1-simplex edges already implied by adjacency; we'll add them via addSimplexToTree
        for (size_t jidx = 0; jidx < nbrs.size(); ++jidx) {
            size_t j = nbrs[jidx];
            if (j <= i) continue; // avoid duplicates
            std::vector<size_t> e = {i, j};
            double filt = max_pairwise_distance(points_, e);
            addSimplexToTree(e, filt);
        }

        if (max_dim >= 2) {
            for (size_t a = 0; a < nbrs.size(); ++a) {
                size_t j = nbrs[a];
                if (j <= i) continue;
                // Intersect neighbor lists approximately by scanning k in nbrs
                for (size_t b = a + 1; b < nbrs.size(); ++b) {
                    size_t k = nbrs[b];
                    if (k <= j) continue;
                    // Check that (j,k) is also an edge
                    // naive check within bounded neighbor sets
                    if (std::find(neighbors_[j].begin(), neighbors_[j].end(), k) == neighbors_[j].end()) continue;
                    std::vector<size_t> tri = {i, j, k};
                    double filt = max_pairwise_distance(points_, tri);
                    addSimplexToTree(tri, filt);
                }
            }
        }

        if (max_dim >= 3) {
            // Build 3-simplices (tetrahedra) from triangles plus one more neighbor, bounded
            const auto& nbrs_i = nbrs;
            for (size_t a = 0; a < nbrs_i.size(); ++a) {
                size_t j = nbrs_i[a]; if (j <= i) continue;
                for (size_t b = a + 1; b < nbrs_i.size(); ++b) {
                    size_t k = nbrs_i[b]; if (k <= j) continue;
                    if (std::find(neighbors_[j].begin(), neighbors_[j].end(), k) == neighbors_[j].end()) continue;
                    for (size_t c = b + 1; c < nbrs_i.size(); ++c) {
                        size_t l = nbrs_i[c]; if (l <= k) continue;
                        // Check edges (j,l) and (k,l)
                        if (std::find(neighbors_[j].begin(), neighbors_[j].end(), l) == neighbors_[j].end()) continue;
                        if (std::find(neighbors_[k].begin(), neighbors_[k].end(), l) == neighbors_[k].end()) continue;
                        std::vector<size_t> tet = {i, j, k, l};
                        double filt = max_pairwise_distance(points_, tet);
                        addSimplexToTree(tet, filt);
                    }
                }
            }
        }
    }

    return tda::core::Result<void>::success();
}

tda::core::Result<void> StreamingCechComplex::computePersistence(int coefficientField) {
    if (!simplex_tree_ || !persistent_cohomology_) {
        return tda::core::Result<void>::failure("Complex not computed");
    }
    try {
        persistent_cohomology_->init_coefficients(coefficientField);
        persistent_cohomology_->compute_persistent_cohomology();
    } catch (const std::exception& e) {
        return tda::core::Result<void>::failure(std::string("Persistent homology failed: ") + e.what());
    }
    return tda::core::Result<void>::success();
}

tda::core::Result<std::vector<tda::core::SimplexInfo>> StreamingCechComplex::getSimplices() const {
    if (!simplex_tree_) return tda::core::Result<std::vector<tda::core::SimplexInfo>>::failure("Complex not computed");
    std::vector<tda::core::SimplexInfo> out;
    for (auto s : simplex_tree_->complex_simplex_range()) {
        tda::core::SimplexInfo info;
        info.dimension = simplex_tree_->dimension(s);
        info.filtration_value = simplex_tree_->filtration(s);
        for (auto v : simplex_tree_->simplex_vertex_range(s)) {
            info.vertices.push_back(static_cast<int>(v));
        }
        out.push_back(std::move(info));
    }
    return tda::core::Result<std::vector<tda::core::SimplexInfo>>::success(std::move(out));
}

tda::core::Result<tda::core::ComplexStatistics> StreamingCechComplex::getStatistics() const {
    if (!simplex_tree_) return tda::core::Result<tda::core::ComplexStatistics>::failure("Complex not computed");
    tda::core::ComplexStatistics stats;
    stats.num_points = points_.size();
    stats.num_simplices = simplex_tree_->num_simplices();
    stats.max_dimension = simplex_tree_->dimension();
    stats.threshold = std::max(0.0, config_.dm.max_distance);
    stats.simplex_count_by_dim.clear();
    for (int d = 0; d <= simplex_tree_->dimension(); ++d) {
        int count = 0;
        for (auto s : simplex_tree_->complex_simplex_range()) {
            if (simplex_tree_->dimension(s) == d) count++;
        }
        stats.simplex_count_by_dim.push_back(count);
    }
    return tda::core::Result<tda::core::ComplexStatistics>::success(std::move(stats));
}

void StreamingCechComplex::clear() {
    points_.clear();
    neighbors_.clear();
    simplex_tree_.reset();
    persistent_cohomology_.reset();
}

// Helpers

double StreamingCechComplex::euclidean(const Point& a, const Point& b) {
    double s = 0.0; for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; s += d * d; } return std::sqrt(s);
}

double StreamingCechComplex::estimateAdaptiveRadius(size_t idx) const {
    if (!config_.useAdaptiveRadius) return config_.radius;
    // Use simple heuristic: average distance to first K neighbors seen in adjacency
    const auto& nbrs = neighbors_[idx];
    if (nbrs.empty()) return config_.radius;
    size_t K = std::min<size_t>(nbrs.size(), 8);
    double s = 0.0; size_t cnt = 0;
    for (size_t t = 0; t < K; ++t) {
        size_t j = nbrs[t];
        s += euclidean(points_[idx], points_[j]);
        cnt++;
    }
    double avg = s / std::max<size_t>(1, cnt);
    return std::min(config_.radius * config_.radiusMultiplier, avg);
}

void StreamingCechComplex::buildNeighborsStreaming() {
    // Configure streaming distance matrix
    tda::utils::StreamingDistanceMatrix dm;
    auto dmcfg = config_.dm;
    // Set threshold as 2*radius (edge if balls of radius r intersect => dist <= 2r)
    dmcfg.max_distance = 2.0 * config_.radius;
    dm.setConfig(dmcfg);

    neighbors_.assign(points_.size(), {});
    neighbors_.shrink_to_fit(); // no extra capacity retained globally

    dm.onEdge([this](size_t i, size_t j, double) {
        neighbors_[i].push_back(j);
        neighbors_[j].push_back(i);
    });

    dm_stats_ = dm.process(points_);

    // Post-process: cap neighbor lists
    for (auto& nbrs : neighbors_) {
        if (nbrs.size() > config_.maxNeighbors) {
            std::partial_sort(nbrs.begin(), nbrs.begin() + config_.maxNeighbors, nbrs.end());
            nbrs.resize(config_.maxNeighbors);
        }
        // Ensure uniqueness
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
}

void StreamingCechComplex::addSimplexToTree(const std::vector<size_t>& simplex, double filtrationValue) {
    if (!simplex_tree_) return;
    // pooled temp
    tda::core::Simplex* tmp = simplex_pool_.acquire(simplex.size());
    const std::vector<tda::core::Index>* src_ptr = nullptr;
    if (tmp) {
        auto& v = tmp->vertices(); v.clear(); v.reserve(simplex.size());
        for (auto idx : simplex) v.push_back(static_cast<tda::core::Index>(idx));
        tmp->setFiltrationValue(filtrationValue);
        src_ptr = &v;
    }
    std::vector<tda::core::Index> fallback;
    if (!src_ptr) {
        fallback.reserve(simplex.size());
        for (auto idx : simplex) fallback.push_back(static_cast<tda::core::Index>(idx));
        src_ptr = &fallback;
    }

    std::vector<Simplex_tree::Vertex_handle> s;
    s.reserve(src_ptr->size());
    for (auto v : *src_ptr) s.push_back(static_cast<Simplex_tree::Vertex_handle>(v));
    simplex_tree_->insert_simplex_and_subfaces(s, filtrationValue);

    if (tmp) simplex_pool_.release(tmp, src_ptr->size());
}

} // namespace tda::algorithms

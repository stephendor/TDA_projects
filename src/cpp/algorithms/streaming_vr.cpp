#include "tda/algorithms/streaming_vr.hpp"
#include "tda/utils/streaming_distance_matrix.hpp"
#include "tda/core/memory_monitor.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
#include <limits>
#include <atomic>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace tda::algorithms {

// Lifecycle

tda::core::Result<void> StreamingVRComplex::initialize(const PointContainer& points) {
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

tda::core::Result<void> StreamingVRComplex::computeComplex() {
    if (points_.empty() || !simplex_tree_) {
        return tda::core::Result<void>::failure("Must call initialize() first");
    }
    simplex_tree_->clear();

    // Build-phase telemetry start
    build_stats_ = BuildStats{};
    auto t0 = std::chrono::high_resolution_clock::now();
    build_stats_.memory_before_bytes = tda::core::MemoryMonitor::getCurrentMemoryUsage();
    build_stats_.peak_memory_bytes = build_stats_.memory_before_bytes;

    // Build neighbor adjacency using streaming distances under epsilon threshold
    buildNeighborsStreaming();

    // Add vertices
    if (build_stats_.simplex_count_by_dim.size() < 4) build_stats_.simplex_count_by_dim.assign(4, 0);
    for (size_t i = 0; i < points_.size(); ++i) {
        std::vector<size_t> v = {i};
        addSimplexToTree(v, 0.0);
        build_stats_.simplex_count_by_dim[0]++;
    }

    // Build higher-dimensional simplices up to config_.maxDimension using bounded neighbors
    const size_t N = points_.size();
    const size_t max_dim = std::min<size_t>(config_.maxDimension, 3);

    for (size_t i = 0; i < N; ++i) {
        auto& nbrs = neighbors_[i];
        // Cap neighbor list
        if (nbrs.size() > config_.maxNeighbors) nbrs.resize(config_.maxNeighbors);

        // 1-simplex edges implied by adjacency; add them
        for (size_t jidx = 0; jidx < nbrs.size(); ++jidx) {
            size_t j = nbrs[jidx];
            if (j <= i) continue; // avoid duplicates
            std::vector<size_t> e = {i, j};
            double filt = max_pairwise_distance(points_, e);
            addSimplexToTree(e, filt);
            if (build_stats_.simplex_count_by_dim.size() < 2) build_stats_.simplex_count_by_dim.resize(2, 0);
            build_stats_.simplex_count_by_dim[1]++;
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
                    if (std::find(neighbors_[j].begin(), neighbors_[j].end(), k) == neighbors_[j].end()) continue;
                    std::vector<size_t> tri = {i, j, k};
                    double filt = max_pairwise_distance(points_, tri);
                    addSimplexToTree(tri, filt);
                    if (build_stats_.simplex_count_by_dim.size() < 3) build_stats_.simplex_count_by_dim.resize(3, 0);
                    build_stats_.simplex_count_by_dim[2]++;
                }
            }
        }

        if (max_dim >= 3) {
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
                        if (build_stats_.simplex_count_by_dim.size() < 4) build_stats_.simplex_count_by_dim.resize(4, 0);
                        build_stats_.simplex_count_by_dim[3]++;
                    }
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    build_stats_.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    build_stats_.memory_after_bytes = tda::core::MemoryMonitor::getCurrentMemoryUsage();
    build_stats_.peak_memory_bytes = std::max(build_stats_.peak_memory_bytes, tda::core::MemoryMonitor::getCurrentMemoryUsage());
    build_stats_.num_simplices_built = simplex_tree_ ? static_cast<size_t>(simplex_tree_->num_simplices()) : 0;
    // SimplexPool telemetry
    {
        auto agg = simplex_pool_.getAggregateStats();
        build_stats_.pool_total_blocks = agg.first;
        build_stats_.pool_free_blocks = agg.second;
        build_stats_.pool_fragmentation = (agg.first > 0) ? (1.0 - (static_cast<double>(agg.second) / static_cast<double>(agg.first))) : 0.0;
        auto buckets = simplex_pool_.getAllBucketStats();
        build_stats_.pool_bucket_stats.clear();
        build_stats_.pool_bucket_stats.reserve(buckets.size());
        for (const auto& b : buckets) {
            build_stats_.pool_bucket_stats.emplace_back(b.vertex_count, std::make_pair(b.total_blocks, b.free_blocks));
        }
    }
    return tda::core::Result<void>::success();
}

tda::core::Result<void> StreamingVRComplex::computePersistence(int coefficientField) {
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

tda::core::Result<std::vector<tda::core::SimplexInfo>> StreamingVRComplex::getSimplices() const {
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

tda::core::Result<tda::core::ComplexStatistics> StreamingVRComplex::getStatistics() const {
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

void StreamingVRComplex::clear() {
    points_.clear();
    neighbors_.clear();
    simplex_tree_.reset();
    persistent_cohomology_.reset();
}

// Helpers

double StreamingVRComplex::euclidean(const Point& a, const Point& b) {
    double s = 0.0; for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; s += d * d; } return std::sqrt(s);
}

double StreamingVRComplex::max_pairwise_distance(const PointContainer& pts, const std::vector<size_t>& simplex) {
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

void StreamingVRComplex::buildNeighborsStreaming() {
    // Configure streaming distance matrix
    tda::utils::StreamingDistanceMatrix dm;
    auto dmcfg = config_.dm;
    // Set threshold as epsilon (override any placeholder)
    dmcfg.max_distance = config_.epsilon;
    // Respect parallel controls; we'll make callback thread-safe
    dmcfg.edge_callback_threadsafe = dmcfg.edge_callback_threadsafe || dmcfg.enable_parallel_threshold;
    dm.setConfig(dmcfg);

    const size_t N = points_.size();
    neighbors_.assign(N, {});
    const size_t K = std::max<size_t>(1, config_.maxNeighbors);
    std::vector<std::vector<std::pair<size_t,double>>> nbr_tmp(N);
    for (size_t i = 0; i < N; ++i) nbr_tmp[i].reserve(K);

    // Lightweight per-vertex spinlocks
    std::vector<std::atomic_flag> locks(N);
    for (size_t i = 0; i < N; ++i) locks[i].clear();

    auto add_bounded = [&](size_t u, size_t v, double dist) {
        auto& vec = nbr_tmp[u];
        if (vec.size() < K) {
            vec.emplace_back(v, dist);
            return;
        }
        // Find worst (max distance) and replace if this is closer
        size_t worst_idx = 0; double worst_d = -std::numeric_limits<double>::infinity();
        for (size_t t = 0; t < vec.size(); ++t) {
            if (vec[t].second > worst_d) { worst_d = vec[t].second; worst_idx = t; }
        }
        if (dist < worst_d) vec[worst_idx] = {v, dist};
    };

    dm.onEdge([&](size_t i, size_t j, double d) {
        // Thread-safe updates
        while (locks[i].test_and_set(std::memory_order_acquire)) { /* spin */ }
        add_bounded(i, j, d);
        locks[i].clear(std::memory_order_release);
        while (locks[j].test_and_set(std::memory_order_acquire)) { /* spin */ }
        add_bounded(j, i, d);
        locks[j].clear(std::memory_order_release);
    });

    dm_stats_ = dm.process(points_);

    // Convert to index-only neighbor lists, sorted by vertex id
    for (size_t i = 0; i < N; ++i) {
        auto& tmp = nbr_tmp[i];
        neighbors_[i].reserve(tmp.size());
        for (auto& p : tmp) neighbors_[i].push_back(p.first);
        std::sort(neighbors_[i].begin(), neighbors_[i].end());
        neighbors_[i].erase(std::unique(neighbors_[i].begin(), neighbors_[i].end()), neighbors_[i].end());
    }

    // Build adjacency histogram telemetry
    size_t max_deg = 0;
    for (const auto& v : neighbors_) max_deg = std::max(max_deg, v.size());
    build_stats_.adjacency_max_degree = max_deg;
    build_stats_.adjacency_histogram.assign(max_deg + 1, 0);
    for (const auto& v : neighbors_) {
        build_stats_.adjacency_histogram[v.size()]++;
    }
}

void StreamingVRComplex::addSimplexToTree(const std::vector<size_t>& simplex, double filtrationValue) {
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

    // Update peak memory during build (cheap snapshot)
    build_stats_.peak_memory_bytes = std::max(build_stats_.peak_memory_bytes, tda::core::MemoryMonitor::getCurrentMemoryUsage());

    if (tmp) simplex_pool_.release(tmp, src_ptr->size());
}

} // namespace tda::algorithms

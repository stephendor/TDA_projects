// Streaming distance matrix computation with bounded memory footprint.
#pragma once

#include <vector>
#include <functional>
#include <string>
#include <cstddef>
#include <chrono>

namespace tda::utils {

struct StreamingDMConfig {
    size_t block_size = 64;           // computation block size
    bool symmetric = true;            // treat matrix as symmetric
    double max_distance = -1.0;       // if >= 0, only report distances <= threshold
    bool use_parallel = true;         // enable OpenMP where compiled
    // Optional early-stop controls (0/<=0 means disabled)
    size_t max_blocks = 0;            // stop after processing this many blocks
    size_t max_pairs = 0;             // stop after computing this many pairs
    double time_limit_seconds = 0.0;  // stop after this many seconds
    // Parallelization control for threshold (edge) mode
    bool enable_parallel_threshold = true; // allow OpenMP in threshold mode
    bool edge_callback_threadsafe = false; // set true only if callback is thread-safe
    // Approximate kNN cap: if >0 in threshold mode, skip distance eval when both endpoints
    // already have >= K emitted edges. This is a soft cap and reduces work/memory.
    size_t knn_cap_per_vertex_soft = 0;    // 0 disables
    // If true, use per-thread local counters within each block and merge once per block
    // (reduces contention vs per-edge atomics, may introduce small bounded overshoot).
    bool softcap_local_merge = false;
};

struct StreamingDMStats {
    size_t total_points = 0;
    size_t total_blocks = 0;
    size_t total_pairs = 0;
    size_t emitted_edges = 0;
    double elapsed_seconds = 0.0;
    // Telemetry
    size_t memory_before_bytes = 0;
    size_t memory_after_bytes = 0;
    size_t peak_memory_bytes = 0;
    // Soft-cap telemetry (when knn_cap_per_vertex_soft > 0)
    size_t softcap_overshoot_sum = 0; // total vertices-by-amount over K observed during merges
    uint32_t softcap_overshoot_max = 0; // max per-vertex overshoot observed in merges
};

// Callback signatures
using EdgeCallback = std::function<void(size_t i, size_t j, double d)>;
using BlockCallback = std::function<void(size_t i0, size_t j0, const std::vector<std::vector<double>>& block)>;

// StreamingDistanceMatrix computes pairwise distances without storing the full matrix.
// It emits either filtered edges (threshold mode) or block tiles to a user-provided sink.
class StreamingDistanceMatrix {
public:
    StreamingDistanceMatrix() = default;
    explicit StreamingDistanceMatrix(const StreamingDMConfig& cfg) : config_(cfg) {}

    void setConfig(const StreamingDMConfig& cfg) { config_ = cfg; }
    const StreamingDMConfig& getConfig() const { return config_; }

    void onEdge(EdgeCallback cb) { edge_cb_ = std::move(cb); }
    void onBlock(BlockCallback cb) { block_cb_ = std::move(cb); }

    // Points are vectors of doubles of equal dimension.
    using Point = std::vector<double>;
    using PointContainer = std::vector<Point>;

    // Process an entire dataset, producing callbacks as we go. No full matrix is stored.
    StreamingDMStats process(const PointContainer& points);

private:
    StreamingDMConfig config_{};
    EdgeCallback edge_cb_{};
    BlockCallback block_cb_{};

    static double euclidean(const Point& a, const Point& b);
};

} // namespace tda::utils

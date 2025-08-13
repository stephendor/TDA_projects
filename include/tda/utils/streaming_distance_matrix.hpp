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

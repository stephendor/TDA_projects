#include <cassert>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "tda/utils/streaming_distance_matrix.hpp"

using Point = std::vector<double>;
using Points = std::vector<Point>;

static Points make_points(size_t n, size_t d) {
    std::mt19937 rng(321);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Points pts(n, Point(d));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) pts[i][j] = dist(rng);
    return pts;
}

int main() {
    const size_t n = 256;
    const size_t d = 3;
    auto pts = make_points(n, d);

    tda::utils::StreamingDMConfig s_cfg;
    s_cfg.block_size = 16;
    s_cfg.symmetric = true;
    s_cfg.max_distance = -1.0; // block mode
    s_cfg.use_parallel = true;

    size_t blocks_seen = 0;
    tda::utils::StreamingDistanceMatrix sdm(s_cfg);
    sdm.onBlock([&](size_t i0, size_t j0, const std::vector<std::vector<double>>& block){
        (void)i0; (void)j0; (void)block; // no-op sink
        ++blocks_seen;
    });

    auto stats = sdm.process(pts);

    // Basic stat checks
    assert(stats.total_points == n);
    assert(stats.total_pairs > 0);
    assert(stats.total_blocks > 0);
    assert(stats.elapsed_seconds >= 0.0);

    // Telemetry checks (weak/non-brittle)
    assert(stats.peak_memory_bytes >= stats.memory_before_bytes);
    assert(stats.memory_after_bytes >= 0);

    // Blocks callback should have been invoked for off-diagonal blocks
    assert(blocks_seen > 0);

    // Now switch to threshold/edge mode and ensure telemetry still records
    tda::utils::StreamingDMConfig e_cfg = s_cfg;
    e_cfg.max_distance = 0.75; // loose threshold to emit some edges

    size_t edges_seen = 0;
    tda::utils::StreamingDistanceMatrix sdm_edges(e_cfg);
    sdm_edges.onEdge([&](size_t i, size_t j, double d){
        (void)i; (void)j; (void)d; // no-op
        ++edges_seen;
    });

    auto e_stats = sdm_edges.process(pts);

    assert(e_stats.total_points == n);
    assert(e_stats.total_pairs > 0);
    assert(e_stats.emitted_edges > 0);
    assert(e_stats.peak_memory_bytes >= e_stats.memory_before_bytes);
    assert(e_stats.memory_after_bytes >= 0);

    std::cout << "StreamingDistanceMatrix telemetry test passed.\n"
              << "Blocks seen: " << blocks_seen
              << ", Edges seen: " << edges_seen
              << ", Peak mem (bytes): " << e_stats.peak_memory_bytes
              << std::endl;
    return 0;
}

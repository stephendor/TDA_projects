#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include "tda/utils/streaming_distance_matrix.hpp"
#include "tda/utils/distance_matrix.hpp"

using Point = std::vector<double>;
using Points = std::vector<Point>;

static Points make_points(size_t n, size_t d) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Points pts(n, Point(d));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) pts[i][j] = dist(rng);
    }
    return pts;
}

int main() {
    // Small dataset for correctness
    const size_t n = 64;
    const size_t d = 4;
    auto pts = make_points(n, d);

    // Dense reference using existing utility (symmetric)
    tda::utils::DistanceMatrixConfig dm_cfg;
    dm_cfg.use_parallel = false;
    dm_cfg.use_simd = false;
    dm_cfg.symmetric = true;
    auto dense = tda::utils::ParallelDistanceMatrix(dm_cfg).compute(pts);

    // Threshold equals 0.75 quantile of distances to get a reasonable number of edges
    std::vector<double> all_dists;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) all_dists.push_back(dense.matrix[i][j]);
    }
    std::nth_element(all_dists.begin(), all_dists.begin() + all_dists.size() * 3 / 4, all_dists.end());
    double thr = all_dists[all_dists.size() * 3 / 4];

    // Collect edges from streaming mode
    std::vector<std::tuple<size_t,size_t,double>> edges_stream;
    tda::utils::StreamingDMConfig s_cfg;
    s_cfg.block_size = 16;
    s_cfg.symmetric = true;
    s_cfg.max_distance = thr;
    s_cfg.use_parallel = true;

    tda::utils::StreamingDistanceMatrix sdm(s_cfg);
    sdm.onEdge([&](size_t i, size_t j, double d){
        edges_stream.emplace_back(i, j, d);
    });

    auto stats = sdm.process(pts);

    // Validate that every emitted edge satisfies threshold and matches dense reference within tolerance
    for (const auto& e : edges_stream) {
        size_t i, j; double d;
        std::tie(i, j, d) = e;
        assert(i < n && j < n && i < j);
        assert(d <= thr + 1e-9);
        double ref = dense.matrix[i][j];
        assert(std::abs(ref - d) < 1e-9);
    }

    // Basic stats sanity
    assert(stats.total_points == n);
    assert(stats.total_pairs > 0);

    std::cout << "StreamingDistanceMatrix basic test passed. Edges: " << edges_stream.size()
              << ", pairs: " << stats.total_pairs << ", blocks: " << stats.total_blocks 
              << ", time: " << stats.elapsed_seconds << "s\n";
    return 0;
}

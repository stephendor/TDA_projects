#include <cassert>
#include <iostream>
#include <vector>
#include <random>

#include "tda/utils/streaming_distance_matrix.hpp"

using Point = std::vector<double>;
using Points = std::vector<Point>;

static Points make_points(size_t n, size_t d) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Points pts(n, Point(d));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) pts[i][j] = dist(rng);
    return pts;
}

int main() {
    // Choose n and block size to ensure multiple blocks exist
    const size_t n = 128;   // will form multiple tiles with block_size=32
    const size_t d = 3;
    auto pts = make_points(n, d);

    // Case A: Early stop by max_blocks
    {
        tda::utils::StreamingDMConfig cfg;
        cfg.block_size = 32;
        cfg.symmetric = true;
        cfg.max_distance = -1.0; // block mode
        cfg.use_parallel = false; // deterministic, makes pair/time early-stop exact
        cfg.max_blocks = 1;       // stop after exactly one block

        size_t blocks_seen = 0;
        tda::utils::StreamingDistanceMatrix sdm(cfg);
        sdm.onBlock([&](size_t i0, size_t j0, const std::vector<std::vector<double>>& blk){
            (void)i0; (void)j0; (void)blk; ++blocks_seen;
        });

        auto stats = sdm.process(pts);

        assert(stats.total_points == n);
        assert(stats.total_blocks == 1 && "max_blocks should cap processed blocks to 1");
        assert(blocks_seen == 1 && "block callback should run exactly once");
        assert(stats.total_pairs > 0);
        std::cout << "EarlyStop max_blocks: total_blocks=" << stats.total_blocks
                  << ", total_pairs=" << stats.total_pairs << "\n";
    }

    // Case B: Early stop by max_pairs (serial path -> exact equality)
    {
        tda::utils::StreamingDMConfig cfg;
        cfg.block_size = 32;
        cfg.symmetric = true;
        cfg.max_distance = -1.0; // block mode
        cfg.use_parallel = false; // ensure serial, so pair cap is exact
        cfg.max_pairs = 50;       // small cap to trigger mid-block break

        size_t blocks_seen = 0;
        tda::utils::StreamingDistanceMatrix sdm(cfg);
        sdm.onBlock([&](size_t i0, size_t j0, const std::vector<std::vector<double>>& blk){
            (void)i0; (void)j0; (void)blk; ++blocks_seen;
        });

        auto stats = sdm.process(pts);

        assert(stats.total_points == n);
        assert(stats.total_pairs == cfg.max_pairs && "serial path should hit exact max_pairs");
        assert(stats.total_blocks == 1 && "should finish the current block then stop");
        assert(blocks_seen == 1);
        std::cout << "EarlyStop max_pairs: total_blocks=" << stats.total_blocks
                  << ", total_pairs=" << stats.total_pairs << "\n";
    }

    std::cout << "StreamingDistanceMatrix early-stop tests passed.\n";
    return 0;
}

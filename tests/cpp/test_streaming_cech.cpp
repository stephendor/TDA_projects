#include "tda/algorithms/streaming_cech.hpp"
#include <cassert>
#include <iostream>

int main() {
    using tda::algorithms::StreamingCechComplex;
    // Small 2D grid
    StreamingCechComplex::PointContainer pts;
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            pts.push_back({double(x), double(y)});
        }
    }

    StreamingCechComplex::Config cfg;
    cfg.radius = 1.1; // adjacency if dist <= 2r -> <= 2.2 captures grid edges
    cfg.maxDimension = 2;
    cfg.maxNeighbors = 8;
    cfg.useAdaptiveRadius = false;

    StreamingCechComplex cech(cfg);
    auto r1 = cech.initialize(pts);
    if (r1.has_error()) { std::cerr << r1.error() << "\n"; return 1; }

    auto r2 = cech.computeComplex();
    if (r2.has_error()) { std::cerr << r2.error() << "\n"; return 1; }

    auto statsRes = cech.getStatistics();
    if (statsRes.has_error()) { std::cerr << statsRes.error() << "\n"; return 1; }
    auto stats = statsRes.value();

    // Basic sanity checks
    assert(stats.num_points == pts.size());
    assert(stats.num_simplices > 0);
    assert(stats.max_dimension >= 1);

    // Verify streaming telemetry is populated
    auto dmStats = cech.getStreamingStats();
    assert(dmStats.total_points == pts.size());
    assert(dmStats.total_blocks > 0);

    std::cout << "StreamingCech basic test passed.\n";
    return 0;
}

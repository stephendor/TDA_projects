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
    // Telemetry sanity
    assert(dmStats.peak_memory_bytes >= dmStats.memory_before_bytes);
    assert(dmStats.memory_after_bytes >= 0);

    // Verify build-phase telemetry
    auto b = cech.getBuildStats();
    assert(b.elapsed_seconds >= 0.0);
    assert(b.peak_memory_bytes >= b.memory_before_bytes);
    assert(b.num_simplices_built == stats.num_simplices);

    std::cout << "StreamingCech basic test passed.\n"
              << "Blocks: " << dmStats.total_blocks
              << ", Edges: " << dmStats.emitted_edges
              << ", Peak mem (bytes): " << dmStats.peak_memory_bytes
              << "\nBuild elapsed: " << b.elapsed_seconds
              << ", Build peak mem (bytes): " << b.peak_memory_bytes
              << std::endl;
    return 0;
}

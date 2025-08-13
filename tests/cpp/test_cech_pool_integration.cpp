#include <cassert>
#include <vector>
#include <iostream>

#include "tda/algorithms/cech_complex.hpp"

using Point = std::vector<double>;
using Points = std::vector<Point>;

static Points make_grid(size_t nx, size_t ny) {
    Points pts;
    pts.reserve(nx * ny);
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            pts.push_back(Point{static_cast<double>(i), static_cast<double>(j)});
        }
    }
    return pts;
}

int main() {
    // Small 3x3 grid in 2D
    auto pts = make_grid(3, 3);

    // Configure modest radius so nearest neighbors connect
    tda::algorithms::CechComplex::Config cfg;
    cfg.radius = 1.1;              // neighbors at distance 1 should connect
    cfg.radiusMultiplier = 1.25;
    cfg.maxDimension = 2;          // allow up to triangles
    cfg.maxNeighbors = 12;         // enough for 2D grid
    cfg.useAdaptiveRadius = false; // fix radius for predictable behavior
    cfg.useWitnessComplex = false;

    tda::algorithms::CechComplex cech(cfg);
    auto r0 = cech.initialize(pts);
    assert(r0.has_value());

    auto r1 = cech.computeComplex();
    assert(r1.has_value());

    // Basic sanity on simplices and stats
    auto simplicesRes = cech.getSimplices();
    assert(simplicesRes.has_value());
    const auto& simplices = simplicesRes.value();
    assert(!simplices.empty());

    auto statsRes = cech.getStatistics();
    assert(statsRes.has_value());
    const auto& stats = statsRes.value();
    assert(stats.num_points == pts.size());
    assert(stats.num_simplices >= pts.size()); // at least the vertices
    assert(stats.max_dimension >= 0);

    std::cout << "CechComplex pooling integration test passed: "
              << "points=" << stats.num_points
              << ", simplices=" << stats.num_simplices
              << ", max_dim=" << stats.max_dimension
              << std::endl;
    return 0;
}

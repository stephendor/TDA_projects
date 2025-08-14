// VR small-n parity test: compare StreamingVRComplex vs classic VietorisRips
#include "tda/algorithms/streaming_vr.hpp"
#include "tda/algorithms/vietoris_rips.hpp"
#include <cassert>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

using tda::algorithms::StreamingVRComplex;
using tda::algorithms::VietorisRips;

static std::vector<std::vector<double>> make_points(size_t n, size_t d, uint32_t seed=123)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<std::vector<double>> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> p(d);
        for (size_t j = 0; j < d; ++j) p[j] = dist(rng);
        pts.emplace_back(std::move(p));
    }
    return pts;
}

int main() {
    const size_t n = 60; // small-n for exact parity
    const size_t d = 3;
    const double eps = 0.9;
    const int maxDim = 2;
    auto pts = make_points(n, d);

    // Classic VR (baseline)
    VietorisRips classic;
    {
        auto r0 = classic.initialize(pts, eps, maxDim, 2);
        if (r0.has_error()) {
            std::cerr << "Classic VR init error: " << r0.error() << "\n"; return 1;
        }
        auto r1 = classic.computeComplex();
        if (r1.has_error()) {
            std::cerr << "Classic VR compute error: " << r1.error() << "\n"; return 1;
        }
        auto r2 = classic.computePersistence();
        if (r2.has_error()) {
            std::cerr << "Classic VR persistence error: " << r2.error() << "\n"; return 1;
        }
    }

    auto baseStatsRes = classic.getStatistics();
    if (baseStatsRes.has_error()) { std::cerr << baseStatsRes.error() << "\n"; return 1; }
    auto baseStats = baseStatsRes.value();

    // Streaming VR (with large neighbor cap to avoid truncation)
    StreamingVRComplex::Config cfg;
    cfg.epsilon = eps;
    cfg.maxDimension = static_cast<size_t>(maxDim);
    cfg.maxNeighbors = n; // ensure no cap for parity
    cfg.dm.block_size = 32;
    cfg.dm.symmetric = true;
    cfg.dm.max_distance = -1.0; // set from epsilon internally
    cfg.dm.enable_parallel_threshold = false; // deterministic

    StreamingVRComplex vr(cfg);
    {
        auto r0 = vr.initialize(pts);
        if (r0.has_error()) { std::cerr << r0.error() << "\n"; return 1; }
        auto r1 = vr.computeComplex();
        if (r1.has_error()) { std::cerr << r1.error() << "\n"; return 1; }
        auto r2 = vr.computePersistence(2);
        if (r2.has_error()) { std::cerr << r2.error() << "\n"; return 1; }
    }
    auto sStatsRes = vr.getStatistics();
    if (sStatsRes.has_error()) { std::cerr << sStatsRes.error() << "\n"; return 1; }
    auto sStats = sStatsRes.value();

    // Parity checks
    assert(sStats.num_points == baseStats.num_points);
    assert(sStats.max_dimension == baseStats.max_dimension);
    // Threshold should echo epsilon
    assert(std::abs(sStats.threshold - eps) < 1e-9);
    // Simplex counts by dimension should match exactly for small-n, no cap
    assert(sStats.simplex_count_by_dim.size() == baseStats.simplex_count_by_dim.size());
    for (size_t i = 0; i < sStats.simplex_count_by_dim.size(); ++i) {
        if (sStats.simplex_count_by_dim[i] != baseStats.simplex_count_by_dim[i]) {
            std::cerr << "Dim " << i << " mismatch: streaming=" << sStats.simplex_count_by_dim[i]
                      << " classic=" << baseStats.simplex_count_by_dim[i] << "\n";
            return 1;
        }
    }

    std::cout << "Streaming VR parity passed: simplices=" << sStats.num_simplices
              << ", dims=" << sStats.max_dimension << ", threshold=" << sStats.threshold << "\n";
    return 0;
}

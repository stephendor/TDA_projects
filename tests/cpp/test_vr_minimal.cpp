#include <iostream>
#include <vector>
#include <cassert>
#include "tda/algorithms/vietoris_rips.hpp"

int main() {
    // 4-point square in 2D
    std::vector<std::vector<double>> points = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}
    };
    double threshold = 1.5;
    int max_dim = 2;

    tda::algorithms::VietorisRips vr;

    auto init = vr.initialize(points, threshold, max_dim);
    assert(init.has_value());

    auto cplx = vr.computeComplex();
    assert(cplx.has_value());

    auto stats_res = vr.getStatistics();
    assert(stats_res.has_value());
    auto stats = stats_res.value();
    // Expect vertices=4, edges>=5 (including diagonals under 1.5, both diagonals length ~1.414), triangles=4
    assert(stats.num_points == 4);
    assert(stats.max_dimension >= 2);
    assert(stats.simplex_count_by_dim.size() >= 3);
    assert(stats.simplex_count_by_dim[0] == 4);
    assert(stats.simplex_count_by_dim[1] >= 5);
    assert(stats.simplex_count_by_dim[2] >= 1);

    auto pers = vr.computePersistence();
    assert(pers.has_value());

    auto pairs_res = vr.getPersistencePairs();
    assert(pairs_res.has_value());
    auto pairs = pairs_res.value();
    assert(!pairs.empty());

    // Basic Betti sanity: H0 should be 1 for connected square with threshold 1.5
    auto betti_res = vr.getBettiNumbers();
    assert(betti_res.has_value());
    auto betti = betti_res.value();
    assert(!betti.empty());
    assert(betti[0] == 1);

    std::cout << "VietorisRips minimal test passed\n";
    return 0;
}

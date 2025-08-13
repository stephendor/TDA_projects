#include "tda/algorithms/dtm_filtration.hpp"
#include <cassert>
#include <iostream>

int main() {
    // Small 2D cloud
    std::vector<std::vector<double>> pts = {
        {0,0},{1,0},{0,1},{1,1},{0.5,0.2}
    };

    tda::algorithms::DTMFiltration dtm({/*k*/3, /*power*/2.0, /*normalize*/true, /*maxDim*/10});
    auto r1 = dtm.initialize(pts);
    assert(!r1.has_error());
    auto r2 = dtm.computeDTMFunction();
    assert(!r2.has_error());
    assert(r2.value().size() == pts.size());
    auto r3 = dtm.buildFiltration(2);
    assert(!r3.has_error());

    auto statsRes = dtm.getStatistics();
    assert(!statsRes.has_error());
    auto stats = statsRes.value();
    assert(stats.num_points == pts.size());
    assert(stats.num_simplices > 0);
    assert(stats.max_dimension >= 0);

    std::cout << "DTM pooling integration: OK\n";
    return 0;
}

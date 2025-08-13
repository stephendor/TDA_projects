#include <cassert>
#include <iostream>
#include <vector>

#include "tda/core/simplex_pool.hpp"

using namespace tda::core;

int main() {
    SimplexPool pool(128);

    // Acquire several triangles (3-vertex simplices)
    std::vector<Simplex*> tris;
    for (int i = 0; i < 256; ++i) {
        Simplex* s = pool.acquire(3);
        assert(s != nullptr);
        s->vertices().push_back(0);
        s->vertices().push_back(1);
        s->vertices().push_back(2);
        s->setFiltrationValue(0.5);
        tris.push_back(s);
    }

    auto [total_before, free_before] = pool.getBucketStats(3);
    assert(total_before >= 256);
    assert(free_before <= total_before);

    // Release all
    for (auto* s : tris) pool.release(s, 3);

    auto [total_after, free_after] = pool.getBucketStats(3);
    assert(total_after == total_before);
    assert(free_after >= free_before);

    // Re-acquire and ensure we can reuse without reallocation
    Simplex* s2 = pool.acquire(3);
    assert(s2 != nullptr);
    assert(s2->vertices().capacity() >= 3);
    pool.release(s2, 3);

    std::cout << "SimplexPool basic test passed. total=" << total_after
              << " free=" << free_after << std::endl;
    return 0;
}

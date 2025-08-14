#include "tda/core/simplex_pool.hpp"

#include <cassert>

namespace tda::core {

SimplexPool::PoolT* SimplexPool::getOrCreatePool(size_t vertex_count) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pools_.find(vertex_count);
    if (it != pools_.end()) return it->second.get();

    auto pool = std::make_unique<PoolT>(block_size_, 1);
    PoolT* ptr = pool.get();
    pools_.emplace(vertex_count, std::move(pool));
    return ptr;
}

Simplex* SimplexPool::acquire(size_t vertex_count) {
    PoolT* pool = getOrCreatePool(vertex_count);
    Simplex* s = pool->allocate();
    if (!s) {
        // If allocation failed, grow by reserving another pool
        {
            std::lock_guard<std::mutex> lock(mtx_);
            pools_[vertex_count]->reserve(1);
        }
        s = pool->allocate();
    }
    if (s) {
        s->clear();
        // Reserve capacity to avoid later growth; fills will push_back up to vertex_count
        s->vertices().reserve(vertex_count);
        s->setFiltrationValue(0.0);
    }
    return s;
}

void SimplexPool::release(Simplex* s, size_t vertex_count) {
    if (!s) return;
    PoolT* pool = getOrCreatePool(vertex_count);
    // Clear heavy data before returning to pool
    s->clear();
    pool->deallocate(s);
}

std::pair<size_t, size_t> SimplexPool::getBucketStats(size_t vertex_count) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pools_.find(vertex_count);
    if (it == pools_.end()) return {0, 0};
    return it->second->getStats();
}

std::pair<size_t, size_t> SimplexPool::getAggregateStats() const {
    std::lock_guard<std::mutex> lock(mtx_);
    size_t total = 0, free = 0;
    for (const auto& kv : pools_) {
        auto stats = kv.second->getStats();
        total += stats.first;
        free  += stats.second;
    }
    return {total, free};
}

std::vector<SimplexPool::BucketStats> SimplexPool::getAllBucketStats() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<BucketStats> out;
    out.reserve(pools_.size());
    for (const auto& kv : pools_) {
        auto stats = kv.second->getStats();
        out.push_back(BucketStats{kv.first, stats.first, stats.second});
    }
    std::sort(out.begin(), out.end(), [](const BucketStats& a, const BucketStats& b){ return a.vertex_count < b.vertex_count; });
    return out;
}

} // namespace tda::core

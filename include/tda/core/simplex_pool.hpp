#pragma once

#include <unordered_map>
#include <memory>
#include <mutex>
#include <utility>
#include <cstddef>
#include <vector>

#include "tda/core/simplex.hpp"
#include "tda/utils/memory_pool.hpp"

namespace tda::core {

/**
 * SimplexPool provides reusable Simplex objects grouped by vertex count (arity).
 * This reduces allocation pressure and helps control memory fragmentation in
 * high-throughput filtration generation.
 */
class SimplexPool {
public:
    explicit SimplexPool(size_t pool_block_size = 1024)
        : block_size_(pool_block_size) {}

    // Acquire a simplex prepared for a given vertex count (capacity reserved).
    // Caller must release it back with the same vertex_count.
    Simplex* acquire(size_t vertex_count);

    // Release a previously acquired simplex back to the corresponding pool.
    void release(Simplex* s, size_t vertex_count);

    // Stats for a vertex_count bucket: returns {total_blocks, free_blocks}
    std::pair<size_t, size_t> getBucketStats(size_t vertex_count) const;

    // Stats across all buckets (sum of totals and frees)
    std::pair<size_t, size_t> getAggregateStats() const;

    struct BucketStats {
        size_t vertex_count;
        size_t total_blocks;
        size_t free_blocks;
    };
    // Enumerate all buckets' stats
    std::vector<BucketStats> getAllBucketStats() const;

private:
    using PoolT = tda::utils::MemoryPool<Simplex>;

    PoolT* getOrCreatePool(size_t vertex_count);

    size_t block_size_;
    std::unordered_map<size_t, std::unique_ptr<PoolT>> pools_;
    mutable std::mutex mtx_;
};

} // namespace tda::core

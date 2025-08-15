#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <cstddef>
#include <cassert>

namespace tda::utils {

/**
 * @brief Memory pool for efficient allocation of fixed-size blocks
 * 
 * This class provides a memory pool that pre-allocates memory blocks
 * and reuses them to reduce allocation overhead and fragmentation.
 */
template<typename T>
class MemoryPool {
private:
    struct Block {
        T data;
        bool in_use;
        
        Block() : data{}, in_use(false) {}
    };
    
    std::vector<std::unique_ptr<Block[]>> pools_;
    std::vector<Block*> free_blocks_;
    size_t block_size_;
    size_t pool_size_;
    mutable std::mutex mutex_;
    
public:
    /**
     * @brief Constructor
     * @param block_size Number of blocks per pool
     * @param initial_pools Number of initial pools to allocate
     */
    explicit MemoryPool(size_t block_size = 1024, size_t initial_pools = 1) 
        : block_size_(block_size), pool_size_(initial_pools) {
        for (size_t i = 0; i < initial_pools; ++i) {
            addPool();
        }
    }
    
    /**
     * @brief Allocate a block from the pool
     * @return Pointer to allocated block, or nullptr if pool is exhausted
     */
    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_blocks_.empty()) {
            addPool();
        }
        
        if (free_blocks_.empty()) {
            return nullptr; // Still no free blocks
        }
        
        Block* block = free_blocks_.back();
        free_blocks_.pop_back();
        block->in_use = true;
        
        return &block->data;
    }
    
    /**
     * @brief Deallocate a block back to the pool
     * @param ptr Pointer to the block to deallocate
     */
    void deallocate(T* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find the block containing this pointer
        for (auto& pool : pools_) {
            Block* block = pool.get();
            for (size_t i = 0; i < block_size_; ++i) {
                if (&block[i].data == ptr) {
                    if (block[i].in_use) {
                        block[i].in_use = false;
                        free_blocks_.push_back(&block[i]);
                    }
                    return;
                }
            }
        }
    }
    
    /**
     * @brief Get current pool statistics
     * @return Pair of (total_blocks, free_blocks)
     */
    std::pair<size_t, size_t> getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = pools_.size() * block_size_;
        size_t free = free_blocks_.size();
        return {total, free};
    }
    
    /**
     * @brief Reserve additional pools
     * @param num_pools Number of pools to add
     */
    void reserve(size_t num_pools) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < num_pools; ++i) {
            addPool();
        }
    }
    
    /**
     * @brief Clear all pools and free memory
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pools_.clear();
        free_blocks_.clear();
    }

    /**
     * @brief Get the number of allocated pools (pages)
     */
    size_t getPoolCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pools_.size();
    }

    /**
     * @brief Get the configured block size (blocks per pool)
     */
    size_t getBlockSize() const {
        // block_size_ is immutable after construction
        return block_size_;
    }

private:
    /**
     * @brief Add a new pool to the memory pool
     */
    void addPool() {
        auto pool = std::make_unique<Block[]>(block_size_);
        Block* pool_ptr = pool.get();
        
        // Add all blocks to free list
        for (size_t i = 0; i < block_size_; ++i) {
            free_blocks_.push_back(&pool_ptr[i]);
        }
        
        pools_.push_back(std::move(pool));
    }
};

/**
 * @brief Memory pool for variable-sized allocations
 * 
 * This class manages memory pools of different sizes to handle
 * variable-sized allocations efficiently.
 */
class VariableMemoryPool {
private:
    struct PoolInfo {
        size_t block_size;
        size_t block_count;
        std::unique_ptr<uint8_t[]> memory;
        std::vector<bool> used;
        std::vector<size_t> free_list;
    };
    
    std::vector<PoolInfo> pools_;
    mutable std::mutex mutex_;
    
    // Power-of-2 block sizes for efficient allocation
    static constexpr size_t MIN_BLOCK_SIZE = 8;
    static constexpr size_t MAX_BLOCK_SIZE = 4096;
    static constexpr size_t BLOCK_COUNT = 512;
    
public:
    /**
     * @brief Constructor
     */
    VariableMemoryPool() {
        // Initialize pools for different block sizes
        for (size_t size = MIN_BLOCK_SIZE; size <= MAX_BLOCK_SIZE; size *= 2) {
            pools_.push_back({size, BLOCK_COUNT, nullptr, {}, {}});
        }
        
        // Allocate memory for each pool
        for (auto& pool : pools_) {
            pool.memory = std::make_unique<uint8_t[]>(pool.block_size * pool.block_count);
            pool.used.resize(pool.block_count, false);
            
            // Initialize free list
            for (size_t i = 0; i < pool.block_count; ++i) {
                pool.free_list.push_back(i);
            }
        }
    }
    
    /**
     * @brief Allocate memory of specified size
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory, or nullptr if allocation fails
     */
    void* allocate(size_t size) {
        if (size == 0 || size > MAX_BLOCK_SIZE) {
            return nullptr;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find appropriate pool
        size_t pool_index = 0;
        size_t block_size = MIN_BLOCK_SIZE;
        while (block_size < size && pool_index < pools_.size() - 1) {
            block_size *= 2;
            pool_index++;
        }
        
        auto& pool = pools_[pool_index];
        if (pool.free_list.empty()) {
            return nullptr; // Pool exhausted
        }
        
        size_t block_index = pool.free_list.back();
        pool.free_list.pop_back();
        pool.used[block_index] = true;
        
        return pool.memory.get() + (block_index * pool.block_size);
    }
    
    /**
     * @brief Deallocate memory back to the pool
     * @param ptr Pointer to deallocate
     */
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find which pool this pointer belongs to
        for (auto& pool : pools_) {
            uint8_t* pool_start = pool.memory.get();
            uint8_t* pool_end = pool_start + (pool.block_size * pool.block_count);
            
            if (ptr >= pool_start && ptr < pool_end) {
                // Calculate block index
                size_t offset = static_cast<uint8_t*>(ptr) - pool_start;
                size_t block_index = offset / pool.block_size;
                
                if (pool.used[block_index]) {
                    pool.used[block_index] = false;
                    pool.free_list.push_back(block_index);
                }
                return;
            }
        }
    }
    
    /**
     * @brief Get memory pool statistics
     * @return Vector of (block_size, total_blocks, free_blocks) tuples
     */
    std::vector<std::tuple<size_t, size_t, size_t>> getStats() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        std::vector<std::tuple<size_t, size_t, size_t>> stats;
        
        for (const auto& pool : pools_) {
            size_t free_blocks = pool.free_list.size();
            size_t total_blocks = pool.block_count;
            stats.emplace_back(pool.block_size, total_blocks, free_blocks);
        }
        
        return stats;
    }
    
    /**
     * @brief Clear all pools and reset
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& pool : pools_) {
            std::fill(pool.used.begin(), pool.used.end(), false);
            pool.free_list.clear();
            
            // Reinitialize free list
            for (size_t i = 0; i < pool.block_count; ++i) {
                pool.free_list.push_back(i);
            }
        }
    }
};

/**
 * @brief RAII wrapper for memory pool allocations
 */
template<typename T>
class PoolAllocator {
private:
    MemoryPool<T>& pool_;
    T* ptr_;
    
public:
    explicit PoolAllocator(MemoryPool<T>& pool) : pool_(pool), ptr_(nullptr) {}
    
    ~PoolAllocator() {
        if (ptr_) {
            pool_.deallocate(ptr_);
        }
    }
    
    /**
     * @brief Allocate a block
     * @return True if allocation successful
     */
    bool allocate() {
        ptr_ = pool_.allocate();
        return ptr_ != nullptr;
    }
    
    /**
     * @brief Get the allocated pointer
     * @return Pointer to allocated block
     */
    T* get() const { return ptr_; }
    
    /**
     * @brief Release ownership without deallocating
     * @return Pointer to allocated block
     */
    T* release() {
        T* temp = ptr_;
        ptr_ = nullptr;
        return temp;
    }
    
    // Disable copy
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;
    
    // Enable move
    PoolAllocator(PoolAllocator&& other) noexcept 
        : pool_(other.pool_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    PoolAllocator& operator=(PoolAllocator&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                pool_.deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
};

} // namespace tda::utils

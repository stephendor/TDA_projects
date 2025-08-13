#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <optional>
#include <memory>

namespace tda::core {

// Forward declarations for conditional compilation
template<typename T> class ThreadSafeVector;
template<typename T> class ThreadSafeQueue;

#ifdef TDA_ENABLE_THREAD_SAFETY
    // Only include complex thread safety when explicitly requested
    template<typename T>
    using SafeVector = ThreadSafeVector<T>;
    
    template<typename T>
    using SafeQueue = ThreadSafeQueue<T>;
#else
    // Default to standard containers for single-threaded performance
    template<typename T>
    using SafeVector = std::vector<T>;
    
    template<typename T>
    using SafeQueue = std::queue<T>;
#endif

// Simple RAII lock helper for conditional thread safety
class ConditionalLock {
    std::unique_ptr<std::unique_lock<std::mutex>> lock_;
public:
    ConditionalLock(std::mutex* m = nullptr) 
        : lock_(m ? std::make_unique<std::unique_lock<std::mutex>>(*m) : nullptr) {}
    ~ConditionalLock() = default;
    
    // Non-copyable but moveable
    ConditionalLock(const ConditionalLock&) = delete;
    ConditionalLock& operator=(const ConditionalLock&) = delete;
    ConditionalLock(ConditionalLock&&) = default;
    ConditionalLock& operator=(ConditionalLock&&) = default;
};

/**
 * @brief Thread-safe vector implementation (only compiled when TDA_ENABLE_THREAD_SAFETY is defined)
 */
template<typename T>
class ThreadSafeVector {
private:
    std::vector<T> data_;
    mutable std::shared_mutex mutex_;

public:
    ThreadSafeVector() = default;
    
    void push_back(const T& value) {
        std::unique_lock lock(mutex_);
        data_.push_back(value);
    }
    
    void push_back(T&& value) {
        std::unique_lock lock(mutex_);
        data_.push_back(std::move(value));
    }
    
    template<typename... Args>
    void emplace_back(Args&&... args) {
        std::unique_lock lock(mutex_);
        data_.emplace_back(std::forward<Args>(args)...);
    }
    
    std::optional<T> at(size_t index) const {
        std::shared_lock lock(mutex_);
        if (index < data_.size()) {
            return data_[index];
        }
        return std::nullopt;
    }
    
    bool try_set(size_t index, const T& value) {
        std::unique_lock lock(mutex_);
        if (index < data_.size()) {
            data_[index] = value;
            return true;
        }
        return false;
    }
    
    size_t size() const {
        std::shared_lock lock(mutex_);
        return data_.size();
    }
    
    bool empty() const {
        std::shared_lock lock(mutex_);
        return data_.empty();
    }
    
    void clear() {
        std::unique_lock lock(mutex_);
        data_.clear();
    }
    
    void reserve(size_t capacity) {
        std::unique_lock lock(mutex_);
        data_.reserve(capacity);
    }
    
    // Thread-safe iteration
    template<typename Func>
    void for_each(Func&& f) const {
        std::shared_lock lock(mutex_);
        for (const auto& item : data_) {
            f(item);
        }
    }
    
    // Thread-safe modification
    template<typename Func>
    void modify_each(Func&& f) {
        std::unique_lock lock(mutex_);
        for (auto& item : data_) {
            f(item);
        }
    }
};

/**
 * @brief Thread-safe queue implementation (only compiled when TDA_ENABLE_THREAD_SAFETY is defined)
 */
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;

public:
    ThreadSafeQueue() = default;
    
    void push(const T& value) {
        {
            std::lock_guard lock(mutex_);
            queue_.push(value);
        }
        not_empty_.notify_one();
    }
    
    void push(T&& value) {
        {
            std::lock_guard lock(mutex_);
            queue_.push(std::move(value));
        }
        not_empty_.notify_one();
    }
    
    template<typename... Args>
    void emplace(Args&&... args) {
        {
            std::lock_guard lock(mutex_);
            queue_.emplace(std::forward<Args>(args)...);
        }
        not_empty_.notify_one();
    }
    
    bool try_pop(T& value) {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    std::optional<T> try_pop() {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }
    
    // Blocking pop with timeout
    template<typename Rep, typename Period>
    bool wait_pop(T& value, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock lock(mutex_);
        if (!not_empty_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }
    
    void clear() {
        std::lock_guard lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }
};

} // namespace tda::core

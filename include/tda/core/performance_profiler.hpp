#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace tda::core {

/**
 * @brief High-resolution timer for precise performance measurement
 */
class HighResTimer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::nanoseconds;
    
    TimePoint start_time_;
    TimePoint end_time_;
    bool is_running_ = false;
    std::string name_;

public:
    explicit HighResTimer(const std::string& name = "Timer") : name_(name) {}
    
    void start() {
        start_time_ = Clock::now();
        is_running_ = true;
    }
    
    void stop() {
        if (is_running_) {
            end_time_ = Clock::now();
            is_running_ = false;
        }
    }
    
    void reset() {
        start_time_ = TimePoint{};
        end_time_ = TimePoint{};
        is_running_ = false;
    }
    
    double elapsed_nanoseconds() const {
        if (is_running_) {
            return std::chrono::duration_cast<Duration>(Clock::now() - start_time_).count();
        }
        return std::chrono::duration_cast<Duration>(end_time_ - start_time_).count();
    }
    
    double elapsed_microseconds() const {
        return elapsed_nanoseconds() / 1000.0;
    }
    
    double elapsed_milliseconds() const {
        return elapsed_nanoseconds() / 1000000.0;
    }
    
    double elapsed_seconds() const {
        return elapsed_nanoseconds() / 1000000000.0;
    }
    
    const std::string& name() const { return name_; }
    
    bool is_running() const { return is_running_; }
};

/**
 * @brief Memory usage tracker
 */
class MemoryTracker {
private:
#ifdef __linux__
    rusage usage_;
    size_t initial_memory_;
    size_t peak_memory_;
#endif

public:
    MemoryTracker() {
#ifdef __linux__
        getrusage(RUSAGE_SELF, &usage_);
        initial_memory_ = usage_.ru_maxrss * 1024; // Convert KB to bytes
        peak_memory_ = initial_memory_;
#endif
    }
    
    void update() {
#ifdef __linux__
        getrusage(RUSAGE_SELF, &usage_);
        size_t current_memory = usage_.ru_maxrss * 1024;
        peak_memory_ = std::max(peak_memory_, current_memory);
#endif
    }
    
    size_t current_memory_bytes() const {
#ifdef __linux__
        getrusage(RUSAGE_SELF, const_cast<rusage*>(&usage_));
        return usage_.ru_maxrss * 1024;
#else
        return 0;
#endif
    }
    
    size_t peak_memory_bytes() const {
#ifdef __linux__
        return peak_memory_;
#else
        return 0;
#endif
    }
    
    size_t memory_increase_bytes() const {
#ifdef __linux__
        return peak_memory_ - initial_memory_;
#else
        return 0;
#endif
    }
    
    double current_memory_mb() const {
        return current_memory_bytes() / (1024.0 * 1024.0);
    }
    
    double peak_memory_mb() const {
        return peak_memory_bytes() / (1024.0 * 1024.0);
    }
    
    double memory_increase_mb() const {
        return memory_increase_bytes() / (1024.0 * 1024.0);
    }
};

/**
 * @brief Performance measurement session
 */
class PerformanceSession {
private:
    std::string session_name_;
    std::vector<std::unique_ptr<HighResTimer>> timers_;
    std::unordered_map<std::string, HighResTimer*> active_timers_;
    MemoryTracker memory_tracker_;
    std::vector<std::pair<std::string, double>> measurements_;
    
public:
    explicit PerformanceSession(const std::string& name) : session_name_(name) {}
    
    /**
     * @brief Start timing a named operation
     */
    void start_timer(const std::string& name) {
        auto timer = std::make_unique<HighResTimer>(name);
        active_timers_[name] = timer.get();
        timer->start();
        timers_.push_back(std::move(timer));
    }
    
    /**
     * @brief Stop timing a named operation
     */
    void stop_timer(const std::string& name) {
        auto it = active_timers_.find(name);
        if (it != active_timers_.end()) {
            it->second->stop();
            active_timers_.erase(it);
        }
    }
    
    /**
     * @brief Record a custom measurement
     */
    void record_measurement(const std::string& name, double value) {
        measurements_.emplace_back(name, value);
    }
    
    /**
     * @brief Update memory tracking
     */
    void update_memory() {
        memory_tracker_.update();
    }
    
    /**
     * @brief Get current memory usage
     */
    double get_current_memory_mb() const {
        return memory_tracker_.current_memory_mb();
    }
    
    /**
     * @brief Get peak memory usage
     */
    double get_peak_memory_mb() const {
        return memory_tracker_.peak_memory_mb();
    }
    
    /**
     * @brief Get memory increase
     */
    double get_memory_increase_mb() const {
        return memory_tracker_.memory_increase_mb();
    }
    
    /**
     * @brief Generate performance report
     */
    std::string generate_report() const {
        std::ostringstream oss;
        oss << "=== Performance Report: " << session_name_ << " ===" << std::endl;
        oss << std::endl;
        
        // Timer results
        if (!timers_.empty()) {
            oss << "Timing Results:" << std::endl;
            oss << std::setw(30) << std::left << "Operation" 
                << std::setw(15) << std::right << "Time (ms)" 
                << std::setw(15) << std::right << "Time (μs)" << std::endl;
            oss << std::string(60, '-') << std::endl;
            
            for (const auto& timer : timers_) {
                if (!timer->is_running()) {
                    oss << std::setw(30) << std::left << timer->name()
                        << std::setw(15) << std::right << std::fixed << std::setprecision(3) 
                        << timer->elapsed_milliseconds()
                        << std::setw(15) << std::right << std::fixed << std::setprecision(3) 
                        << timer->elapsed_microseconds() << std::endl;
                }
            }
            oss << std::endl;
        }
        
        // Memory results
        oss << "Memory Usage:" << std::endl;
        oss << std::setw(30) << std::left << "Current Memory" 
            << std::setw(15) << std::right << std::fixed << std::setprecision(2) 
            << get_current_memory_mb() << " MB" << std::endl;
        oss << std::setw(30) << std::left << "Peak Memory" 
            << std::setw(15) << std::right << std::fixed << std::setprecision(2) 
            << get_peak_memory_mb() << " MB" << std::endl;
        oss << std::setw(30) << std::left << "Memory Increase" 
            << std::setw(15) << std::right << std::fixed << std::setprecision(2) 
            << get_memory_increase_mb() << " MB" << std::endl;
        oss << std::endl;
        
        // Custom measurements
        if (!measurements_.empty()) {
            oss << "Custom Measurements:" << std::endl;
            for (const auto& [name, value] : measurements_) {
                oss << std::setw(30) << std::left << name 
                    << std::setw(15) << std::right << std::fixed << std::setprecision(6) 
                    << value << std::endl;
            }
            oss << std::endl;
        }
        
        return oss.str();
    }
    
    /**
     * @brief Print performance report to console
     */
    void print_report() const {
        std::cout << generate_report();
    }
    
    /**
     * @brief Save performance report to file
     */
    bool save_report(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << generate_report();
            file.close();
            return true;
        }
        return false;
    }
    
    /**
     * @brief Get timer by name
     */
    HighResTimer* get_timer(const std::string& name) {
        auto it = active_timers_.find(name);
        return (it != active_timers_.end()) ? it->second : nullptr;
    }
    
    /**
     * @brief Get all timers
     */
    const std::vector<std::unique_ptr<HighResTimer>>& get_timers() const {
        return timers_;
    }
    
    /**
     * @brief Get measurement by name
     */
    double get_measurement(const std::string& name) const {
        for (const auto& [meas_name, value] : measurements_) {
            if (meas_name == name) {
                return value;
            }
        }
        return 0.0; // Default value if not found
    }
    
    /**
     * @brief Check if timer is active
     */
    bool is_timer_active(const std::string& name) const {
        return active_timers_.find(name) != active_timers_.end();
    }
    
    /**
     * @brief Clear all timers and measurements
     */
    void clear() {
        timers_.clear();
        active_timers_.clear();
        measurements_.clear();
    }
};

/**
 * @brief RAII wrapper for automatic timer management
 */
class ScopedTimer {
private:
    PerformanceSession& session_;
    std::string timer_name_;
    
public:
    ScopedTimer(PerformanceSession& session, const std::string& name) 
        : session_(session), timer_name_(name) {
        session_.start_timer(timer_name_);
    }
    
    ~ScopedTimer() {
        session_.stop_timer(timer_name_);
    }
    
    // Disable copy and assignment
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};

/**
 * @brief Convenience macro for automatic timing
 */
#define TDA_PROFILE(session, name) \
    tda::core::ScopedTimer tda_timer_##__LINE__(session, name)

/**
 * @brief Convenience macro for timing a specific block
 */
#define TDA_PROFILE_BLOCK(session, name) \
    tda::core::ScopedTimer tda_block_timer_##__LINE__(session, name)

} // namespace tda::core

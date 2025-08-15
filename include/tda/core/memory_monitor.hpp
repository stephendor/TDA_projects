#pragma once

#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/task_info.h>
#else
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace tda::core {

/**
 * @brief Cross-platform memory monitoring utility
 */
class MemoryMonitor {
public:
    /**
     * @brief Get current memory usage in bytes (cached for performance)
     */
    static size_t getCurrentMemoryUsage() {
        static thread_local size_t cached_usage = 0;
        static thread_local std::chrono::steady_clock::time_point last_update;
        
        auto now = std::chrono::steady_clock::now();
        // Only update every 100ms to reduce overhead
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() > 100) {
            cached_usage = getCurrentMemoryUsageImpl();
            last_update = now;
        }
        return cached_usage;
    }

private:
    /**
     * @brief Implementation of memory usage retrieval (moved to private for caching)
     */
    static size_t getCurrentMemoryUsageImpl() {
        #ifdef _WIN32
            PROCESS_MEMORY_COUNTERS_EX pmc;
            if (GetProcessMemoryInfo(GetCurrentProcess(), 
                                   reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), 
                                   sizeof(pmc))) {
                return pmc.WorkingSetSize;
            }
            return 0;
        #elif defined(__APPLE__)
            mach_task_basic_info info;
            mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
            if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, 
                         reinterpret_cast<task_info_t>(&info), 
                         &count) == KERN_SUCCESS) {
                return info.resident_size;
            }
            return 0;
        #else
            // Linux - use getrusage instead of file I/O for better performance
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                return usage.ru_maxrss * 1024; // Convert KB to bytes
            }
            return 0;
        #endif
    }

public:

    /**
     * @brief Get peak memory usage in bytes (cached to avoid file I/O overhead)
     */
    static size_t getPeakMemoryUsage() {
        static thread_local size_t cached_peak = 0;
        static thread_local std::chrono::steady_clock::time_point last_peak_update;
        
        auto now = std::chrono::steady_clock::now();
        // Only update peak every 1 second to reduce overhead
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_peak_update).count() > 1) {
            cached_peak = getPeakMemoryUsageImpl();
            last_peak_update = now;
        }
        return cached_peak;
    }

private:
    /**
     * @brief Implementation of peak memory usage retrieval (optimized for performance)
     */
    static size_t getPeakMemoryUsageImpl() {
        #ifdef _WIN32
            PROCESS_MEMORY_COUNTERS_EX pmc;
            if (GetProcessMemoryInfo(GetCurrentProcess(), 
                                   reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), 
                                   sizeof(pmc))) {
                return pmc.PeakWorkingSetSize;
            }
            return 0;
        #elif defined(__APPLE__)
            rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                return usage.ru_maxrss * 1024; // Convert KB to bytes
            }
            return 0;
        #else
            // Linux - Use getrusage (no file I/O for better performance)
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                return usage.ru_maxrss * 1024; // Convert KB to bytes (ru_maxrss is KB on Linux)
            }
            return 0;
        #endif
    }

public:

    /**
     * @brief Get available system memory in bytes (cached to avoid file I/O overhead)
     */
    static size_t getAvailableSystemMemory() {
        static thread_local size_t cached_available = 0;
        static thread_local std::chrono::steady_clock::time_point last_available_update;
        
        auto now = std::chrono::steady_clock::now();
        // Only update available memory every 5 seconds (it changes slowly)
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_available_update).count() > 5) {
            cached_available = getAvailableSystemMemoryImpl();
            last_available_update = now;
        }
        return cached_available;
    }

private:
    /**
     * @brief Implementation of available system memory retrieval
     */
    static size_t getAvailableSystemMemoryImpl() {
        #ifdef _WIN32
            MEMORYSTATUSEX memInfo;
            memInfo.dwLength = sizeof(MEMORYSTATUSEX);
            if (GlobalMemoryStatusEx(&memInfo)) {
                return memInfo.ullAvailPhys;
            }
            return 0;
        #elif defined(__APPLE__)
            mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
            vm_statistics_data_t vmstat;
            if (host_statistics(mach_host_self(), HOST_VM_INFO,
                              reinterpret_cast<host_info_t>(&vmstat), &count) == KERN_SUCCESS) {
                return vmstat.free_count * static_cast<size_t>(getpagesize());
            }
            return 0;
        #else
            // Linux - Use system call instead of file I/O for better performance
            struct sysinfo si;
            if (sysinfo(&si) == 0) {
                return si.freeram * si.mem_unit;
            }
            return 0;
        #endif
    }

public:

    /**
     * @brief Check if enough memory is available
     * @param required_bytes Required memory in bytes
     * @param safety_factor Multiplier for required memory (default: 2.0)
     */
    static bool isMemoryAvailable(size_t required_bytes, double safety_factor = 2.0) {
        size_t available = getAvailableSystemMemory();
        return available >= static_cast<size_t>(required_bytes * safety_factor);
    }

    /**
     * @brief Get memory usage as string with units
     * @param bytes Memory size in bytes
     */
    static std::string formatMemorySize(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB"};
        size_t unit_idx = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unit_idx < 3) {
            size /= 1024.0;
            unit_idx++;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
        return oss.str();
    }
};

} // namespace tda::core

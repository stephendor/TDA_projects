#pragma once

#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/task_info.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace tda::core {

/**
 * @brief Cross-platform memory monitoring utility
 */
class MemoryMonitor {
public:
    /**
     * @brief Get current memory usage in bytes
     */
    static size_t getCurrentMemoryUsage() {
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
            // Linux implementation
            std::ifstream status_file("/proc/self/status");
            std::string line;
            size_t memory_kb = 0;
            
            if (status_file.is_open()) {
                while (std::getline(status_file, line)) {
                    if (line.substr(0, 6) == "VmRSS:") {
                        std::istringstream iss(line.substr(7));
                        iss >> memory_kb;
                        break;
                    }
                }
                status_file.close();
            }
            return memory_kb * 1024; // Convert KB to bytes
        #endif
    }

    /**
     * @brief Get peak memory usage in bytes
     */
    static size_t getPeakMemoryUsage() {
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
            // Linux implementation
            std::ifstream status_file("/proc/self/status");
            std::string line;
            size_t memory_kb = 0;
            
            if (status_file.is_open()) {
                while (std::getline(status_file, line)) {
                    if (line.substr(0, 7) == "VmHWM:") {
                        std::istringstream iss(line.substr(8));
                        iss >> memory_kb;
                        break;
                    }
                }
                status_file.close();
            }
            return memory_kb * 1024; // Convert KB to bytes
        #endif
    }

    /**
     * @brief Get available system memory in bytes
     */
    static size_t getAvailableSystemMemory() {
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
            // Linux implementation
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            size_t available_kb = 0;
            
            if (meminfo.is_open()) {
                while (std::getline(meminfo, line)) {
                    if (line.substr(0, 13) == "MemAvailable:") {
                        std::istringstream iss(line.substr(14));
                        iss >> available_kb;
                        break;
                    }
                }
                meminfo.close();
            }
            return available_kb * 1024; // Convert KB to bytes
        #endif
    }

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

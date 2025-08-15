#include "tda/core/types.hpp"

namespace tda::core {
    // Implementation of core types
    // This is a minimal implementation to get the build working
    
    // Placeholder implementations - will be expanded later
    bool is_power_of_two(std::size_t n) noexcept {
        return n > 0 && (n & (n - 1)) == 0;
    }
    
    std::size_t next_power_of_two(std::size_t n) noexcept {
        if (n <= 1) return 1;
        std::size_t power = 1;
        while (power < n) power <<= 1;
        return power;
    }
    
    std::size_t aligned_size(std::size_t count) noexcept {
        return ((count + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;
    }
}

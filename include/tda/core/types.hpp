#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <concepts>
#include <ranges>
#include <span>
#include <cmath>
#include <numeric>
#include <variant>

namespace tda::core {

// Forward declarations
class PointCloud;
class Simplex;
class Filtration;
class PersistentHomology;

// Type aliases for common types
using Index = std::size_t;
using Dimension = std::uint32_t;
using Birth = double;
using Death = double;
using Persistence = double;

// SIMD-friendly vector types
using Vector2D = std::array<double, 2>;
using Vector3D = std::array<double, 3>;
using Vector4D = std::array<double, 4>;

// Dynamic vector type
using DynamicVector = std::vector<double>;

// Concepts for type constraints
template<typename T>
concept PointType = requires(T t) {
    { t[0] } -> std::convertible_to<double>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept ContainerType = requires(T t) {
    typename T::value_type;
    typename T::iterator;
    { t.begin() } -> std::input_iterator;
    { t.end() } -> std::input_iterator;
    { t.size() } -> std::convertible_to<std::size_t>;
};

// Persistence pair structure
struct PersistencePair {
    Birth birth;
    Death death;
    Dimension dimension;
    Index birth_simplex;
    Index death_simplex;
    
    constexpr PersistencePair(Birth b, Death d, Dimension dim, Index b_sim, Index d_sim)
        : birth(b), death(d), dimension(dim), birth_simplex(b_sim), death_simplex(d_sim) {}
    
    constexpr Persistence get_persistence() const noexcept {
        return death - birth;
    }
    
    constexpr bool is_finite() const noexcept {
        return std::isfinite(death);
    }
    
    constexpr bool is_infinite() const noexcept {
        return !std::isfinite(death);
    }
};

// Betti numbers structure
struct BettiNumbers {
    std::vector<Index> betti;
    
    BettiNumbers() = default;
    explicit BettiNumbers(Dimension max_dim) : betti(max_dim + 1, 0) {}
    
    Index& operator[](Dimension dim) { return betti[dim]; }
    const Index& operator[](Dimension dim) const { return betti[dim]; }
    
    Dimension max_dimension() const noexcept { return static_cast<Dimension>(betti.size() - 1); }
    Index total_betti() const noexcept { 
        return std::accumulate(betti.begin(), betti.end(), Index{0}); 
    }
};

// Error handling types
enum class ErrorCode {
    Success = 0,
    InvalidInput,
    MemoryAllocationFailed,
    ComputationFailed,
    NotImplemented
};

// Result type for error handling
template<typename T>
class Result {
    std::variant<T, ErrorCode> data_;
    
public:
    Result(T value) : data_(std::move(value)) {}
    Result(ErrorCode error) : data_(error) {}
    
    bool is_success() const noexcept { return std::holds_alternative<T>(data_); }
    bool is_error() const noexcept { return std::holds_alternative<ErrorCode>(data_); }
    
    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }
    
    ErrorCode error() const { return std::get<ErrorCode>(data_); }
    
    T& operator*() { return value(); }
    const T& operator*() const { return value(); }
    
    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }
};

// Specialization for void
template<>
class Result<void> {
    ErrorCode error_;
    bool is_success_;
    
public:
    Result() : error_(ErrorCode::Success), is_success_(true) {}
    Result(ErrorCode error) : error_(error), is_success_(false) {}
    
    bool is_success() const noexcept { return is_success_; }
    bool is_error() const noexcept { return !is_success_; }
    
    ErrorCode error() const { return error_; }
};

// Utility functions
template<typename T>
constexpr bool is_power_of_two(T n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

template<typename T>
constexpr T next_power_of_two(T n) noexcept {
    if (n <= 1) return 1;
    T power = 1;
    while (power < n) power <<= 1;
    return power;
}

// Memory alignment utilities
constexpr std::size_t SIMD_ALIGNMENT = 32;
constexpr std::size_t CACHE_LINE_SIZE = 64;

template<typename T>
constexpr std::size_t aligned_size(std::size_t count) noexcept {
    return ((count * sizeof(T) + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;
}

} // namespace tda::core

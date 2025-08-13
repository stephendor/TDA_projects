#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <concepts>
#include <ranges>
#include <span>
#include <variant> // Added for Result type
#include <numeric> // Added for std::accumulate
#include <cmath> // Added for std::isfinite
#include <algorithm> // Added for std::min, std::max

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
    
    // Default constructor
    constexpr PersistencePair() : birth(0.0), death(0.0), dimension(0), birth_simplex(0), death_simplex(0) {}
    
    // Full constructor
    constexpr PersistencePair(Birth b, Death d, Dimension dim, Index b_sim, Index d_sim)
        : birth(b), death(d), dimension(dim), birth_simplex(b_sim), death_simplex(d_sim) {}
    
    // Simple constructor for TDA algorithms
    constexpr PersistencePair(Dimension dim, Birth b, Death d)
        : birth(b), death(d), dimension(dim), birth_simplex(0), death_simplex(0) {}
    
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
    std::variant<T, std::string> data_;
    
public:
    Result(T value) : data_(std::move(value)) {}
    Result(std::string error) : data_(std::move(error)) {}
    
    // Static factory methods
    static Result<T> success(T value) { return Result<T>(std::move(value)); }
    static Result<T> failure(std::string error) { return Result<T>(std::move(error)); }
    
    bool has_value() const noexcept { return std::holds_alternative<T>(data_); }
    bool has_error() const noexcept { return std::holds_alternative<std::string>(data_); }
    
    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }
    
    std::string error() const { return std::get<std::string>(data_); }
    
    T& operator*() { return value(); }
    const T& operator*() const { return value(); }
    
    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }
};

// Specialization for void
template<>
class Result<void> {
    std::string error_;
    bool is_success_;
    
public:
    Result() : error_(""), is_success_(true) {}
    Result(std::string error) : error_(std::move(error)), is_success_(false) {}
    
    // Static factory methods
    static Result<void> success() { return Result<void>(); }
    static Result<void> failure(std::string error) { return Result<void>(std::move(error)); }
    
    bool has_value() const noexcept { return is_success_; }
    bool has_error() const noexcept { return !is_success_; }
    
    std::string error() const { return error_; }
};

// Utility functions
template<typename T>
constexpr bool is_power_of_two(T n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

// Simplex information structure for TDA algorithms
struct SimplexInfo {
    int dimension;
    double filtration_value;
    std::vector<int> vertices;
    
    SimplexInfo() : dimension(0), filtration_value(0.0) {}
    SimplexInfo(int dim, double filt, std::vector<int> verts)
        : dimension(dim), filtration_value(filt), vertices(std::move(verts)) {}
};

// Complex statistics structure
struct ComplexStatistics {
    size_t num_points;
    size_t num_simplices;
    int max_dimension;
    double threshold;
    std::vector<int> simplex_count_by_dim;
    
    ComplexStatistics() : num_points(0), num_simplices(0), max_dimension(0), threshold(0.0) {}
};



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

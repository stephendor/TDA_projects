# Few-Shot Examples: Python ↔ C++23 Translations

## 🎯 **Overview**

This document provides specific examples of translating Python patterns to C++23 for the TDA Vector Stack project. Each example shows the Python reference implementation and its C++23 counterpart.

## 🔄 **Example 1: Point Cloud Processing**

### **Python Reference**
```python
def process_point_cloud(points: List[List[float]], max_distance: float) -> List[List[float]]:
    """Filter points by distance and normalize coordinates"""
    
    # Filter points within max_distance of origin
    filtered_points = [
        point for point in points 
        if sum(x*x for x in point) <= max_distance * max_distance
    ]
    
    # Normalize coordinates
    if filtered_points:
        max_coord = max(max(abs(x) for x in point) for point in filtered_points)
        normalized_points = [
            [x / max_coord for x in point] 
            for point in filtered_points
        ]
        return normalized_points
    
    return []
```

### **C++23 Implementation**
```cpp
#include <vector>
#include <ranges>
#include <span>
#include <algorithm>
#include <cmath>
#include <expected>

namespace tda::core {

std::expected<std::vector<std::vector<double>>, ProcessingError> 
processPointCloud(std::span<const std::vector<double>> points, double max_distance) {
    
    // Filter points within max_distance of origin using ranges
    auto filtered_points = points | std::views::filter([max_distance](const auto& point) {
        double sum_squares = std::accumulate(point.begin(), point.end(), 0.0, 
            [](double acc, double x) { return acc + x * x; });
        return sum_squares <= max_distance * max_distance;
    });
    
    // Convert filtered view to vector
    std::vector<std::vector<double>> filtered_vector(filtered_points.begin(), filtered_points.end());
    
    if (filtered_vector.empty()) {
        return std::vector<std::vector<double>>{};
    }
    
    // Find maximum coordinate value
    auto max_coord = std::ranges::max(
        filtered_vector | std::views::transform([](const auto& point) {
            return std::ranges::max(point | std::views::transform([](double x) { 
                return std::abs(x); 
            }));
        })
    );
    
    // Normalize coordinates using ranges
    auto normalized_points = filtered_vector | std::views::transform([max_coord](const auto& point) {
        std::vector<double> normalized;
        normalized.reserve(point.size());
        std::ranges::transform(point, std::back_inserter(normalized),
            [max_coord](double x) { return x / max_coord; });
        return normalized;
    });
    
    return std::vector<std::vector<double>>(normalized_points.begin(), normalized_points.end());
}

} // namespace tda::core
```

## 🔄 **Example 2: Persistence Diagram Analysis**

### **Python Reference**
```python
def analyze_persistence_diagram(diagram: List[Tuple[float, float, int]], 
                               epsilon: float) -> Dict[int, int]:
    """Count features by dimension within epsilon range"""
    
    # Group features by dimension
    features_by_dim = {}
    for birth, death, dimension in diagram:
        if birth <= epsilon <= death:
            if dimension not in features_by_dim:
                features_by_dim[dimension] = 0
            features_by_dim[dimension] += 1
    
    return features_by_dim

def compute_persistence_statistics(diagram: List[Tuple[float, float, int]]) -> Dict[str, float]:
    """Compute statistics about persistence features"""
    
    if not diagram:
        return {}
    
    # Extract persistence values (death - birth)
    persistences = [death - birth for birth, death, _ in diagram]
    
    # Compute statistics
    stats = {
        'mean_persistence': sum(persistences) / len(persistences),
        'max_persistence': max(persistences),
        'min_persistence': min(persistences),
        'total_features': len(diagram)
    }
    
    return stats
```

### **C++23 Implementation**
```cpp
#include <vector>
#include <ranges>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <expected>

namespace tda::core {

struct PersistenceFeature {
    double birth;
    double death;
    int dimension;
    
    double persistence() const { return death - birth; }
};

// Count features by dimension within epsilon range
std::expected<std::unordered_map<int, int>, AnalysisError> 
analyzePersistenceDiagram(std::span<const PersistenceFeature> diagram, double epsilon) {
    
    std::unordered_map<int, int> features_by_dim;
    
    // Use ranges to filter and group features
    auto active_features = diagram | std::views::filter([epsilon](const auto& feature) {
        return feature.birth <= epsilon && epsilon <= feature.death;
    });
    
    // Group by dimension using ranges
    for (const auto& feature : active_features) {
        features_by_dim[feature.dimension]++;
    }
    
    return features_by_dim;
}

// Compute persistence statistics
std::expected<std::unordered_map<std::string, double>, AnalysisError> 
computePersistenceStatistics(std::span<const PersistenceFeature> diagram) {
    
    if (diagram.empty()) {
        return std::unordered_map<std::string, double>{};
    }
    
    // Extract persistence values using ranges
    auto persistences = diagram | std::views::transform([](const auto& feature) {
        return feature.persistence();
    });
    
    // Convert to vector for analysis
    std::vector<double> persistence_vector(persistences.begin(), persistences.end());
    
    // Compute statistics using modern C++23
    auto [min_it, max_it] = std::ranges::minmax_element(persistence_vector);
    double mean_persistence = std::accumulate(persistence_vector.begin(), 
                                           persistence_vector.end(), 0.0) / persistence_vector.size();
    
    std::unordered_map<std::string, double> stats;
    stats["mean_persistence"] = mean_persistence;
    stats["max_persistence"] = *max_it;
    stats["min_persistence"] = *min_it;
    stats["total_features"] = static_cast<double>(diagram.size());
    
    return stats;
}

} // namespace tda::core
```

## 🔄 **Example 3: Vector Stack Configuration**

### **Python Reference**
```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class VectorStackConfig:
    enable_persistence_images: bool = True
    enable_landscapes: bool = True
    enable_sliced_wasserstein: bool = True
    
    image_resolution: int = 100
    landscape_resolution: int = 100
    sw_num_angles: int = 100
    
    intensity_normalize_images: bool = True
    use_log_lifetime_images: bool = False
    global_landscape_range: tuple[float, float] = (0.0, 1.0)

class VectorStack:
    def __init__(self, config: VectorStackConfig):
        self.config = config
        self.blocks = self._initialize_blocks()
    
    def _initialize_blocks(self) -> List[VectorizationBlock]:
        blocks = []
        
        if self.config.enable_persistence_images:
            blocks.append(PersistenceImageBlock(
                resolution=self.config.image_resolution,
                intensity_normalize=self.config.intensity_normalize_images,
                use_log_lifetime=self.config.use_log_lifetime_images
            ))
        
        if self.config.enable_landscapes:
            blocks.append(PersistenceLandscapeBlock(
                resolution=self.config.landscape_resolution,
                global_range=self.config.global_landscape_range
            ))
        
        return blocks
```

### **C++23 Implementation**
```cpp
#include <vector>
#include <memory>
#include <optional>
#include <string>

namespace tda::vector_stack {

// Configuration structure matching Python dataclass
struct VectorStackConfig {
    bool enable_persistence_images = true;
    bool enable_landscapes = true;
    bool enable_sliced_wasserstein = true;
    
    int image_resolution = 100;
    int landscape_resolution = 100;
    int sw_num_angles = 100;
    
    bool intensity_normalize_images = true;
    bool use_log_lifetime_images = false;
    std::pair<double, double> global_landscape_range = {0.0, 1.0};
};

// Base block interface using C++23 concepts
template<typename T>
concept VectorizationBlock = requires(T t, const PersistenceDiagram& pd) {
    { t.compute(pd) } -> std::convertible_to<std::vector<double>>;
    { t.name() } -> std::convertible_to<std::string>;
};

// Vector stack implementation
class VectorStack {
public:
    explicit VectorStack(const VectorStackConfig& config) 
        : config_(config) {
        initializeBlocks();
    }
    
    // Get configuration
    const VectorStackConfig& config() const { return config_; }
    
    // Get blocks
    const auto& blocks() const { return blocks_; }
    
private:
    void initializeBlocks() {
        if (config_.enable_persistence_images) {
            blocks_.push_back(std::make_unique<PersistenceImageBlock>(
                config_.image_resolution,
                config_.intensity_normalize_images,
                config_.use_log_lifetime_images
            ));
        }
        
        if (config_.enable_landscapes) {
            blocks_.push_back(std::make_unique<PersistenceLandscapeBlock>(
                config_.landscape_resolution,
                config_.global_landscape_range
            ));
        }
        
        if (config_.enable_sliced_wasserstein) {
            blocks_.push_back(std::make_unique<SlicedWassersteinBlock>(
                config_.sw_num_angles,
                AngleStrategy::Halton,
                config_.sw_resolution
            ));
        }
    }
    
    VectorStackConfig config_;
    std::vector<std::unique_ptr<VectorizationBlock>> blocks_;
};

} // namespace tda::vector_stack
```

## 🔄 **Example 4: Error Handling and Validation**

### **Python Reference**
```python
from typing import Union, Optional
from enum import Enum

class TDAError(Exception):
    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ValidationError(TDAError):
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")

def validate_point_cloud(points: List[List[float]]) -> Union[List[List[float]], ValidationError]:
    """Validate and clean point cloud data"""
    
    if not points:
        raise ValidationError("Point cloud cannot be empty")
    
    if len(points) > 1_000_000:
        raise ValidationError("Point cloud too large (max 1M points)")
    
    # Validate individual points
    cleaned_points = []
    for i, point in enumerate(points):
        if not isinstance(point, list):
            raise ValidationError(f"Point {i} is not a list")
        
        if len(point) < 2:
            raise ValidationError(f"Point {i} has insufficient dimensions")
        
        # Check for finite values
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in point):
            raise ValidationError(f"Point {i} contains non-finite values")
        
        cleaned_points.append(point)
    
    return cleaned_points
```

### **C++23 Implementation**
```cpp
#include <expected>
#include <string>
#include <vector>
#include <cmath>
#include <ranges>

namespace tda::core {

// Error codes using enum class
enum class ValidationErrorCode {
    Success = 0,
    EmptyPointCloud,
    PointCloudTooLarge,
    InvalidPointType,
    InsufficientDimensions,
    NonFiniteValues
};

// Error structure
struct ValidationError {
    ValidationErrorCode code;
    std::string message;
    
    ValidationError(ValidationErrorCode c, std::string msg) 
        : code(c), message(std::move(msg)) {}
};

// Validation function using std::expected
std::expected<std::vector<std::vector<double>>, ValidationError> 
validatePointCloud(std::span<const std::vector<double>> points) {
    
    if (points.empty()) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::EmptyPointCloud, 
            "Point cloud cannot be empty"
        ));
    }
    
    if (points.size() > 1'000'000) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::PointCloudTooLarge, 
            "Point cloud too large (max 1M points)"
        ));
    }
    
    // Validate individual points using ranges
    std::vector<std::vector<double>> cleaned_points;
    cleaned_points.reserve(points.size());
    
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        
        if (point.size() < 2) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InsufficientDimensions,
                "Point " + std::to_string(i) + " has insufficient dimensions"
            ));
        }
        
        // Check for finite values using ranges
        bool all_finite = std::ranges::all_of(point, [](double x) {
            return std::isfinite(x);
        });
        
        if (!all_finite) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::NonFiniteValues,
                "Point " + std::to_string(i) + " contains non-finite values"
            ));
        }
        
        cleaned_points.push_back(point);
    }
    
    return cleaned_points;
}

} // namespace tda::core
```

## 🔄 **Example 5: Performance Optimization with SIMD**

### **Python Reference**
```python
import numpy as np
from typing import List, Tuple

def compute_distances_batch(points: np.ndarray, query_point: np.ndarray) -> np.ndarray:
    """Compute distances from query point to all points in batch"""
    
    # Vectorized computation using NumPy
    diff = points - query_point
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    
    return distances

def find_nearest_neighbors(points: np.ndarray, query_point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors using distance computation"""
    
    distances = compute_distances_batch(points, query_point)
    
    # Get indices of k smallest distances
    indices = np.argpartition(distances, k)[:k]
    
    # Sort by distance
    sorted_indices = indices[np.argsort(distances[indices])]
    sorted_distances = distances[sorted_indices]
    
    return sorted_distances, sorted_indices
```

### **C++23 Implementation with SIMD**
```cpp
#include <vector>
#include <ranges>
#include <algorithm>
#include <immintrin.h> // For AVX2/AVX-512
#include <span>

namespace tda::optimization {

// SIMD-optimized distance computation
std::vector<double> computeDistancesBatch(
    std::span<const std::vector<double>> points,
    std::span<const double> query_point) {
    
    std::vector<double> distances;
    distances.reserve(points.size());
    
    const size_t dim = query_point.size();
    
    // Use SIMD for vectorized computation
    #ifdef __AVX2__
    if (dim == 3) {
        // Optimized for 3D points using AVX2
        __m256d query_vec = _mm256_set_pd(0.0, query_point[2], query_point[1], query_point[0]);
        
        for (const auto& point : points) {
            __m256d point_vec = _mm256_set_pd(0.0, point[2], point[1], point[0]);
            __m256d diff = _mm256_sub_pd(point_vec, query_vec);
            __m256d diff_squared = _mm256_mul_pd(diff, diff);
            
            // Horizontal sum
            double sum = _mm256_cvtsd_f64(_mm256_hadd_pd(diff_squared, diff_squared));
            distances.push_back(std::sqrt(sum));
        }
    } else
    #endif
    {
        // Fallback for other dimensions
        for (const auto& point : points) {
            double sum_squares = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                double diff = point[i] - query_point[i];
                sum_squares += diff * diff;
            }
            distances.push_back(std::sqrt(sum_squares));
        }
    }
    
    return distances;
}

// Find k nearest neighbors using modern C++23
std::pair<std::vector<double>, std::vector<size_t>> findNearestNeighbors(
    std::span<const std::vector<double>> points,
    std::span<const double> query_point,
    size_t k) {
    
    auto distances = computeDistancesBatch(points, query_point);
    
    // Create indices vector
    std::vector<size_t> indices(distances.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Partial sort to get k smallest elements
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&distances](size_t a, size_t b) { return distances[a] < distances[b]; });
    
    // Extract k nearest neighbors
    std::vector<double> k_distances;
    std::vector<size_t> k_indices;
    k_distances.reserve(k);
    k_indices.reserve(k);
    
    for (size_t i = 0; i < k; ++i) {
        k_distances.push_back(distances[indices[i]]);
        k_indices.push_back(indices[i]);
    }
    
    return {k_distances, k_indices};
}

} // namespace tda::optimization
```

## 🎯 **Key Translation Patterns**

### **1. Data Structures**
- **Python**: `List[List[float]]` → **C++23**: `std::vector<std::vector<double>>`
- **Python**: `Optional[T]` → **C++23**: `std::optional<T>`
- **Python**: `Dict[K, V]` → **C++23**: `std::unordered_map<K, V>`

### **2. Error Handling**
- **Python**: `raise Exception` → **C++23**: `std::expected<T, Error>`
- **Python**: `try/except` → **C++23**: `if (!result.has_value()) return std::unexpected(...)`

### **3. Iteration and Processing**
- **Python**: List comprehensions → **C++23**: `std::ranges` and `std::views`
- **Python**: `for point in points` → **C++23**: `for (const auto& point : points)`
- **Python**: `filter()` → **C++23**: `std::views::filter`

### **4. Performance Optimization**
- **Python**: NumPy vectorization → **C++23**: SIMD intrinsics and `std::ranges`
- **Python**: Built-in functions → **C++23**: `std::ranges::max`, `std::accumulate`
- **Python**: Memory management → **C++23**: RAII and smart pointers

## 🧪 **Testing Both Implementations**

### **Integration Test Example**
```cpp
// Test that C++ implementation matches Python reference
TEST_CASE("Python-C++ consistency") {
    // Create test data
    std::vector<std::vector<double>> points = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    std::vector<double> query = {0.0, 0.0};
    
    // C++ implementation
    auto cpp_result = tda::optimization::findNearestNeighbors(points, query, 2);
    
    // Python reference (through pybind11)
    auto python_result = python_module.find_nearest_neighbors(points, query, 2);
    
    // Verify consistency
    REQUIRE(cpp_result.first.size() == python_result.first.size());
    for (size_t i = 0; i < cpp_result.first.size(); ++i) {
        REQUIRE(std::abs(cpp_result.first[i] - python_result.first[i]) < 1e-10);
    }
}
```

---

## 🎯 **Next Steps**

1. **Implement C++23 versions** of all Python examples in the rules
2. **Create pybind11 bindings** for seamless integration
3. **Write bilingual tests** to ensure consistency
4. **Performance benchmark** both implementations
5. **Document translation patterns** for the team

---

*These examples demonstrate how to maintain the best of both worlds: Python's expressiveness for API design and C++23's performance for core computations.*

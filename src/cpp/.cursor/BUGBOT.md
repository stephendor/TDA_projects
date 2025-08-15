# TDA Platform - C++ Core Algorithms BugBot Rules

## C++ Core Architecture Overview
The C++ core provides **high-performance TDA algorithms** that must handle datasets with >1M points in <60 seconds. The codebase uses **C++23 standard** with modern features, SIMD optimization, and RAII patterns for memory safety.

## Core Algorithm Components

### 1. Algorithm Implementations
- **Vietoris-Rips Filtration**: Simplicial complex construction from point clouds
- **Alpha Complex**: Geometric filtration using Delaunay triangulation
- **Čech Complex**: Approximation algorithms for computational efficiency
- **DTM Filtration**: Distance-to-measure for noise-resistant analysis
- **Mapper Algorithm**: Topological network construction with customizable filters

### 2. Performance Requirements
- **Scalability**: Handle 1M+ points efficiently
- **Memory Usage**: Optimized data structures, avoid memory leaks
- **SIMD Optimization**: Vectorized operations for mathematical computations
- **Parallel Processing**: Multi-threading for independent operations

### 3. Integration Points
- **Python Bindings**: pybind11 integration for backend services
- **Data Exchange**: Efficient NumPy array handling
- **Error Handling**: Proper exception translation to Python

## C++23 Best Practices

### 1. Modern C++ Features
```cpp
// ✅ DO: Use C++23 features appropriately
#include <concepts>
#include <ranges>
#include <memory>
#include <span>

// Concepts for type constraints
template<typename T>
concept PointCloud = requires(T t) {
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.dimension() } -> std::convertible_to<std::size_t>;
    { t.point_at(0) } -> std::convertible_to<std::span<const double>>;
};

// RAII with smart pointers
class VietorisRipsFiltration {
private:
    std::unique_ptr<SimplicialComplex> complex_;
    std::vector<double> birth_values_;
    
public:
    explicit VietorisRipsFiltration(std::unique_ptr<SimplicialComplex> complex)
        : complex_(std::move(complex)) {}
    
    // No need for destructor - RAII handles cleanup
    ~VietorisRipsFiltration() = default;
    
    // Move semantics
    VietorisRipsFiltration(VietorisRipsFiltration&&) = default;
    VietorisRipsFiltration& operator=(VietorisRipsFiltration&&) = default;
    
    // Delete copy operations for unique ownership
    VietorisRipsFiltration(const VietorisRipsFiltration&) = delete;
    VietorisRipsFiltration& operator=(const VietorisRipsFiltration&) = delete;
};

// ❌ DON'T: Use outdated C++ patterns
class VietorisRipsFiltration {
private:
    SimplicialComplex* complex_;  // Bad: raw pointer
    
public:
    ~VietorisRipsFiltration() {  // Bad: manual cleanup
        delete complex_;
    }
};
```

### 2. Memory Management
```cpp
// ✅ DO: Use RAII and smart pointers
#include <memory>
#include <vector>

class PersistenceDiagram {
private:
    std::vector<std::unique_ptr<PersistencePair>> pairs_;
    std::unique_ptr<DiagramMetadata> metadata_;
    
public:
    void add_pair(std::unique_ptr<PersistencePair> pair) {
        pairs_.push_back(std::move(pair));
    }
    
    // Return by value for efficiency (move semantics)
    std::vector<std::unique_ptr<PersistencePair>> extract_pairs() {
        return std::move(pairs_);
    }
};

// Factory function returning unique_ptr
std::unique_ptr<FiltrationAlgorithm> create_filtration(
    FiltrationType type,
    const PointCloud& points
) {
    switch (type) {
        case FiltrationType::VietorisRips:
            return std::make_unique<VietorisRipsFiltration>(points);
        case FiltrationType::AlphaComplex:
            return std::make_unique<AlphaComplexFiltration>(points);
        default:
            throw std::invalid_argument("Unsupported filtration type");
    }
}

// ❌ DON'T: Manual memory management
class PersistenceDiagram {
private:
    PersistencePair** pairs_;  // Bad: raw pointer array
    int count_;
    
public:
    ~PersistenceDiagram() {  // Bad: manual cleanup
        for (int i = 0; i < count_; ++i) {
            delete pairs_[i];
        }
        delete[] pairs_;
    }
};
```

### 3. Exception Safety
```cpp
// ✅ DO: Provide strong exception guarantees
class TDAEngine {
private:
    std::unique_ptr<AlgorithmRegistry> algorithms_;
    std::unique_ptr<ResultCache> cache_;
    
public:
    // Strong exception guarantee: either succeeds completely or leaves unchanged
    void register_algorithm(std::string name, std::unique_ptr<Algorithm> algorithm) {
        auto new_algorithms = std::make_unique<AlgorithmRegistry>(*algorithms_);
        new_algorithms->add(name, std::move(algorithm));
        
        // Only commit changes if everything succeeds
        algorithms_ = std::move(new_algorithms);
    }
    
    // No-throw operations where possible
    [[nodiscard]] bool has_algorithm(const std::string& name) const noexcept {
        return algorithms_ && algorithms_->contains(name);
    }
};

// ❌ DON'T: Weak exception guarantees
class TDAEngine {
public:
    void register_algorithm(std::string name, std::unique_ptr<Algorithm> algorithm) {
        // Bad: could leave object in inconsistent state
        algorithms_->add(name, std::move(algorithm));
    }
};
```

## TDA-Specific Algorithm Patterns

### 1. Point Cloud Processing
```cpp
// ✅ DO: Efficient point cloud handling with SIMD
#include <immintrin.h>
#include <span>

class PointCloudProcessor {
private:
    std::vector<double> data_;
    std::size_t dimension_;
    
public:
    // Use spans for efficient data access
    void process_points(std::span<const double> points) {
        if (points.size() % dimension_ != 0) {
            throw std::invalid_argument("Point count must be multiple of dimension");
        }
        
        const std::size_t num_points = points.size() / dimension_;
        
        // SIMD optimization for distance calculations
        #ifdef __AVX2__
        process_points_avx2(points, num_points);
        #else
        process_points_scalar(points, num_points);
        #endif
    }
    
private:
    void process_points_avx2(std::span<const double> points, std::size_t num_points) {
        const std::size_t vector_size = 4; // AVX2 processes 4 doubles at once
        const std::size_t vectorized_points = (num_points / vector_size) * vector_size;
        
        for (std::size_t i = 0; i < vectorized_points; i += vector_size) {
            __m256d point1 = _mm256_loadu_pd(&points[i * dimension_]);
            __m256d point2 = _mm256_loadu_pd(&points[(i + 1) * dimension_]);
            
            // Vectorized distance calculation
            __m256d diff = _mm256_sub_pd(point1, point2);
            __m256d squared = _mm256_mul_pd(diff, diff);
            
            // Horizontal sum
            double distance = _mm256_reduce_add_pd(squared);
            distance = std::sqrt(distance);
            
            // Process distance...
        }
        
        // Handle remaining points with scalar code
        for (std::size_t i = vectorized_points; i < num_points; ++i) {
            process_single_point(&points[i * dimension_]);
        }
    }
};

// ❌ DON'T: Inefficient point processing
class PointCloudProcessor {
public:
    void process_points(const std::vector<double>& points) {
        // Bad: no SIMD optimization, inefficient access patterns
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = i + 1; j < points.size(); ++j) {
                double distance = calculate_distance(points[i], points[j]);
                // Process distance...
            }
        }
    }
};
```

### 2. Filtration Construction
```cpp
// ✅ DO: Efficient filtration algorithms with proper data structures
#include <queue>
#include <unordered_set>

class VietorisRipsFiltration {
private:
    struct Edge {
        std::size_t vertex1, vertex2;
        double birth_time;
        
        bool operator>(const Edge& other) const {
            return birth_time > other.birth_time;
        }
    };
    
    std::priority_queue<Edge, std::vector<Edge>, std::greater<>> edge_queue_;
    std::vector<std::unordered_set<std::size_t>> adjacency_lists_;
    
public:
    void build_filtration(const PointCloud& points, double max_radius) {
        const std::size_t num_points = points.size();
        adjacency_lists_.resize(num_points);
        
        // Build edge queue efficiently
        for (std::size_t i = 0; i < num_points; ++i) {
            for (std::size_t j = i + 1; j < num_points; ++j) {
                double distance = calculate_distance(points, i, j);
                if (distance <= max_radius) {
                    edge_queue_.push({i, j, distance});
                }
            }
        }
        
        // Process edges in birth time order
        while (!edge_queue_.empty()) {
            Edge edge = edge_queue_.top();
            edge_queue_.pop();
            
            add_edge_to_complex(edge);
            update_persistence_pairs(edge);
        }
    }
    
private:
    void add_edge_to_complex(const Edge& edge) {
        adjacency_lists_[edge.vertex1].insert(edge.vertex2);
        adjacency_lists_[edge.vertex2].insert(edge.vertex1);
        
        // Check for triangle formation
        check_triangles(edge.vertex1, edge.vertex2);
    }
    
    void check_triangles(std::size_t v1, std::size_t v2) {
        // Find common neighbors efficiently
        std::vector<std::size_t> common_neighbors;
        std::set_intersection(
            adjacency_lists_[v1].begin(), adjacency_lists_[v1].end(),
            adjacency_lists_[v2].begin(), adjacency_lists_[v2].end(),
            std::back_inserter(common_neighbors)
        );
        
        for (std::size_t v3 : common_neighbors) {
            add_triangle(v1, v2, v3);
        }
    }
};

// ❌ DON'T: Inefficient filtration construction
class VietorisRipsFiltration {
public:
    void build_filtration(const PointCloud& points, double max_radius) {
        // Bad: O(n³) complexity, no optimization
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < points.size(); ++j) {
                for (size_t k = 0; k < points.size(); ++k) {
                    // Check all possible triangles
                    if (i != j && j != k && i != k) {
                        // Process triangle...
                    }
                }
            }
        }
    }
};
```

### 3. Persistence Computation
```cpp
// ✅ DO: Efficient persistence algorithm implementation
#include <vector>
#include <algorithm>

class PersistenceCalculator {
private:
    struct Simplex {
        std::vector<std::size_t> vertices;
        double birth_time;
        std::size_t dimension;
        std::size_t index;
    };
    
    struct PersistencePair {
        std::size_t birth_simplex;
        std::size_t death_simplex;
        double birth_time;
        double death_time;
        std::size_t dimension;
    };
    
public:
    std::vector<PersistencePair> compute_persistence(
        const std::vector<Simplex>& simplices
    ) {
        std::vector<PersistencePair> pairs;
        std::vector<bool> processed(simplices.size(), false);
        
        // Sort simplices by birth time
        std::vector<std::size_t> sorted_indices(simplices.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](std::size_t a, std::size_t b) {
                return simplices[a].birth_time < simplices[b].birth_time;
            });
        
        // Process simplices in order
        for (std::size_t idx : sorted_indices) {
            if (processed[idx]) continue;
            
            const Simplex& simplex = simplices[idx];
            
            if (simplex.dimension == 0) {
                // 0-simplex always creates a new component
                continue;
            }
            
            // Find boundary and check for cycle
            auto boundary = compute_boundary(simplex);
            auto cycle = find_cycle(boundary, processed);
            
            if (cycle.empty()) {
                // New cycle created
                continue;
            } else {
                // Cycle dies - create persistence pair
                std::size_t death_simplex = find_death_simplex(cycle, simplex);
                if (death_simplex != simplex.index) {
                    pairs.push_back({
                        .birth_simplex = death_simplex,
                        .death_simplex = simplex.index,
                        .birth_time = simplices[death_simplex].birth_time,
                        .death_time = simplex.birth_time,
                        .dimension = simplex.dimension - 1
                    });
                }
            }
            
            processed[idx] = true;
        }
        
        return pairs;
    }
    
private:
    std::vector<std::size_t> compute_boundary(const Simplex& simplex) {
        std::vector<std::size_t> boundary;
        const std::size_t dim = simplex.vertices.size();
        
        for (std::size_t i = 0; i < dim; ++i) {
            std::vector<std::size_t> face_vertices;
            for (std::size_t j = 0; j < dim; ++j) {
                if (j != i) {
                    face_vertices.push_back(simplex.vertices[j]);
                }
            }
            // Find face index and add to boundary
            std::size_t face_index = find_face_index(face_vertices);
            boundary.push_back(face_index);
        }
        
        return boundary;
    }
};
```

## Performance Optimization

### 1. SIMD Vectorization
```cpp
// ✅ DO: Use SIMD for mathematical operations
#include <immintrin.h>

class VectorizedMath {
public:
    // Vectorized distance calculation
    static double distance_squared_avx2(
        const double* point1,
        const double* point2,
        std::size_t dimension
    ) {
        __m256d sum = _mm256_setzero_pd();
        const std::size_t vector_size = 4;
        const std::size_t vectorized_dim = (dimension / vector_size) * vector_size;
        
        for (std::size_t i = 0; i < vectorized_dim; i += vector_size) {
            __m256d diff = _mm256_sub_pd(
                _mm256_loadu_pd(&point1[i]),
                _mm256_loadu_pd(&point2[i])
            );
            sum = _mm256_fmadd_pd(diff, diff, sum);
        }
        
        // Horizontal sum
        double result = _mm256_reduce_add_pd(sum);
        
        // Handle remaining dimensions
        for (std::size_t i = vectorized_dim; i < dimension; ++i) {
            double diff = point1[i] - point2[i];
            result += diff * diff;
        }
        
        return result;
    }
    
    // Vectorized matrix operations
    static void matrix_multiply_avx2(
        const double* A,
        const double* B,
        double* C,
        std::size_t rows,
        std::size_t cols,
        std::size_t inner_dim
    ) {
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; j += 4) {
                __m256d sum = _mm256_setzero_pd();
                
                for (std::size_t k = 0; k < inner_dim; ++k) {
                    __m256d a = _mm256_set1_pd(A[i * inner_dim + k]);
                    __m256d b = _mm256_loadu_pd(&B[k * cols + j]);
                    sum = _mm256_fmadd_pd(a, b, sum);
                }
                
                _mm256_storeu_pd(&C[i * cols + j], sum);
            }
        }
    }
};
```

### 2. Memory Layout Optimization
```cpp
// ✅ DO: Optimize memory layout for cache efficiency
class OptimizedPointCloud {
private:
    // Structure of Arrays (SoA) for better cache locality
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;
    std::vector<double> z_coords_;
    std::size_t size_;
    
public:
    explicit OptimizedPointCloud(std::size_t size) 
        : size_(size) {
        x_coords_.reserve(size);
        y_coords_.reserve(size);
        z_coords_.reserve(size);
    }
    
    void add_point(double x, double y, double z) {
        x_coords_.push_back(x);
        y_coords_.push_back(y);
        z_coords_.push_back(z);
    }
    
    // Efficient access patterns
    double get_x(std::size_t index) const { return x_coords_[index]; }
    double get_y(std::size_t index) const { return y_coords_[index]; }
    double get_z(std::size_t index) const { return z_coords_[index]; }
    
    // Vectorized operations on coordinates
    void scale_x_coordinates(double factor) {
        const std::size_t vector_size = 4;
        const std::size_t vectorized_size = (size_ / vector_size) * vector_size;
        
        #ifdef __AVX2__
        __m256d scale_factor = _mm256_set1_pd(factor);
        
        for (std::size_t i = 0; i < vectorized_size; i += vector_size) {
            __m256d coords = _mm256_loadu_pd(&x_coords_[i]);
            coords = _mm256_mul_pd(coords, scale_factor);
            _mm256_storeu_pd(&x_coords_[i], coords);
        }
        #endif
        
        // Handle remaining elements
        for (std::size_t i = vectorized_size; i < size_; ++i) {
            x_coords_[i] *= factor;
        }
    }
};

// ❌ DON'T: Array of Structures (AoS) for large datasets
struct Point {  // Bad: poor cache locality
    double x, y, z;
};

class PointCloud {
private:
    std::vector<Point> points_;  // Bad: scattered memory access
};
```

## Python Integration

### 1. pybind11 Bindings
```cpp
// ✅ DO: Proper pybind11 integration with error handling
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(tda_core, m) {
    m.doc() = "TDA Core Algorithms Module";
    
    // Expose point cloud class
    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<std::size_t>())
        .def("add_point", &PointCloud::add_point)
        .def("size", &PointCloud::size)
        .def("dimension", &PointCloud::dimension)
        .def("get_point", &PointCloud::get_point)
        .def("scale", &PointCloud::scale);
    
    // Expose filtration algorithms
    py::class_<VietorisRipsFiltration>(m, "VietorisRipsFiltration")
        .def(py::init<const PointCloud&>())
        .def("build_filtration", &VietorisRipsFiltration::build_filtration)
        .def("get_simplices", &VietorisRipsFiltration::get_simplices)
        .def("get_persistence_pairs", &VietorisRipsFiltration::get_persistence_pairs);
    
    // Expose persistence calculator
    py::class_<PersistenceCalculator>(m, "PersistenceCalculator")
        .def(py::init<>())
        .def("compute_persistence", &PersistenceCalculator::compute_persistence);
    
    // Utility functions
    m.def("compute_vietoris_rips", [](const py::array_t<double>& points) {
        // Validate input
        if (points.ndim() != 2) {
            throw py::value_error("Points must be a 2D array");
        }
        
        const std::size_t num_points = points.shape(0);
        const std::size_t dimension = points.shape(1);
        
        if (dimension < 2 || dimension > 3) {
            throw py::value_error("Dimension must be 2 or 3");
        }
        
        // Create point cloud
        PointCloud cloud(num_points);
        auto points_data = points.unchecked<2>();
        
        for (std::size_t i = 0; i < num_points; ++i) {
            if (dimension == 2) {
                cloud.add_point(points_data(i, 0), points_data(i, 1), 0.0);
            } else {
                cloud.add_point(points_data(i, 0), points_data(i, 1), points_data(i, 2));
            }
        }
        
        // Compute filtration and persistence
        VietorisRipsFiltration filtration(cloud);
        filtration.build_filtration(cloud, std::numeric_limits<double>::infinity());
        
        auto pairs = filtration.get_persistence_pairs();
        
        // Convert to Python-friendly format
        py::list result;
        for (const auto& pair : pairs) {
            py::dict pair_dict;
            pair_dict["birth"] = pair.birth_time;
            pair_dict["death"] = pair.death_time;
            pair_dict["dimension"] = pair.dimension;
            result.append(pair_dict);
        }
        
        return result;
    });
}

// ❌ DON'T: Basic pybind11 without error handling
PYBIND11_MODULE(tda_core, m) {
    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<std::size_t>())
        .def("add_point", &PointCloud::add_point);  // Bad: no validation
}
```

### 2. Error Handling
```cpp
// ✅ DO: Proper C++ to Python exception translation
class TDAException : public std::runtime_error {
public:
    explicit TDAException(const std::string& message) 
        : std::runtime_error(message) {}
};

// Custom exception translator
void translate_tda_exception(const TDAException& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

// Register exception translator
PYBIND11_MODULE(tda_core, m) {
    // Register custom exception
    py::register_exception<TDAException>(m, "TDAException");
    
    // Register translator
    py::register_exception_translator<TDAException>(translate_tda_exception);
    
    // Functions that may throw
    m.def("compute_persistence", [](const PointCloud& cloud) {
        try {
            PersistenceCalculator calc;
            return calc.compute_persistence(cloud);
        } catch (const std::exception& e) {
            throw TDAException(std::string("Persistence computation failed: ") + e.what());
        }
    });
}
```

## Testing & Validation

### 1. Unit Testing
```cpp
// ✅ DO: Comprehensive unit testing with proper fixtures
#include <gtest/gtest.h>
#include <gmock/gmock.h>

class VietorisRipsFiltrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test point cloud
        cloud_ = std::make_unique<PointCloud>(4);
        cloud_->add_point(0.0, 0.0, 0.0);
        cloud_->add_point(1.0, 0.0, 0.0);
        cloud_->add_point(0.0, 1.0, 0.0);
        cloud_->add_point(1.0, 1.0, 0.0);
        
        filtration_ = std::make_unique<VietorisRipsFiltration>(*cloud_);
    }
    
    std::unique_ptr<PointCloud> cloud_;
    std::unique_ptr<VietorisRipsFiltration> filtration_;
};

TEST_F(VietorisRipsFiltrationTest, BuildsCorrectFiltration) {
    filtration_->build_filtration(*cloud_, 2.0);
    
    auto simplices = filtration_->get_simplices();
    EXPECT_EQ(simplices.size(), 10); // 4 vertices + 6 edges
    
    // Check that all edges are present
    std::set<std::pair<std::size_t, std::size_t>> expected_edges = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };
    
    for (const auto& simplex : simplices) {
        if (simplex.dimension == 1) {
            std::pair<std::size_t, std::size_t> edge = {
                simplex.vertices[0], simplex.vertices[1]
            };
            EXPECT_TRUE(expected_edges.count(edge) > 0);
        }
    }
}

TEST_F(VietorisRipsFiltrationTest, ComputesCorrectPersistence) {
    filtration_->build_filtration(*cloud_, 2.0);
    
    auto pairs = filtration_->get_persistence_pairs();
    
    // Should have 1 persistent feature in dimension 1
    auto dim1_pairs = std::count_if(pairs.begin(), pairs.end(),
        [](const PersistencePair& pair) { return pair.dimension == 1; });
    
    EXPECT_EQ(dim1_pairs, 1);
}
```

### 2. Performance Testing
```cpp
// ✅ DO: Performance benchmarks for critical paths
#include <benchmark/benchmark.h>

static void BM_VietorisRipsConstruction(benchmark::State& state) {
    // Create large point cloud
    const std::size_t num_points = state.range(0);
    PointCloud cloud(num_points);
    
    // Generate random points
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    for (std::size_t i = 0; i < num_points; ++i) {
        cloud.add_point(dis(gen), dis(gen), dis(gen));
    }
    
    for (auto _ : state) {
        VietorisRipsFiltration filtration(cloud);
        filtration.build_filtration(cloud, 10.0);
        benchmark::DoNotOptimize(filtration);
    }
    
    state.SetComplexityN(num_points);
}

BENCHMARK(BM_VietorisRipsConstruction)
    ->RangeMultiplier(2)
    ->Range(1000, 100000)
    ->Complexity(benchmark::oNSquared);
```

## Common C++ Issues

### 1. Memory Management
- **Memory Leaks**: Use smart pointers, RAII patterns
- **Dangling References**: Ensure proper lifetime management
- **Buffer Overflows**: Use bounds checking, std::span for safety

### 2. Performance Issues
- **Cache Misses**: Optimize memory layout, use SoA patterns
- **Unnecessary Copies**: Use move semantics, const references
- **Inefficient Algorithms**: Profile and optimize critical paths

### 3. Exception Safety
- **Weak Guarantees**: Provide strong exception guarantees where possible
- **Resource Leaks**: Use RAII to prevent leaks during exceptions
- **Exception Propagation**: Properly translate C++ exceptions to Python

### 4. Thread Safety
- **Race Conditions**: Use proper synchronization for shared data
- **Data Races**: Ensure atomic operations or proper locking
- **Deadlocks**: Avoid nested locks, use lock ordering

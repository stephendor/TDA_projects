# TDA Platform Performance Guide

This guide provides comprehensive performance optimization strategies, benchmarking techniques, and best practices for the TDA Platform. Learn how to achieve the target performance of processing 1M+ points in under 60 seconds.

## 📚 Performance Documentation Structure

### 🎯 [Performance Overview](./overview.md)
- **Performance Targets** - Current benchmarks and goals ✅
- **Bottleneck Analysis** - Identifying performance constraints ✅
- **Scaling Strategies** - Horizontal and vertical scaling approaches ✅

### ⚡ [Optimization Techniques](./optimization/)
- **SIMD Vectorization** - CPU instruction optimization ✅
- **Memory Management** - Efficient data structures and allocation ✅
- **Parallelization** - OpenMP, threading, and distributed computing ✅
- **Algorithm Optimization** - Mathematical and computational improvements ✅

### 📊 [Benchmarking & Profiling](./benchmarking/)
- **Performance Testing** - Comprehensive benchmark suites ✅
- **Profiling Tools** - gprof, Valgrind, custom instrumentation ✅
- **Metrics & KPIs** - Key performance indicators ✅
- **Regression Testing** - Performance regression prevention 🔄

### 🚀 [High-Performance Computing](./hpc/)
- **GPU Acceleration** - CUDA implementation strategies 🔄
- **Distributed Computing** - Spark, Flink, and MPI 🔄
- **Cloud Optimization** - AWS, GCP, and Azure strategies 🔄
- **Container Performance** - Docker and Kubernetes optimization 🔄

## 🎯 Performance Targets

### Current Benchmarks (Task 1 Completed) ✅
- **Small Datasets** (<10K points): <1 second ✅
- **Medium Datasets** (100K points): <10 seconds ✅
- **Large Datasets** (1M points): <60 seconds ✅ ⭐
- **Very Large** (10M+ points): <600 seconds 🔄 (in validation)

### Performance KPIs
- **Throughput**: Points processed per second ✅
- **Latency**: Time to first result ✅
- **Memory Efficiency**: RAM usage per million points ✅
- **Scalability**: Performance improvement with cores/GPUs ✅

## ⚡ Core Optimization Strategies

### 1. **SIMD Vectorization** ✅
```cpp
// Before: Scalar distance computation
double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// After: SIMD-optimized distance computation
#include <immintrin.h>

double distance_simd(const Point& p1, const Point& p2) {
    __m256d v1 = _mm256_load_pd(&p1.x);
    __m256d v2 = _mm256_load_pd(&p2.x);
    __m256d diff = _mm256_sub_pd(v1, v2);
    __m256d squared = _mm256_mul_pd(diff, diff);
    
    // Horizontal sum and sqrt
    double result[4];
    _mm256_store_pd(result, squared);
    return std::sqrt(result[0] + result[1] + result[2]);
}
```

## Experimental Feature Notes

See softcap_local_merge.md for the experimental soft kNN cap local-merge mode in the streaming distance matrix. It is OFF by default; enable it only for controlled experiments and monitor overshoot telemetry and adjacency histograms. Use baseline comparisons to validate accuracy impact.

### Baseline comparisons without memory overlap

For QA and accuracy checks, compare a soft-cap run against a race-free baseline (no soft cap, serial threshold). To avoid cumulative peak RSS when running both paths, use the separate-process baseline mode:

- --baseline-compare 1: enable baseline compare in Čech mode
- --baseline-separate-process 1: run the baseline in a fresh process to avoid allocator page retention
- --baseline-maxDim N: optionally override maxDim for the baseline run

Example:

```bash
build/release/tests/cpp/test_streaming_cech_perf \
    --n 8000 --d 3 --radius 0.9 --maxDim 2 --maxNeighbors 32 --block 128 \
    --soft-knn-cap 16 --parallel-threshold 0 \
    --baseline-compare 1 --baseline-separate-process 1 \
    --adj-hist-csv /tmp/adj_cur.csv --adj-hist-csv-baseline /tmp/adj_base.csv
```

Notes:

- Keep --parallel-threshold 0 when using a soft cap for correctness/telemetry runs.
- Inspect overshoot telemetry (overshoot_sum/overshoot_max) in CSV/JSON outputs.

### 2. **Memory Optimization** ✅

```cpp
// Memory pool for simplex allocation
class SimplexPool {
private:
    std::vector<std::vector<Simplex>> pools_;
    std::vector<size_t> free_indices_;
    
public:
    Simplex* allocate(size_t dimension) {
        if (free_indices_[dimension] < pools_[dimension].size()) {
            return &pools_[dimension][free_indices_[dimension]++];
        }
        // Expand pool if needed
        expand_pool(dimension);
        return allocate(dimension);
    }
    
    void reset() {
        std::fill(free_indices_.begin(), free_indices_.end(), 0);
    }
};
```

### 3. **Parallelization Strategies** ✅

```cpp
// OpenMP parallelization for distance matrix computation
void compute_distance_matrix_parallel(
    const std::vector<Point>& points,
    std::vector<std::vector<double>>& distances
) {
    const size_t n = points.size();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = distance(points[i], points[j]);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
}
```

## 📊 Benchmarking Framework

### Performance Test Suite ✅

```cpp
// Comprehensive performance testing
class PerformanceTestSuite {
public:
    void run_scalability_tests() {
        std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
        
        for (size_t size : sizes) {
            auto points = generate_test_data(size);
            
            // Test different algorithms
            benchmark_vietoris_rips(points);
            benchmark_alpha_complex(points);
            benchmark_cech_complex(points);
            benchmark_dtm_filtration(points);
        }
    }
    
    void benchmark_vietoris_rips(const std::vector<Point>& points) {
        auto start = std::chrono::high_resolution_clock::now();
        
        VietorisRipsFiltration filtration(points);
        auto results = filtration.compute_persistence(3);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "VR Filtration (" << points.size() << " points): "
                  << duration.count() << "ms\n";
    }
};
```

### Profiling Tools Integration ✅

```cpp
// Custom performance monitoring
class PerformanceMonitor {
private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> timers_;
    std::map<std::string, std::vector<double>> measurements_;
    
public:
    void start_timer(const std::string& name) {
        timers_[name] = std::chrono::high_resolution_clock::now();
    }
    
    void stop_timer(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto start = timers_[name];
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        measurements_[name].push_back(duration.count() / 1000.0); // Convert to ms
    }
    
    void print_summary() {
        for (const auto& [name, values] : measurements_) {
            double avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            double min = *std::min_element(values.begin(), values.end());
            double max = *std::max_element(values.begin(), values.end());
            
            std::cout << name << ": avg=" << avg << "ms, min=" << min 
                      << "ms, max=" << max << "ms\n";
        }
    }
};
```

## 🚀 High-Performance Computing

### GPU Acceleration with CUDA 🔄

```cpp
// CUDA kernel for distance computation
__global__ void compute_distances_kernel(
    const double* points,
    double* distances,
    const int n,
    const int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n && j < n && i < j) {
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            double diff = points[i * dim + d] - points[j * dim + d];
            sum += diff * diff;
        }
        distances[i * n + j] = sqrt(sum);
        distances[j * n + i] = sqrt(sum);
    }
}

// CUDA wrapper
class CUDADistanceMatrix {
public:
    void compute(const std::vector<Point>& points) {
        // Allocate device memory
        // Copy data to GPU
        // Launch kernel
        // Copy results back
    }
};
```

### Distributed Computing with Apache Spark 🔄

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

def compute_persistence_distributed(spark, points_rdd):
    """Distributed persistence computation using Spark"""
    
    # Convert to Spark DataFrame
    points_df = points_rdd.toDF(["features"])
    
    # Apply persistence computation to each partition
    def compute_partition_persistence(iterator):
        for row in iterator:
            # Run TDA computation on partition
            yield compute_local_persistence(row.features)
    
    results = points_df.rdd.mapPartitions(compute_partition_persistence)
    
    return results.collect()
```

## 🔧 Performance Tuning

### Build Optimization ✅

```bash
# Release build with maximum optimization
./build.sh release OFF false false false

# Custom optimization flags
cmake ../.. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math" \
  -DENABLE_SIMD=ON \
  -DENABLE_OPENMP=ON \
  -DENABLE_CUDA=ON
```

### Runtime Configuration ✅

```cpp
// Performance configuration
struct PerformanceConfig {
    size_t max_threads = std::thread::hardware_concurrency();
    size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB
    bool enable_simd = true;
    bool enable_gpu = false;
    size_t batch_size = 10000;
    
    // Algorithm-specific parameters
    double rips_epsilon = 1.0;
    size_t alpha_complex_precision = 8;
    size_t cech_approximation_factor = 4;
};
```

## 📈 Performance Monitoring

### Real-time Metrics ✅

```cpp
// Performance metrics collection
class MetricsCollector {
public:
    void record_operation(const std::string& name, double duration_ms) {
        metrics_[name].count++;
        metrics_[name].total_time += duration_ms;
        metrics_[name].min_time = std::min(metrics_[name].min_time, duration_ms);
        metrics_[name].max_time = std::max(metrics_[name].max_time, duration_ms);
    }
    
    void export_metrics(const std::string& filename) {
        // Export to Prometheus, InfluxDB, or custom format
    }
};
```

### Performance Dashboards 🔄

- **Grafana Dashboards** - Real-time performance monitoring (planned)
- **Prometheus Metrics** - Time-series performance data (planned)
- **Custom Visualizations** - Algorithm-specific performance charts (planned)

## 🎯 Best Practices

### 1. **Profile First, Optimize Second** ✅

- Use profiling tools to identify bottlenecks ✅
- Focus optimization efforts on the slowest 20% of code ✅
- Measure before and after each optimization ✅

### 2. **Memory Management** ✅

- Minimize allocations in hot paths ✅
- Use memory pools for frequently allocated objects ✅
- Prefer stack allocation over heap allocation ✅

### 3. **Algorithm Selection** ✅

- Choose the right algorithm for your data characteristics ✅
- Consider approximation algorithms for large datasets ✅
- Balance accuracy vs. performance requirements ✅

### 4. **Parallelization Strategy** ✅

- Start with OpenMP for shared-memory parallelism ✅
- Use std::thread for more control ✅
- Consider distributed computing for very large datasets 🔄

## 🔗 Related Documentation

- **[API Reference](../api/)** - Performance-focused API usage
- **[Integration Guides](../integration/)** - Performance optimization workflows
- **[Troubleshooting](../troubleshooting/)** - Performance problem diagnosis
- **[Examples](../examples/)** - Performance optimization examples

## 📞 Performance Support

- **Performance Issues**: Include benchmark results and system specs
- **Optimization Questions**: Describe current performance and goals
- **Benchmark Requests**: Specify dataset characteristics and requirements

## 🚧 Implementation Status

### ✅ **Completed Optimizations**

- **SIMD Vectorization**: AVX2/AVX-512 support for distance computations
- **Memory Management**: Memory pools, efficient data structures
- **Parallelization**: OpenMP integration for complex operations
- **Algorithm Optimization**: Approximation algorithms for Čech complex
- **Spatial Indexing**: KD-trees and Ball-trees for efficient search
- **Performance Profiling**: Comprehensive benchmarking framework

### 🔄 **In Development**

- **GPU Acceleration**: CUDA implementation for distance matrix computation
- **Distributed Computing**: Spark/Flink integration for very large datasets
- **Performance Dashboards**: Real-time monitoring and visualization

### 🔄 **Planned Features**

- **Cloud Optimization**: AWS, GCP, Azure deployment strategies
- **Container Performance**: Docker and Kubernetes optimization
- **Advanced Profiling**: Custom performance analysis tools

---

*Ready to optimize? Start with [Performance Overview](./overview.md) to understand current benchmarks, then dive into [Optimization Techniques](./optimization/) for specific strategies.*

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

```bash
./scripts/run_performance_validation.sh \
    --harness build/release/tests/cpp/test_streaming_cech_perf \
    --n 8000 --d 8 --radius 0.5 --maxDim 1 --K 16
```

Staircase runs (comma-separated sizes):

```bash
./scripts/run_performance_validation.sh \
    --harness build/release/tests/cpp/test_streaming_cech_perf \
    --staircase 200000,500000,1000000 --d 8 --radius 0.5 --maxDim 1 --K 16
```
 
For QA and accuracy checks, compare a soft-cap run against a race-free baseline (no soft cap, serial threshold). To avoid cumulative peak RSS when running both paths, use the separate-process baseline mode:

- --baseline-compare 1: enable baseline compare in Čech mode (separate subprocess recommended)

## Accuracy checks (small-n exact vs soft-cap)

Use `scripts/run_accuracy_check.sh` to generate an accuracy report comparing exact small-n runs against soft-cap variants (K in a list). It computes deltas in edge counts and H1 intervals and emits a compact JSON report.

Example:

```bash
./scripts/run_accuracy_check.sh \
    --harness build/release/tests/cpp/test_streaming_cech_perf \
    --n 4000 --d 8 --radius 0.5 --maxDim 1 --klist 8,16,32
```

Artifacts:

- `accuracy_baseline_N.jsonl` and child `accuracy_baseline_child_N.jsonl`
- `accuracy_KK_N.jsonl` for each K
- `accuracy_report_N.json` with per-K deltas (% and absolute)
- --baseline-separate-process 1: run the baseline in a fresh process to avoid allocator page retention
- --baseline-maxDim N: optionally override maxDim for the baseline run
- --baseline-json-out \<path\>: direct the child baseline process to write its JSONL telemetry to a deterministic file you control (useful for CI parsing)

Example:

```bash
build/release/tests/cpp/test_streaming_cech_perf \
    --n 8000 --d 3 --radius 0.9 --maxDim 2 --maxNeighbors 32 --block 128 \
    --soft-knn-cap 16 --parallel-threshold 0 \
    --baseline-compare 1 --baseline-separate-process 1 \
    --baseline-json-out /tmp/cech_baseline.jsonl \
    --adj-hist-csv /tmp/adj_cur.csv --adj-hist-csv-baseline /tmp/adj_base.csv
```

Notes:

- Keep --parallel-threshold 0 when using a soft cap for correctness/telemetry runs.
- Inspect overshoot telemetry (softcap_overshoot_sum / softcap_overshoot_max) in JSON outputs, and overshoot_sum/overshoot_max in CSV.
- The separate baseline process will print its own summary line before the parent prints the delta summary and the parent’s final line; this is expected ordering.

## VR mode usage (Streaming Vietoris–Rips)

The performance harness supports VR mode with telemetry parity to Čech. Use `--mode vr` and provide `--epsilon` (distance threshold). The filtration value for a simplex is the max pairwise edge distance.

Examples:

```bash
# Small VR probe with soft-cap and serial threshold (telemetry-friendly)
build/release/tests/cpp/test_streaming_cech_perf \
    --mode vr --n 6000 --d 6 --epsilon 0.5 --maxDim 1 \
    --soft-knn-cap 16 --parallel-threshold 0 \
    --adj-hist-csv /tmp/adj_vr_parent.csv \
    --json /tmp/run_vr.jsonl

# Baseline compare pattern (recommended for Čech; for VR use single-run comparisons)
scripts/run_performance_validation.sh \
    --harness build/release/tests/cpp/test_streaming_cech_perf \
    --artifacts /tmp/tda-artifacts \
    --n 8000 --d 8 --maxDim 1 --K 16 --radius 0.5
```

Analyzer expectations in VR mode:

- When `parallel_threshold==0` and soft cap is used, `softcap_overshoot_sum==0` and `softcap_overshoot_max==0` are required for a PASS gate.
- Memory fields `dm_peakMB` and `rss_peakMB` must be present.
- CSV adjacency histograms are optional unless `--require-csv` is set for the analyzer.

Compact summary lines printed in CI (example):

```text
[summary] run_vr.jsonl: mode=vr, dm_blocks=38862, dm_edges=346567, dm_peakMB=106.55, rss_peakMB=120.31, softcap_overshoot_sum=0, softcap_overshoot_max=0, parallel_threshold=0
```

### Telemetry fields written by the harness (JSONL)

- dm_blocks, dm_edges: total processed blocks and emitted edges
- dm_peakMB: peak memory during distance streaming (MiB)
- rss_peakMB: peak memory during build (MiB)
- simplices: total simplices built (when mode=Cech)
- parallel_threshold: 0 or 1 (used by analyzer to enforce race-free runs under soft cap)
- softcap_overshoot_sum, softcap_overshoot_max: overshoot telemetry for soft kNN cap

Backward-compatible underscore keys are also present: dm_peak_mb, build_peak_mb, overshoot_sum, overshoot_max.

## Optional GPU path (CUDA)

The streaming distance stage supports an optional, flag-gated CUDA path for block computations.

- Build with CUDA enabled:

```bash
./build.sh release ON false  # or: cmake -DENABLE_CUDA=ON ...
```

- Turn on GPU path at runtime (two equivalent ways):

```bash
# CLI flag (sets env internally)
build/release/tests/cpp/test_streaming_cech_perf --dm-only --n 100000 --d 3 --radius 0.5 --cuda 1 --json /tmp/run.jsonl

# or via environment variable
export TDA_USE_CUDA=1
build/release/tests/cpp/test_streaming_cech_perf --dm-only --n 100000 --d 3 --radius 0.5 --json /tmp/run.jsonl
```

Notes:

- CUDA is optional and off by default. If not compiled with CUDA (`ENABLE_CUDA=OFF`) or no device is available, the harness falls back to the CPU path.
- The `manifest_hash` includes `use_cuda` so CI can distinguish CPU/GPU runs deterministically.
- Analyzer gating is unchanged by GPU usage.

## Distributed hooks (Flink/Spark) — reference format

To integrate with stream processors, represent work and results via compact NDJSON:

- Block task schema (per tile):

```json
{ "i0": 0, "j0": 256, "block": 64, "threshold": 1.0, "mode": "threshold", "seed": 42 }
```

- Edge emission schema (threshold mode):

```json
{ "i": 123, "j": 987, "d": 0.4123 }
```

Reference mapping (pseudocode):

```python
# Flink Python pseudocode
from pyflink.datastream import StreamExecutionEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
tasks = env.from_source("block_tasks")  # (i0,j0,block,threshold,...)

def compute_tile(task):
    # Load points shard(s), compute distances for tile (i0..i0+B-1, j0..j0+B-1)
    # Emit edges with d <= threshold
    yield from edges

edges_ds = tasks.flat_map(compute_tile)
edges_ds.add_sink("edges_out")
env.execute("tda-edges")
```

This reference pipeline is documentation-only and not gated in CI.

### Accuracy reports and thresholds

Use `scripts/run_accuracy_check.sh` to emit `accuracy_report_*.json` comparing exact (serial, no soft cap) to soft-cap variants (e.g., K=8,16,32). The analyzer can parse these and optionally enforce percentage thresholds:

```bash
python3 scripts/analyze_performance_logs.py \
    --artifacts /tmp/tda-artifacts \
    --accuracy-edges-pct-threshold 5.0 \
    --accuracy-h1-pct-threshold 5.0
```

To gate on these thresholds (CI-fail), add `--accuracy-gate --gate`.

#### Analyzer gating via environment variables

You can also control gating via environment variables without changing CLI flags:

- Set `ACCURACY_GATE=1` to enable accuracy gating.
- Set `REQUIRE_CSV=1` to require adjacency histogram CSVs.
- Set `ACCURACY_EDGES_PCT_THRESHOLD` and `ACCURACY_H1_PCT_THRESHOLD` to numeric values (e.g., `5.0`).

Example:

```bash
export TDA_ARTIFACT_DIR=/tmp/tda-artifacts
export ACCURACY_GATE=1 REQUIRE_CSV=1
export ACCURACY_EDGES_PCT_THRESHOLD=5.0 ACCURACY_H1_PCT_THRESHOLD=5.0
python3 scripts/analyze_performance_logs.py --artifacts "$TDA_ARTIFACT_DIR" --gate
```

## 📦 Reference artifacts (2025-08-14, 1M Čech, soft-cap vs baseline)

Artifacts directory: `docs/performance/artifacts/2025-08-14_1m_cech/`

- `run_cech_1m_t600.jsonl` (parent, soft-knn-cap=16, parallel-threshold=0)
- `baseline_cech_1m_t600.jsonl` (separate process, no soft cap, serial threshold)
- `adj_parent_1m_t600.csv`, `adj_baseline_1m_t600.csv` (degree histograms)
- `analyzer_1m_final.log` (gating PASS)

Summary (parent): dm_blocks=371,011, dm_edges=3,115,973, simplices=1,602,678, dm_peakMB=755.746, rss_peakMB=789.586, overshoot_sum/max=0.

Summary (baseline): dm_blocks=371,930, dm_edges=3,299,630, simplices=1,604,812, dm_peakMB=752.184, rss_peakMB=791.402.

Deltas (parent − baseline): Δedges=−183,657, Δsimplices=−2,134. Gating: PASS (require-csv, memory fields present, overshoot=0 in serial threshold).

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

## 🧪 Phase 1 Validation Scripts

For CI gating and quick local probes without memory overlap, use the provided scripts:

- scripts/run_performance_validation.sh: orchestrates a small probe with the streaming Čech harness, exporting JSONL telemetry and adjacency histogram CSVs. It runs the baseline in a separate process and writes artifacts to TDA_ARTIFACT_DIR (defaults to /tmp/tda-artifacts).
- scripts/analyze_performance_logs.py: parses artifacts and enforces gates:
    - If parallel_threshold==0, then softcap_overshoot_sum/max must be 0
    - Memory fields dm_peakMB and rss_peakMB must be present
    - Optionally require adjacency histogram CSVs

Example:

```bash
./scripts/run_performance_validation.sh \
    --harness build/release/tests/cpp/test_streaming_cech_perf \
    --artifacts /tmp/tda-artifacts
```

Notes:

- Keep --parallel-threshold 0 when --soft-knn-cap > 0 for accuracy/telemetry runs.
- The baseline JSONL path is set deterministically via --baseline-json-out; both parent and baseline JSONLs are analyzed.
- CI uses these scripts to gate overshoot==0 and presence of memory fields, and uploads artifacts for inspection.

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

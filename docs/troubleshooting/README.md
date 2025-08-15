# TDA Platform Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the TDA Platform, from build problems to runtime errors and performance issues.

## 📚 Troubleshooting Documentation Structure

### 🚨 [Common Issues](./common-issues/)
- **Build Failures** - Compilation and linking problems
- **Runtime Errors** - Application crashes and exceptions
- **Performance Problems** - Slow execution and bottlenecks
- **Integration Issues** - Third-party tool problems

### 🐛 [Debugging Guide](./debugging/)
- **Debug Tools** - gdb, Valgrind, and custom debugging
- **Log Analysis** - Understanding log messages and errors
- **Performance Profiling** - Identifying bottlenecks
- **Memory Debugging** - Memory leaks and corruption

### ⚡ [Performance Problems](./performance/)
- **Slow Execution** - Identifying performance bottlenecks
- **Memory Issues** - High memory usage and leaks
- **Scalability Problems** - Performance degradation with scale
- **Optimization Tips** - Quick performance fixes

### 🔧 [Error Reference](./error-reference/)
- **Error Codes** - Complete error code documentation
- **Exception Types** - C++ and Python exception handling
- **Status Codes** - API response status codes
- **Log Levels** - Understanding log message severity

## 🚨 Quick Problem Diagnosis

### 1. **Build Issues**
```bash
# Check compiler version
gcc --version          # Should be 13+
clang --version       # Should be 16+

# Check CMake version
cmake --version        # Should be 3.20+

# Clean build directory
rm -rf build/
./build.sh release

# Check dependencies
ldconfig -p | grep eigen
```

### 2. **Runtime Issues**
```bash
# Check if TDA engine is working
./bin/test_vietoris_rips

# Check Python bindings
cd backend && python -c "from tda_backend import TDAEngine; print('OK')"

# Check API service
curl http://localhost:8000/health
```

### 3. **Performance Issues**
```bash
# Run performance benchmarks
./bin/tda_benchmarks

# Check system resources
htop
nvidia-smi  # If using GPU

# Profile specific operations
valgrind --tool=callgrind ./bin/test_performance_benchmarks
```

## 🔍 Common Issues and Solutions

### Build Failures

#### **Compiler Not Found**
```bash
# Error: No C++23 compiler found
# Solution: Install GCC 13+ or Clang 16+

# Ubuntu/Debian
sudo apt update
sudo apt install gcc-13 g++-13

# Set as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130

# macOS
brew install gcc@13

# Verify installation
gcc --version  # Should show 13.x.x
```

#### **CMake Configuration Errors**
```bash
# Error: CMake version too old
# Solution: Install CMake 3.20+

# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# From source (if needed)
wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0.tar.gz
tar -xzf cmake-3.28.0.tar.gz
cd cmake-3.28.0
./bootstrap && make && sudo make install
```

#### **Missing Dependencies**
```bash
# Error: Eigen3 not found
# Solution: Install Eigen3

# Ubuntu/Debian
sudo apt install libeigen3-dev

# macOS
brew install eigen

# Verify installation
pkg-config --modversion eigen3
```

### Runtime Errors

#### **Segmentation Faults**
```cpp
// Common cause: Uninitialized pointers or out-of-bounds access
// Solution: Add bounds checking and initialization

class SafePointCloud {
private:
    std::vector<Point> points_;
    
public:
    Point& operator[](size_t index) {
        if (index >= points_.size()) {
            throw std::out_of_range("Index out of bounds");
        }
        return points_[index];
    }
    
    const Point& operator[](size_t index) const {
        if (index >= points_.size()) {
            throw std::out_of_range("Index out of bounds");
        }
        return points_[index];
    }
};
```

#### **Memory Allocation Failures**
```cpp
// Common cause: Large dataset memory requirements
// Solution: Implement memory-efficient processing

class MemoryEfficientProcessor {
private:
    size_t max_memory_mb_;
    
public:
    void process_large_dataset(const std::vector<Point>& points) {
        // Check available memory
        size_t required_memory = estimate_memory_requirement(points.size());
        if (required_memory > max_memory_mb_ * 1024 * 1024) {
            // Process in chunks
            process_in_chunks(points);
        } else {
            // Process all at once
            process_all(points);
        }
    }
    
private:
    void process_in_chunks(const std::vector<Point>& points) {
        const size_t chunk_size = 10000; // Adjust based on memory
        
        for (size_t i = 0; i < points.size(); i += chunk_size) {
            size_t end = std::min(i + chunk_size, points.size());
            std::vector<Point> chunk(points.begin() + i, points.begin() + end);
            process_chunk(chunk);
        }
    }
};
```

#### **Python Binding Errors**
```python
# Error: Module not found
# Solution: Check Python path and installation

import sys
print(sys.path)  # Check Python path

# Reinstall Python bindings
cd backend
pip install -e .

# Check if C++ library is built
ls -la ../build/release/lib/  # Should contain TDA libraries
```

### Performance Problems

#### **Slow Execution**
```cpp
// Common cause: Inefficient algorithms or poor memory access patterns
// Solution: Profile and optimize

class PerformanceOptimizer {
public:
    void optimize_distance_computation(const std::vector<Point>& points) {
        // 1. Use SIMD instructions
        #ifdef __AVX2__
        compute_distances_simd(points);
        #else
        compute_distances_scalar(points);
        #endif
        
        // 2. Parallelize computation
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = i + 1; j < points.size(); ++j) {
                compute_distance_optimized(points[i], points[j]);
            }
        }
    }
    
private:
    void compute_distances_simd(const std::vector<Point>& points) {
        // AVX2-optimized distance computation
        // Implementation details...
    }
};
```

#### **High Memory Usage**
```cpp
// Common cause: Memory leaks or inefficient data structures
// Solution: Use memory pools and smart pointers

class MemoryPool {
private:
    std::vector<std::vector<Simplex>> pools_;
    std::vector<size_t> free_indices_;
    
public:
    Simplex* allocate(size_t dimension) {
        if (free_indices_[dimension] < pools_[dimension].size()) {
            return &pools_[dimension][free_indices_[dimension]++];
        }
        expand_pool(dimension);
        return allocate(dimension);
    }
    
    void reset() {
        std::fill(free_indices_.begin(), free_indices_.end(), 0);
    }
    
private:
    void expand_pool(size_t dimension) {
        // Add more memory to the pool
        pools_[dimension].resize(pools_[dimension].size() * 2);
    }
};
```

## 🐛 Debugging Techniques

### Using GDB for C++ Debugging
```bash
# Compile with debug symbols
./build.sh debug

# Run with GDB
gdb ./bin/test_vietoris_rips

# GDB commands
(gdb) break main
(gdb) run
(gdb) next
(gdb) print points.size()
(gdb) backtrace  # If crashed
(gdb) info locals
(gdb) continue
```

### Using Valgrind for Memory Debugging
```bash
# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./bin/test_vietoris_rips

# Check for memory errors
valgrind --tool=memcheck --track-origins=yes ./bin/test_vietoris_rips

# Profile memory usage
valgrind --tool=massif ./bin/test_performance_benchmarks
ms_print massif.out.* > memory_profile.txt
```

### Custom Debugging with Performance Monitor
```cpp
// Custom performance monitoring
class DebugMonitor {
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
        
        measurements_[name].push_back(duration.count() / 1000.0);
    }
    
    void print_debug_info() {
        std::cout << "=== Debug Information ===\n";
        for (const auto& [name, values] : measurements_) {
            double avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            std::cout << name << ": " << avg << "ms (avg)\n";
        }
    }
};
```

## 📊 Performance Problem Diagnosis

### Identifying Bottlenecks
```cpp
// Performance profiling framework
class PerformanceProfiler {
public:
    void profile_operation(const std::string& name, std::function<void()> operation) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run operation
        operation();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << name << " took " << duration.count() / 1000.0 << "ms\n";
    }
    
    void profile_memory_usage(const std::string& operation_name) {
        // Get memory usage before
        size_t memory_before = get_current_memory_usage();
        
        // Run operation
        // ... operation code ...
        
        // Get memory usage after
        size_t memory_after = get_current_memory_usage();
        
        std::cout << operation_name << " memory delta: " 
                  << (memory_after - memory_before) / (1024 * 1024) << "MB\n";
    }
    
private:
    size_t get_current_memory_usage() {
        // Platform-specific memory usage implementation
        // Linux: /proc/self/status
        // macOS: mach_task_basic_info
        // Windows: GetProcessMemoryInfo
        return 0; // Placeholder
    }
};
```

### Memory Leak Detection
```cpp
// Memory leak detector
class MemoryLeakDetector {
private:
    std::map<void*, std::string> allocations_;
    std::mutex mutex_;
    
public:
    void* track_allocation(size_t size, const std::string& location) {
        void* ptr = malloc(size);
        
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = location;
        
        return ptr;
    }
    
    void track_deallocation(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_.erase(ptr);
        free(ptr);
    }
    
    void report_leaks() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!allocations_.empty()) {
            std::cout << "=== Memory Leaks Detected ===\n";
            for (const auto& [ptr, location] : allocations_) {
                std::cout << "Leak at " << ptr << " from " << location << "\n";
            }
        } else {
            std::cout << "No memory leaks detected\n";
        }
    }
};

// Usage
#define new new(__FILE__, __LINE__)
#define delete delete(__FILE__, __LINE__)
```

## 🔧 Error Reference

### Common Error Codes
```cpp
// TDA Platform error codes
enum class TDAErrorCode {
    SUCCESS = 0,
    INVALID_INPUT = 1001,
    MEMORY_ALLOCATION_FAILED = 1002,
    COMPUTATION_FAILED = 1003,
    FILE_NOT_FOUND = 1004,
    INVALID_FORMAT = 1005,
    OUT_OF_MEMORY = 1006,
    TIMEOUT = 1007
};

// Error handling
class TDAException : public std::exception {
private:
    TDAErrorCode code_;
    std::string message_;
    
public:
    TDAException(TDAErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
    
    TDAErrorCode code() const { return code_; }
};
```

### Python Exception Handling
```python
# Python exception handling
from tda_backend import TDAEngine, TDAException

try:
    engine = TDAEngine()
    results = engine.compute_persistent_homology(
        points, method="vietoris_rips", max_dimension=2
    )
except TDAException as e:
    print(f"TDA Error {e.code}: {e.message}")
    # Handle specific error codes
    if e.code == 1002:  # Memory allocation failed
        print("Try reducing dataset size or increasing memory")
    elif e.code == 1003:  # Computation failed
        print("Check input data validity")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📋 Troubleshooting Checklist

### Before Reporting an Issue
- [ ] Check the [Common Issues](./common-issues/) section
- [ ] Verify your environment meets requirements
- [ ] Try the quick fixes above
- [ ] Collect error logs and system information
- [ ] Reproduce the issue with minimal test case

### Information to Include
- **Environment**: OS, compiler version, Python version
- **Error Message**: Complete error text and stack trace
- **Steps to Reproduce**: Exact commands and input data
- **Expected vs Actual**: What you expected vs what happened
- **Previous Working**: When it last worked (if applicable)

### Getting Help
- **GitHub Issues**: Create detailed issue reports
- **Documentation**: Check related sections
- **Community**: Use project discussion forums
- **Support**: Contact maintainers for critical issues

## 🔗 Related Documentation

- **[Performance Guide](../performance/)** - Performance optimization
- **[API Reference](../api/)** - API usage and error handling
- **[Integration Guides](../integration/)** - Integration troubleshooting
- **[Examples](../examples/)** - Working examples and patterns

## 📞 Troubleshooting Support

- **Documentation Issues**: Create a GitHub issue
- **Code Problems**: Include error logs and reproduction steps
- **Performance Issues**: Provide benchmark results and system specs

---

*Still having trouble? Check the [Common Issues](./common-issues/) section for specific problems, or use the [Debugging Guide](./debugging/) for advanced troubleshooting techniques.*

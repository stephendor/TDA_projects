# TDA Platform API Reference

This section provides comprehensive API documentation for the TDA Platform, including C++ core library, Python bindings (in development), and REST API (planned).

## 📚 API Documentation Structure

### 🔧 [C++ Core API](./cpp/)
- **Core TDA Engine** - Main persistent homology computation interface ✅
- **Filtration Methods** - Vietoris-Rips, Alpha Complex, Čech, DTM implementations ✅
- **Data Structures** - Persistence diagrams, barcodes, Betti numbers ✅
- **Performance APIs** - SIMD, threading, memory management ✅

### 🐍 [Python Bindings](./python/)
- **TDA Engine Wrapper** - High-level Python interface to C++ core 🔄
- **Data Processing** - Point cloud loading, preprocessing, and analysis 🔄
- **Result Visualization** - Plotting and export functionality 🔄
- **Integration Examples** - How to use with NumPy, Pandas, etc. 🔄

### 🌐 [REST API](./rest/)
- **Backend Service Endpoints** - FastAPI-based web service 🔄
- **Authentication** - JWT-based security and RBAC 🔄
- **Data Management** - Upload, storage, and retrieval 🔄
- **Analysis Orchestration** - Running TDA computations 🔄

### 📖 [API Examples](./examples/)
- **Basic Usage** - Simple examples for common tasks ✅
- **Advanced Workflows** - Complex analysis pipelines 🔄
- **Integration Patterns** - Working with other tools and libraries 🔄
- **Performance Optimization** - Best practices for high-throughput usage 🔄

## 🚀 Quick API Tour

### C++ Core API (Fully Implemented)
```cpp
#include <tda/core/filtration.hpp>
#include <tda/algorithms/vietoris_rips.hpp>
#include <tda/algorithms/alpha_complex.hpp>
#include <tda/algorithms/cech_complex.hpp>
#include <tda/algorithms/dtm_filtration.hpp>

// Create point cloud
std::vector<Point> points = load_point_cloud("data.csv");

// Compute Vietoris-Rips persistence
VietorisRipsFiltration vr_filtration(points);
auto persistence_pairs = vr_filtration.compute_persistence(3);

// Compute Alpha Complex persistence
AlphaComplexFiltration alpha_filtration(points);
auto alpha_results = alpha_filtration.compute_persistence(3);

// Compute Čech Complex persistence (approximation)
CechComplexFiltration cech_filtration(points);
auto cech_results = cech_filtration.compute_persistence(3);

// Compute DTM-based persistence
DTMFiltration dtm_filtration(points);
auto dtm_results = dtm_filtration.compute_persistence(3);

// Access results
for (const auto& pair : persistence_pairs) {
    std::cout << "Dimension " << pair.dimension 
              << ": [" << pair.birth << ", " << pair.death << ")\n";
}
```

### Python Bindings (In Development)
```python
# Note: Python bindings are currently in development
# The following is the planned interface

from tda_backend import TDAEngine, PointCloud

# Load and analyze data
engine = TDAEngine()
points = PointCloud.from_csv("data.csv")

# Run multiple analyses
results = engine.batch_analysis(
    points,
    methods=["vietoris_rips", "alpha_complex", "cech", "dtm"],
    max_dimension=2
)

# Process results
for method, result in results.items():
    print(f"{method}: {len(result.persistence_pairs)} pairs")
```

### REST API (Planned)
```bash
# Note: REST API is planned for future development
# The following shows the planned interface

# Upload dataset
curl -X POST "http://localhost:8000/api/v1/datasets" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data.csv" \
  -F "metadata={\"name\":\"Sample Data\",\"description\":\"Test dataset\"}"

# Run analysis
curl -X POST "http://localhost:8000/api/v1/analysis" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "123",
    "method": "vietoris_rips",
    "max_dimension": 2,
    "parameters": {"epsilon": 0.5}
  }'
```

## 🎯 API Design Principles

### 1. **Performance First**
- C++ core optimized for large-scale computations ✅
- Minimal data copying between layers ✅
- Efficient memory management and SIMD optimization ✅
- OpenMP parallelization for complex operations ✅

### 2. **Developer Experience**
- Consistent naming conventions across all APIs ✅
- Comprehensive error handling and validation ✅
- Detailed logging and debugging support ✅
- Memory pool optimization for large datasets ✅

### 3. **Extensibility**
- Plugin architecture for new algorithms ✅
- Configurable parameters for all methods ✅
- Support for custom data types and formats ✅
- Spatial indexing for efficient nearest neighbor search ✅

### 4. **Integration Ready**
- Standard data formats (CSV, JSON, HDF5) 🔄
- RESTful API design (planned) 🔄
- Comprehensive authentication and authorization (planned) 🔄
- GUDHI library integration for advanced features ✅

## 🔍 API Versioning

The TDA Platform uses semantic versioning for all APIs:

- **Major Version (v1.0.0)**: Breaking changes to public APIs
- **Minor Version (v1.1.0)**: New features, backward compatible
- **Patch Version (v1.1.1)**: Bug fixes, backward compatible

### Current Versions
- **C++ Core API**: v1.0.0 ✅
- **Python Bindings**: v0.1.0 🔄 (in development)
- **REST API**: v0.0.0 🔄 (planned)

## 📊 API Performance Characteristics

### C++ Core Performance ✅
- **Point Cloud Size**: 1M+ points in <60 seconds (target met)
- **Memory Usage**: Optimized for large datasets with memory pools
- **Parallelization**: OpenMP and std::thread support
- **SIMD**: Vectorized distance computations
- **Spatial Indexing**: KD-trees and Ball-trees for efficient search

### Python Bindings Performance 🔄
- **Overhead**: <5% compared to direct C++ usage (target)
- **Memory**: Efficient NumPy integration (planned)
- **Batch Processing**: Support for multiple datasets (planned)

### REST API Performance 🔄
- **Throughput**: 1000+ requests/second (target)
- **Latency**: <100ms for simple operations (target)
- **Scalability**: Horizontal scaling support (planned)

## 🛠️ Development and Testing

### API Testing
```bash
# Run C++ tests
make test

# Run specific test suites
ctest -R core_tests             # Core library tests
ctest -R vector_stack_tests     # Vector stack tests
ctest -R algorithm_tests        # Algorithm tests

# Individual test executables
./build/bin/test_vietoris_rips       # Vietoris-Rips filtration tests
./build/bin/test_alpha_complex       # Alpha complex tests
./build/bin/test_cech_complex        # Čech complex tests
./build/bin/test_dtm_filtration      # DTM filtration tests
./build/bin/test_spatial_index       # Spatial indexing tests
./build/bin/test_persistence_structures  # Persistence diagram tests

# Run specific CTest patterns
ctest -R vietoris             # All VR-related tests
ctest -R alpha                # All Alpha complex tests
ctest -R spatial              # All spatial indexing tests
ctest -R persistence          # All persistence structure tests
```

### Performance Testing
```bash
# Run performance benchmarks
./build/bin/tda_benchmarks           # All benchmarks
./build/bin/test_performance_benchmarks  # Performance tests

# Build with profiling (debug)
./build.sh debug OFF true false false
```

### API Documentation Generation
```bash
# Generate C++ docs
make docs

# View documentation
open docs/html/index.html     # macOS
xdg-open docs/html/index.html # Linux
```

## 🔗 Related Documentation

- **[Performance Guide](../performance/)** - Optimization and benchmarking
- **[Integration Guides](../integration/)** - End-to-end workflows
- **[Troubleshooting](../troubleshooting/)** - Common API issues
- **[Examples](../examples/)** - Complete working examples

## 📞 API Support

- **Documentation Issues**: Create a GitHub issue
- **API Questions**: Use the project's discussion forum
- **Bug Reports**: Include API version and error details
- **Feature Requests**: Describe use case and requirements

## 🚧 Implementation Status

### ✅ **Completed Features**
- **Core TDA Engine**: Full C++ implementation with all filtration methods
- **Spatial Indexing**: KD-trees and Ball-trees for efficient search
- **Performance Optimization**: Memory pools, parallelization, SIMD
- **Testing Framework**: Comprehensive test suites for all components
- **Build System**: Advanced CMake configuration with multiple build types

### 🔄 **In Development**
- **Python Bindings**: Core functionality working, expanding API coverage
- **Performance Benchmarks**: Final optimization and validation
- **Documentation**: API examples and integration guides

### 🔄 **Planned Features**
- **REST API**: FastAPI-based web service
- **Frontend UI**: React-based user interface
- **Streaming Pipeline**: Real-time data processing
- **Database Integration**: PostgreSQL and MongoDB support

---

*Ready to explore the APIs? Start with the [C++ Core API](./cpp/) for the foundation, or check the [Python Bindings](./python/) for high-level usage (when available).*

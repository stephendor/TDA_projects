# TDA Vector Stack

A modern, high-performance C++23 implementation of Topological Data Analysis (TDA) with a focus on vector stack operations and persistent homology computation.

## 🚀 Features

- **Modern C++23**: Leverages latest C++ features including concepts, ranges, and coroutines
- **High Performance**: SIMD optimizations, OpenMP parallelization, and memory pools
- **Clean Architecture**: Modular design with clear separation of concerns
- **Vector Stack Focus**: Optimized data structure for TDA operations
- **Python Bindings**: Seamless integration via pybind11
- **Comprehensive Testing**: Full test suite with benchmarks

## 🏗️ Architecture

```
src/
├── cpp/
│   ├── core/           # Core TDA types and algorithms
│   ├── vector_stack/   # Main vector stack implementation
│   ├── algorithms/     # TDA algorithms (VR, Alpha, etc.)
│   └── utils/          # Performance utilities
├── python/             # Python bindings
└── tests/              # Test suite

include/
└── tda/                # Public headers

examples/               # Example implementations
├── vectorization_storage_example.md         # Documentation example
├── vectorization_storage_example_simplified.cpp  # Simplified standalone example
└── README.md           # Example documentation

research/               # Extracted high-value research
├── papers/             # Research papers
├── implementations/    # Key algorithm implementations
└── benchmarks/         # Performance benchmarks
```

## 📊 Examples

The project includes examples demonstrating key TDA concepts:

### Vectorization and Storage Example

A comprehensive example showing:

- Persistence diagram computation
- Vectorization techniques (Betti curves, persistence landscapes, persistence images)
- Storage in databases (PostgreSQL for metadata, MongoDB for raw data)

To build and run the simplified standalone example:

```bash
./direct_build.sh
./build/direct/vectorization_example
```

For a guided walkthrough with explanations:

```bash
./run_example_with_explanation.sh
```

See the `examples` directory for more information.

## 📋 Requirements

- **Compiler**: GCC 13+ or Clang 16+ with C++23 support
- **CMake**: 3.20+
- **Dependencies**: Eigen3, OpenMP, pybind11
- **Optional**: CUDA for GPU acceleration

## 🛠️ Building

### Quick Build
```bash
./build.sh release
```

### Custom Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Options
- `BUILD_TYPE`: debug, release, relwithdebinfo
- `ENABLE_CUDA`: ON/OFF for GPU acceleration
- `CLEAN_BUILD`: true/false for clean rebuild
- `RUN_TESTS`: true/false to run tests
- `RUN_BENCHMARKS`: true/false to run benchmarks

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test suite
ctest -R core_tests
ctest -R vector_stack_tests
ctest -R algorithm_tests
```

## 📊 Benchmarks

```bash
# Run performance benchmarks
./bin/tda_benchmarks

# Specific benchmark categories
./bin/tda_benchmarks --vector-operations
./bin/tda_benchmarks --persistent-homology
./bin/tda_benchmarks --memory-usage
```

## 🐍 Python Usage

```python
import tda

# Create a vector stack
vs = tda.VectorStack(100, 3)  # 100 points in 3D

# Compute persistent homology
persistence_diagram = tda.compute_persistence(vs)

# Get Betti numbers
betti = tda.compute_betti_numbers(persistence_diagram)
```

## 🔬 Research Integration

This project integrates key research from:
- **ICLR 2021 Challenge**: Noise-invariant topological features
- **ICLR 2022 Challenge**: Tree embeddings and structured data
- **Geomstats**: Riemannian geometry and statistics
- **Giotto-Deep**: TDA + Deep Learning integration
- **TopoBench**: Performance benchmarking

## 📚 Documentation

- **API Documentation**: `make docs` generates Doxygen documentation
- **Design Documents**: See `docs/design/` for architecture details
- **Performance Guides**: See `docs/performance/` for optimization tips
- **Background Agents**: See `docs/background_agents/` for agent playbooks (Tasks 2, 5, 6, 7, 8, 9, 10)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement with C++23 best practices
4. Add comprehensive tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ICLR Challenge organizers and participants
- Geomstats and Giotto-Deep communities
- C++ Standards Committee for C++23 features
- OpenMP and Eigen3 maintainers

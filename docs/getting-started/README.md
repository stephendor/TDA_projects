# Getting Started with TDA Platform

Welcome to the Topological Data Analysis (TDA) Platform! This guide will help you get up and running quickly, whether you're a developer, researcher, or analyst.

## 🚀 Quick Start (10 minutes)

### Prerequisites
- **C++23 Compiler**: GCC 13+ or Clang 16+ (strictly required)
- **Python 3.9+** with pip/poetry
- **CMake 3.20+**
- **Git**
- **GUDHI Library**: `libgudhi-dev` package

### 1. Clone and Build
```bash
# Clone the repository
git clone <your-repo-url>
cd TDA_projects

# Build the platform (recommended)
./build.sh release

# Alternative build options
./build.sh debug              # Debug build with sanitizers
./build.sh release ON true   # Release build with CUDA support
./build.sh debug OFF true    # Debug build, clean, run tests
```

### 2. Run Your First Analysis
```bash
# Test the core TDA engine
./build/bin/test_vietoris_rips
./build/bin/test_alpha_complex
./build/bin/test_cech_complex
./build/bin/test_dtm_filtration

# Run performance benchmarks
./build/bin/tda_benchmarks
```

### 3. Explore Results
```bash
# Run comprehensive tests
make test

# Check specific test suites
ctest -R core_tests
ctest -R vector_stack_tests
ctest -R algorithm_tests
```

## 📖 What You'll Learn

This getting started guide covers:

- **Platform Overview** - Understanding the TDA platform architecture
- **Core Concepts** - Essential TDA and persistent homology concepts
- **Installation** - Complete development environment setup
- **First Analysis** - Running your first topological analysis
- **Next Steps** - Where to go from here

## 🎯 Choose Your Path

### 👨‍💻 **I'm a Developer**
- [Development Environment Setup](../training/development_environment_setup.md)
- [C++23 Training Guide](../training/cpp23_training_guide.md)
- [Codebase Navigation](../training/codebase_navigation.md)

### 🔬 **I'm a Researcher**
- [Core Algorithms](../algorithms/)
- [Performance Optimization](../performance/)
- [Integration Examples](../examples/)

### 📊 **I'm an Analyst**
- [API Reference](../api/)
- [End-to-End Workflows](../integration/)
- [Troubleshooting](../troubleshooting/)

## 🏗️ Platform Architecture Overview

The TDA platform consists of several key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   Backend API   │    │   C++ Core      │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (TDA Engine)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Streaming     │
                       │   (Kafka/Flink) │
                       └─────────────────┘
```

### Core Components

1. **C++ TDA Engine** - High-performance persistent homology computation
   - **Core Library**: `tda_core` - Basic data structures and types
   - **Vector Stack**: `tda_vector_stack` - Main TDA computation engine
   - **Algorithms**: Vietoris-Rips, Alpha Complex, Čech, DTM filtrations
   - **Spatial Indexing**: KD-trees and Ball-trees for efficient search

2. **Python Backend** - API orchestration and data management (in development)
3. **React Frontend** - User interface and visualization (planned)
4. **Streaming Pipeline** - Real-time data processing (planned)
5. **Database Layer** - PostgreSQL + MongoDB hybrid storage (planned)

## 🧠 Key TDA Concepts

### Persistent Homology
Persistent homology tracks how topological features (connected components, holes, voids) appear and disappear as you vary a parameter (like distance threshold).

### Filtration Methods
- **Vietoris-Rips** - Distance-based complex construction ✅
- **Alpha Complex** - Geometric complex with better properties ✅
- **Čech Complex** - Intersection-based complex (approximation) ✅
- **DTM Filtration** - Noise-resistant distance-to-measure ✅

### Persistence Diagrams
Visual representation of when topological features are "born" and "die" during the filtration process.

## 🔧 Development Environment

### Required Tools
- **Compiler**: GCC 13+ or Clang 16+ (C++23 support required)
- **Build System**: CMake 3.20+
- **Python**: 3.9+ with virtual environment
- **Dependencies**: GUDHI, Eigen3, OpenMP

### Optional Tools
- **CUDA Toolkit** - For GPU acceleration
- **Intel MKL** - For optimized linear algebra
- **Docker** - For containerized development

### Build Configuration
```bash
# Available build options
./build.sh [BUILD_TYPE] [ENABLE_CUDA] [CLEAN_BUILD] [RUN_TESTS] [RUN_BENCHMARKS]

# Examples
./build.sh release              # Standard release build
./build.sh debug               # Debug build with sanitizers
./build.sh release ON false    # Release with CUDA, no clean
./build.sh debug OFF true      # Debug, clean, run tests
```

## 📚 Next Steps

After completing this guide:

1. **Explore Examples** - Check out the [examples directory](../examples/)
2. **Read API Docs** - Understand the [API reference](../api/)
3. **Run Benchmarks** - Test performance with [benchmarking tools](../performance/)
4. **Join Development** - Contribute to the platform

## 🆘 Need Help?

- **Documentation Issues**: Create a GitHub issue
- **Code Problems**: Check the [troubleshooting guide](../troubleshooting/)
- **Questions**: Use the project's discussion forum

---

*Ready to dive deeper? Choose your path above or continue with the [Installation Guide](installation.md).*

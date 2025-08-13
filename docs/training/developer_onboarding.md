# Developer Onboarding Guide

Welcome to the TDA Platform development team! This guide will help you get up to speed with our codebase, development practices, and workflow.

## 🎯 What You'll Learn

- **Platform Overview** - Understanding the TDA platform architecture
- **Development Environment** - Setting up your development workspace
- **Codebase Navigation** - Finding your way around the project
- **Development Workflow** - Our coding standards and practices
- **Testing & Quality** - How we ensure code quality
- **Common Tasks** - Typical development activities

## 🏗️ Platform Architecture Overview

### High-Level Architecture
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

### Key Components

1. **C++ TDA Engine** - High-performance persistent homology computation ✅
   - **Core Library**: `tda_core` - Basic data structures and types
   - **Vector Stack**: `tda_vector_stack` - Main TDA computation engine
   - **Algorithms**: Vietoris-Rips, Alpha Complex, Čech, DTM filtrations
   - **Spatial Indexing**: KD-trees and Ball-trees for efficient search

2. **Python Backend** - API orchestration and data management 🔄 (in development)
3. **React Frontend** - User interface and visualization 🔄 (planned)
4. **Streaming Pipeline** - Real-time data processing 🔄 (planned)
5. **Database Layer** - PostgreSQL + MongoDB hybrid storage 🔄 (planned)

## 🚀 Quick Start (First Day)

### 1. **Clone and Build**
```bash
# Clone the repository
git clone <your-repo-url>
cd TDA_projects

# Build the platform (recommended)
./build.sh release

# Alternative build options
./build.sh debug              # Debug build with sanitizers
./build.sh release ON false  # Release with CUDA support
./build.sh debug OFF true    # Debug, clean, run tests
```

### 2. **Run Your First Test**
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
```

### 3. **Start Development Server**
```bash
# Backend API (in development)
cd backend
python -m uvicorn tda_backend.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (planned)
cd frontend
npm start
```

## 🔧 Development Environment Setup

### Required Tools
- **C++23 Compiler**: GCC 13+ or Clang 16+ (strictly required)
- **CMake**: 3.20+
- **Python**: 3.9+ with virtual environment
- **Node.js**: 18+ for frontend development (when available)
- **Git**: Latest version with proper configuration
- **GUDHI Library**: `libgudhi-dev` package

### IDE Setup
We recommend using **Cursor** with our custom rules, but any modern IDE will work:

#### Cursor (Recommended)
- Install Cursor
- Clone the repository
- Open in Cursor - rules will auto-load
- Use AI assistance for development

#### VS Code
- Install C++ extension
- Install Python extension
- Install CMake Tools extension
- Configure IntelliSense for C++

#### CLion
- Open project as CMake project
- Configure Python interpreter
- Set up debugging configurations

### Environment Variables
```bash
# Add to your ~/.bashrc or ~/.zshrc
export TDA_LOG_LEVEL=DEBUG
export TDA_DATABASE_URL=postgresql://user:pass@localhost/tda
export TDA_MONGODB_URL=mongodb://localhost:27017/tda

# For development
export TDA_ENV=development
export TDA_DEBUG=true
```

## 🗺️ Codebase Navigation

### Project Structure
```
TDA_projects/
├── include/           # C++ headers
│   └── tda/
│       ├── algorithms/    # TDA algorithm implementations ✅
│       ├── core/          # Core data structures ✅
│       ├── spatial/       # Spatial indexing ✅
│       └── utils/         # Utility functions ✅
├── src/               # C++ source files
│   ├── cpp/           # Main C++ implementation ✅
│   └── python/        # Python bindings 🔄
├── backend/           # Python backend service 🔄
│   ├── tda_backend/   # Main backend package
│   ├── tests/         # Backend tests
│   └── pyproject.toml # Python dependencies
├── frontend/          # React frontend 🔄
├── docs/              # Documentation ✅
├── tests/             # C++ tests ✅
└── build.sh           # Build script ✅
```

### Key Files to Know
- **`build.sh`** - Main build script for the entire platform ✅
- **`CMakeLists.txt`** - C++ build configuration ✅
- **`backend/pyproject.toml`** - Python dependencies 🔄
- **`frontend/package.json`** - Frontend dependencies 🔄
- **`include/tda/core/filtration.hpp`** - Core TDA interface ✅
- **`src/cpp/core/filtration.cpp`** - Core TDA implementation ✅

### Finding Your Way Around
```bash
# Find C++ functions
grep -r "class.*Filtration" include/
grep -r "compute_persistence" src/cpp/

# Find Python functions
grep -r "def compute_persistent_homology" backend/

# Find test files
find . -name "*test*.py" -o -name "*test*.cpp"
```

## 🧠 Understanding TDA Concepts

### What is Topological Data Analysis?
TDA is a mathematical approach to analyzing data by studying its shape and structure. Think of it as understanding the "holes" and "connections" in your data.

### Key Concepts
- **Point Cloud**: A set of points in space (your data)
- **Filtration**: Building a sequence of shapes from your data
- **Persistent Homology**: Tracking how features appear and disappear
- **Persistence Diagram**: Visual representation of the results

### Example: Vietoris-Rips Filtration
```cpp
// 1. Start with points
std::vector<Point> points = {{0,0}, {1,0}, {0,1}};

// 2. Build complex by connecting nearby points
// At ε=0: Just the points
// At ε=1: Points + edges
// At ε=√2: Points + edges + triangle

// 3. Track when features appear/disappear
// H0 (connected components): 3 → 2 → 1
// H1 (holes): 0 → 0 → 1
```

## 🔄 Development Workflow

### 1. **Getting a Task**
- Check Taskmaster for your assigned tasks
- Read the task description and requirements
- Understand dependencies and acceptance criteria

### 2. **Development Cycle**
```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes
# ... edit code ...

# 3. Test your changes
make test                    # C++ tests
cd backend && pytest         # Python tests (when available)
cd frontend && npm test      # Frontend tests (when available)

# 4. Build and verify
./build.sh release
./build/bin/test_vietoris_rips

# 5. Commit and push
git add .
git commit -m "feat: implement your feature"
git push origin feature/your-feature-name
```

### 3. **Code Review Process**
- Create a pull request
- Request review from team members
- Address feedback and iterate
- Merge after approval

## 📝 Coding Standards

### C++ Standards
```cpp
// Use C++23 features
#include <ranges>
#include <concepts>

// Prefer modern C++ patterns
auto points = std::vector<Point>{};
auto result = std::ranges::transform_view(points, [](const auto& p) {
    return p.normalize();
});

// Use RAII and smart pointers
class SafeResource {
private:
    std::unique_ptr<Resource> resource_;
    
public:
    SafeResource() : resource_(std::make_unique<Resource>()) {}
    // Destructor automatically cleans up
};
```

### Python Standards
```python
# Use type hints
from typing import List, Optional, Dict
import numpy as np

def compute_persistent_homology(
    points: np.ndarray,
    method: str = "vietoris_rips",
    max_dimension: int = 2
) -> Dict[str, np.ndarray]:
    """Compute persistent homology for a point cloud.
    
    Args:
        points: Input point cloud as numpy array
        method: Filtration method to use
        max_dimension: Maximum homology dimension
        
    Returns:
        Dictionary containing persistence pairs and other results
    """
    # Implementation...
    pass

# Use dataclasses for data structures
from dataclasses import dataclass

@dataclass
class PersistencePair:
    dimension: int
    birth: float
    death: float
    confidence: Optional[float] = None
```

### Documentation Standards
- **Code Comments**: Explain WHY, not WHAT
- **Function Documentation**: Use docstrings with examples
- **README Files**: Keep updated with usage examples
- **API Documentation**: Include parameter descriptions and return values

## 🧪 Testing & Quality

### Testing Philosophy
- **Test-Driven Development**: Write tests first when possible
- **Coverage**: Aim for >90% test coverage
- **Integration Tests**: Test complete workflows
- **Performance Tests**: Ensure performance requirements are met

### Running Tests
```bash
# C++ Tests ✅
make test                    # All tests
ctest -R core_tests         # Specific test suite
./build/bin/test_vietoris_rips    # Individual test executable

# Python Tests 🔄
cd backend
pytest tests/                # All tests (when available)
pytest tests/api/           # API tests only (when available)
pytest -v -k "test_upload"  # Tests matching pattern (when available)

# Frontend Tests 🔄
cd frontend
npm test                     # Unit tests (when available)
npm run test:integration     # Integration tests (when available)
```

### Writing Tests
```cpp
// C++ Test Example ✅
#include <gtest/gtest.h>
#include <tda/algorithms/vietoris_rips.hpp>

class VietorisRipsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test data
        points_ = {{0,0}, {1,0}, {0,1}, {1,1}};
    }
    
    std::vector<Point> points_;
};

TEST_F(VietorisRipsTest, BasicComputation) {
    VietorisRipsFiltration filtration(points_);
    auto results = filtration.compute_persistence(2);
    
    EXPECT_GT(results.persistence_pairs.size(), 0);
    EXPECT_EQ(results.persistence_pairs[0].dimension, 0);
}
```

```python
# Python Test Example 🔄
import pytest
import numpy as np
from tda_backend import TDAEngine

class TestTDAEngine:
    def setup_method(self):
        self.engine = TDAEngine()
        self.test_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    def test_basic_computation(self):
        results = self.engine.compute_persistent_homology(
            self.test_points, method="vietoris_rips", max_dimension=2
        )
        
        assert len(results.persistence_pairs) > 0
        assert results.persistence_pairs[0].dimension == 0
    
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            self.engine.compute_persistent_homology(
                np.array([]), method="vietoris_rips"
            )
```

## 🚀 Common Development Tasks

### Adding a New Algorithm ✅
1. **Create Header File**
```cpp
// include/tda/algorithms/your_algorithm.hpp
#pragma once
#include <tda/core/filtration.hpp>
#include <vector>

namespace tda {
namespace algorithms {

class YourAlgorithm : public Filtration {
public:
    explicit YourAlgorithm(const std::vector<Point>& points);
    
    std::vector<PersistencePair> compute_persistence(size_t max_dimension) override;
    
private:
    std::vector<Point> points_;
    // Implementation details...
};

} // namespace algorithms
} // namespace tda
```

2. **Implement Algorithm**
```cpp
// src/cpp/algorithms/your_algorithm.cpp
#include <tda/algorithms/your_algorithm.hpp>

namespace tda {
namespace algorithms {

YourAlgorithm::YourAlgorithm(const std::vector<Point>& points)
    : points_(points) {}

std::vector<PersistencePair> YourAlgorithm::compute_persistence(size_t max_dimension) {
    // Your algorithm implementation
    std::vector<PersistencePair> results;
    
    // ... computation logic ...
    
    return results;
}

} // namespace algorithms
} // namespace tda
```

3. **Add Tests**
```cpp
// tests/cpp/test_your_algorithm.cpp
#include <gtest/gtest.h>
#include <tda/algorithms/your_algorithm.hpp>

TEST(YourAlgorithmTest, BasicComputation) {
    std::vector<Point> points = {{0,0}, {1,0}, {0,1}};
    YourAlgorithm algorithm(points);
    
    auto results = algorithm.compute_persistence(2);
    EXPECT_GT(results.size(), 0);
}
```

4. **Update Build System**
```cmake
# CMakeLists.txt
add_executable(test_your_algorithm tests/cpp/test_your_algorithm.cpp)
target_link_libraries(test_your_algorithm tda_core gtest)
add_test(NAME YourAlgorithmTest COMMAND test_your_algorithm)
```

### Adding Python Bindings 🔄
```cpp
// src/python/your_algorithm_bindings.cpp
#include <pybind11/pybind11.h>
#include <tda/algorithms/your_algorithm.hpp>

namespace py = pybind11;

PYBIND11_MODULE(your_algorithm_module, m) {
    py::class_<tda::algorithms::YourAlgorithm>(m, "YourAlgorithm")
        .def(py::init<const std::vector<Point>&>())
        .def("compute_persistence", &tda::algorithms::YourAlgorithm::compute_persistence)
        .def("get_points", &tda::algorithms::YourAlgorithm::get_points);
}
```

### Adding API Endpoints 🔄
```python
# backend/tda_backend/api/v1/your_endpoint.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tda_backend.services.tda_service import TDAService

router = APIRouter()
tda_service = TDAService()

class YourRequest(BaseModel):
    data: list
    parameters: dict

class YourResponse(BaseModel):
    result: list
    metadata: dict

@router.post("/your-endpoint", response_model=YourResponse)
async def your_endpoint(request: YourRequest):
    try:
        result = await tda_service.process_your_request(request.data, request.parameters)
        return YourResponse(result=result, metadata={"status": "success"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🔍 Debugging Tips

### C++ Debugging ✅
```bash
# Build with debug symbols
./build.sh debug

# Use GDB
gdb ./build/bin/test_vietoris_rips
(gdb) break main
(gdb) run
(gdb) next
(gdb) print variable_name

# Use Valgrind for memory issues
valgrind --tool=memcheck --leak-check=full ./build/bin/test_vietoris_rips
```

### Python Debugging 🔄
```python
# Use pdb
import pdb; pdb.set_trace()

# Use logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Performance Debugging ✅
```bash
# Profile C++ code
valgrind --tool=callgrind ./build/bin/test_performance_benchmarks
kcachegrind callgrind.out.*

# Profile Python code
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(10)"
```

## 📚 Learning Resources

### TDA Theory
- **Books**: "Computational Topology" by Edelsbrunner & Harer
- **Papers**: Check the `research/papers/` directory
- **Online**: Computational Topology course materials

### C++23 Features
- **C++ Reference**: https://en.cppreference.com/
- **Modern C++**: Effective Modern C++ by Scott Meyers
- **C++ Core Guidelines**: https://isocpp.github.io/CppCoreGuidelines/

### Python Development
- **Python Docs**: https://docs.python.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **NumPy**: https://numpy.org/doc/

### Development Tools
- **Git**: https://git-scm.com/doc
- **CMake**: https://cmake.org/documentation/
- **Docker**: https://docs.docker.com/

## 🤝 Getting Help

### Team Communication
- **Daily Standups**: Join team standup meetings
- **Code Reviews**: Ask questions during reviews
- **Pair Programming**: Work with experienced team members
- **Documentation**: Check docs first, then ask

### When You're Stuck
1. **Check Documentation**: Look in `docs/` directory
2. **Search Issues**: Check GitHub issues for similar problems
3. **Ask Team**: Reach out to team members
4. **Create Issue**: Document the problem for the team

### Contributing to Documentation
- Keep documentation up to date
- Add examples for new features
- Update this guide as you learn
- Share tips with other team members

## 🎯 Next Steps

### Week 1 Goals
- [ ] Set up development environment
- [ ] Build and run the platform
- [ ] Understand basic TDA concepts
- [ ] Complete first simple task

### Week 2 Goals
- [ ] Contribute to a small feature
- [ ] Write tests for your code
- [ ] Participate in code reviews
- [ ] Understand the development workflow

### Month 1 Goals
- [ ] Contribute to a major feature
- [ ] Understand the full codebase
- [ ] Help other team members
- [ ] Suggest improvements

## 🔗 Related Documentation

- **[Development Environment Setup](development_environment_setup.md)** - Detailed setup instructions
- **[C++23 Training Guide](cpp23_training_guide.md)** - C++ specific training
- **[API Reference](../api/)** - Understanding the APIs
- **[Performance Guide](../performance/)** - Optimization techniques
- **[Troubleshooting](../troubleshooting/)** - Common problems and solutions

## 🚧 Implementation Status

### ✅ **Completed Features**
- **Core TDA Engine**: Full C++ implementation with all filtration methods
- **Spatial Indexing**: KD-trees and Ball-trees for efficient search
- **Performance Optimization**: Memory pools, parallelization, SIMD
- **Testing Framework**: Comprehensive test suites for all components
- **Build System**: Advanced CMake configuration with multiple build types

### 🔄 **In Development**
- **Python Bindings**: Core functionality working, expanding API coverage
- **Backend API**: FastAPI service structure in place
- **Performance Benchmarks**: Final optimization and validation

### 🔄 **Planned Features**
- **Frontend UI**: React-based user interface
- **Streaming Pipeline**: Real-time data processing
- **Database Integration**: PostgreSQL and MongoDB support

---

*Welcome to the team! Remember, there are no stupid questions. We're all here to learn and build something amazing together.*

*Need help? Check the [Troubleshooting Guide](../troubleshooting/) or ask your team members.*

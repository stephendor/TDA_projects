# TDA Vector Stack Development Environment Setup Guide

## 🎯 **Overview**

This guide walks you through setting up the complete development environment for the TDA Vector Stack project. We'll cover everything from system requirements to running your first build.

## 🖥️ **System Requirements**

### **Operating System**
- **Linux**: Ubuntu 20.04+ (recommended)
- **macOS**: 10.15+ (with Homebrew)
- **Windows**: Windows 10+ with WSL2 (recommended) or Visual Studio 2022

### **Hardware Requirements**
- **CPU**: x86_64 with AVX2 support (for SIMD optimizations)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB+ free space for dependencies and build artifacts

## 🛠️ **Required Software**

### **1. C++ Compiler**
- **GCC**: 13.0+ (recommended)
- **Clang**: 16.0+ (alternative)
- **MSVC**: 2022 17.0+ (Windows only)

**Verify C++23 Support**:
```bash
g++ --version
g++ -std=c++23 -E -x c++ /dev/null
```

### **2. Build System**
- **CMake**: 3.20+ (required)
- **Make**: 4.0+ (Linux/macOS)
- **Ninja**: 1.10+ (alternative, faster)

**Install CMake**:
```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# Or use snap
snap install cmake --classic
```

### **3. Python Environment**
- **Python**: 3.8+ (3.13+ recommended)
- **Virtual Environment**: python3-venv

**Setup Virtual Environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## 📦 **Dependencies Installation**

### **Core Dependencies**

#### **Eigen3 (Linear Algebra)**
```bash
# Ubuntu/Debian
sudo apt install libeigen3-dev

# macOS
brew install eigen

# Verify installation
pkg-config --modversion eigen3
```

#### **OpenMP (Parallelization)**
```bash
# Ubuntu/Debian
sudo apt install libomp-dev

# macOS
brew install libomp

# Verify installation
g++ -fopenmp -E -x c++ /dev/null
```

#### **pybind11 (Python Bindings)**
```bash
# Activate virtual environment first
source .venv/bin/activate

# Install pybind11
pip install pybind11

# Verify installation
python -c "import pybind11; print(pybind11.__version__)"
```

### **Optional Dependencies**

#### **CUDA (GPU Acceleration)**
```bash
# Install CUDA toolkit if you have NVIDIA GPU
# This is optional for development
sudo apt install nvidia-cuda-toolkit
```

#### **Doxygen (Documentation)**
```bash
# Ubuntu/Debian
sudo apt install doxygen

# macOS
brew install doxygen
```

## 🚀 **Project Setup**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd TDA_projects
```

### **2. Initialize Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if requirements.txt exists
```

### **3. Verify Project Structure**
```bash
ls -la
# Should see:
# - CMakeLists.txt
# - build.sh
# - src/
# - include/
# - tests/
# - docs/
```

## 🔨 **Building the Project**

### **First Build**
```bash
# Make build script executable
chmod +x build.sh

# Run the build script
./build.sh
```

### **Build Options**
```bash
# Release build (optimized)
./build.sh release

# Debug build (with sanitizers)
./build.sh debug

# Clean build
./build.sh release true

# Skip tests
./build.sh release false false
```

### **Manual CMake Build**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 🧪 **Testing Your Setup**

### **1. Run C++23 Feature Test**
```bash
# Compile the test program
g++ -std=c++23 -O3 -march=native -o test_cpp23 test_cpp23_features.cpp

# Run the test
./test_cpp23
```

**Expected Output**:
```
🧪 Testing C++23 Features in TDA Vector Stack Environment
=======================================================
✅ All C++23 features tested successfully!
🚀 TDA Vector Stack environment is ready for development!
```

### **2. Test Python Bindings**
```bash
# Set Python path to build directory
export PYTHONPATH="build/release/lib:$PYTHONPATH"

# Test Python module
python -c "import tda_python; print('Python module loaded successfully!')"
```

### **3. Run Benchmarks**
```bash
# Run the benchmark executable
./build/release/bin/tda_benchmarks
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Compiler Not Found**
```bash
# Check if compiler is in PATH
which g++
g++ --version

# Install if missing
sudo apt install build-essential
```

#### **CMake Not Found**
```bash
# Check CMake installation
which cmake
cmake --version

# Install if missing
sudo apt install cmake
```

#### **Dependencies Missing**
```bash
# Check Eigen3
pkg-config --exists eigen3

# Check OpenMP
g++ -fopenmp -E -x c++ /dev/null

# Install missing dependencies
sudo apt install libeigen3-dev libomp-dev
```

#### **Python Module Import Error**
```bash
# Check if module was built
ls -la build/release/lib/tda_python*

# Check Python path
echo $PYTHONPATH

# Rebuild if necessary
./build.sh
```

### **Performance Issues**

#### **Slow Builds**
- Use `make -j$(nproc)` for parallel builds
- Consider using Ninja instead of Make
- Use `ccache` for incremental builds

#### **Memory Issues**
- Reduce parallel build jobs: `make -j4`
- Use swap space if RAM is limited
- Build in release mode for smaller binaries

## 📚 **Development Workflow**

### **Daily Development**
1. **Activate environment**: `source .venv/bin/activate`
2. **Make changes** to source files
3. **Build**: `./build.sh` or `make -j$(nproc)`
4. **Test**: Run tests and benchmarks
5. **Commit**: `git add . && git commit -m "description"`

### **Adding New Features**
1. **Create header** in `include/tda/`
2. **Implement** in `src/cpp/`
3. **Add tests** in `tests/cpp/`
4. **Update CMakeLists.txt** if needed
5. **Build and test**

### **Debugging**
1. **Debug build**: `./build.sh debug`
2. **Run with sanitizers**: Built-in with debug mode
3. **Use GDB/LLDB**: `gdb ./build/debug/bin/tda_benchmarks`
4. **Check logs**: Build output and runtime logs

## 🎯 **Next Steps**

1. **Complete setup**: Follow this guide step by step
2. **Run tests**: Verify everything is working
3. **Read training materials**: Review C++23 guide
4. **Start coding**: Begin implementing TDA algorithms
5. **Join the team**: Participate in code reviews and discussions

## 📞 **Getting Help**

- **Documentation**: Check `docs/` directory
- **Issues**: Create GitHub issues for bugs
- **Team**: Reach out to team members
- **Training**: Review training materials

---

*This guide is maintained by the TDA Vector Stack development team. Last updated: [Current Date]*

#!/bin/bash

# TDA Vector Stack Build Script
# Clean, focused build for the C++23 vector stack implementation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build configuration
BUILD_TYPE=${1:-release}
ENABLE_CUDA=${2:-OFF}
CLEAN_BUILD=${3:-false}
RUN_TESTS=${4:-true}
RUN_BENCHMARKS=${5:-false}

echo -e "${BLUE}🚀 TDA Vector Stack Build Script${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Build Type: ${YELLOW}${BUILD_TYPE}${NC}"
echo -e "CUDA Support: ${YELLOW}${ENABLE_CUDA}${NC}"
echo -e "Clean Build: ${YELLOW}${CLEAN_BUILD}${NC}"
echo -e "Run Tests: ${YELLOW}${RUN_TESTS}${NC}"
echo -e "Run Benchmarks: ${YELLOW}${RUN_BENCHMARKS}${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}❌ Error: CMakeLists.txt not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Check C++ compiler
echo -e "${BLUE}🔍 Checking C++ compiler...${NC}"
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo -e "Found: ${GREEN}${GCC_VERSION}${NC}"
    
    # Check if GCC supports C++23
    if g++ -std=c++23 -E -x c++ /dev/null &> /dev/null; then
        echo -e "${GREEN}✅ GCC supports C++23${NC}"
    else
        echo -e "${RED}❌ GCC does not support C++23. Please upgrade to GCC 13+${NC}"
        exit 1
    fi
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    echo -e "Found: ${GREEN}${CLANG_VERSION}${NC}"
    
    # Check if Clang supports C++23
    if clang++ -std=c++23 -E -x c++ /dev/null &> /dev/null; then
        echo -e "${GREEN}✅ Clang supports C++23${NC}"
    else
        echo -e "${RED}❌ Clang does not support C++23. Please upgrade to Clang 16+${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Error: No C++ compiler found. Please install GCC 13+ or Clang 16+${NC}"
    exit 1
fi

# Check CMake
echo -e "${BLUE}🔍 Checking CMake...${NC}"
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo -e "Found: ${GREEN}${CMAKE_VERSION}${NC}"
    
    # Check CMake version
    CMAKE_MAJOR=$(cmake --version | head -n1 | sed 's/.*version \([0-9]*\)\.\([0-9]*\).*/\1/')
    CMAKE_MINOR=$(cmake --version | head -n1 | sed 's/.*version \([0-9]*\)\.\([0-9]*\).*/\2/')
    
    if [ "$CMAKE_MAJOR" -gt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -ge 20 ]); then
        echo -e "${GREEN}✅ CMake version is sufficient (>= 3.20)${NC}"
    else
        echo -e "${RED}❌ CMake version too old. Please upgrade to 3.20+${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Error: CMake not found. Please install CMake 3.20+${NC}"
    exit 1
fi

# Check dependencies
echo -e "${BLUE}🔍 Checking dependencies...${NC}"

# Check Eigen3
if pkg-config --exists eigen3; then
    EIGEN_VERSION=$(pkg-config --modversion eigen3)
    echo -e "Found: ${GREEN}Eigen3 ${EIGEN_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠️  Eigen3 not found via pkg-config. Will try CMake find_package.${NC}"
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "Found: ${GREEN}${PYTHON_VERSION}${NC}"
    
    # Check pybind11
    if python3 -c "import pybind11" &> /dev/null; then
        echo -e "${GREEN}✅ pybind11 found${NC}"
    else
        echo -e "${YELLOW}⚠️  pybind11 not found. Checking virtual environment...${NC}"
        if [ -f ".venv/bin/python" ]; then
            echo -e "${BLUE}Using virtual environment...${NC}"
            .venv/bin/python -m pip install pybind11
        else
            echo -e "${YELLOW}⚠️  Virtual environment not found. Please create one with: python3 -m venv .venv${NC}"
            echo -e "${YELLOW}⚠️  Then activate it and install pybind11 manually${NC}"
        fi
    fi
else
    echo -e "${RED}❌ Error: Python3 not found${NC}"
    exit 1
fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" = "true" ]; then
    echo -e "${BLUE}🧹 Cleaning build directory...${NC}"
    rm -rf build/
fi

# Create build directory
mkdir -p build/${BUILD_TYPE}
cd build/${BUILD_TYPE}

# Configure with CMake
echo -e "${BLUE}⚙️  Configuring with CMake...${NC}"
cmake ../.. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_CUDA=${ENABLE_CUDA} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -std=c++23" \
    -DCMAKE_CXX_FLAGS_DEBUG="-g3 -O0 -fsanitize=address,undefined" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -flto -march=native"

# Build
echo -e "${BLUE}🔨 Building TDA Vector Stack...${NC}"
make -j$(nproc)

# Run tests if requested
if [ "$RUN_TESTS" = "true" ]; then
    echo -e "${BLUE}🧪 Running tests...${NC}"
    if make test; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
    else
        echo -e "${RED}❌ Some tests failed!${NC}"
        exit 1
    fi
fi

# Run benchmarks if requested
if [ "$RUN_BENCHMARKS" = "true" ]; then
    echo -e "${BLUE}📊 Running benchmarks...${NC}"
    if [ -f "bin/tda_benchmarks" ]; then
        ./bin/tda_benchmarks
    else
        echo -e "${YELLOW}⚠️  Benchmark executable not found${NC}"
    fi
fi

# Install
echo -e "${BLUE}📦 Installing TDA Vector Stack...${NC}"
make install

echo -e "${GREEN}🎉 TDA Vector Stack built successfully!${NC}"
echo -e "${BLUE}📁 Build artifacts are in: ${YELLOW}build/${BUILD_TYPE}${NC}"
echo -e "${BLUE}📚 Documentation: ${YELLOW}make docs${NC}"
echo -e "${BLUE}🧪 Run tests: ${YELLOW}make test${NC}"
echo -e "${BLUE}📊 Run benchmarks: ${YELLOW}./bin/tda_benchmarks${NC}"

# Return to project root
cd ../..

echo -e "${GREEN}✅ Build complete!${NC}"


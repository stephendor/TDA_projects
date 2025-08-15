#!/bin/bash

# Script to build the simplified vectorization and storage example

# Set up environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"
BUILD_DIR="${SCRIPT_DIR}/build/simplified"

echo "Building simplified vectorization and storage example..."
echo "Examples directory: ${EXAMPLES_DIR}"
echo "Build directory: ${BUILD_DIR}"

# Create build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || { echo "Failed to change to build directory"; exit 1; }

echo "Using CMakeLists.txt.simplified from examples directory"
# Build directly from the examples directory - no need to copy files
cmake "${EXAMPLES_DIR}" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CXX_FLAGS="-O2 -Wall -Wextra" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_PROJECT_INCLUDE="${EXAMPLES_DIR}/CMakeLists.txt.simplified"

echo "Building..."
make -j"$(nproc)"

if [ $? -eq 0 ]; then
    echo -e "\nBuild successful! Run the example with:"
    echo -e "${BUILD_DIR}/vectorization_example\n"
else
    echo -e "\nBuild failed. Please check the error messages above.\n"
fi

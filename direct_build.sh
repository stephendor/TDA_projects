#!/bin/bash

# A simple direct build script for the simplified vectorization example
# No CMake, just direct g++ compilation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="${SCRIPT_DIR}/examples/vectorization_storage_example_simplified.cpp"
OUTPUT_DIR="${SCRIPT_DIR}/build/direct"
OUTPUT_FILE="${OUTPUT_DIR}/vectorization_example"

echo "Building simplified vectorization example using direct compilation..."
echo "Source file: ${SOURCE_FILE}"
echo "Output file: ${OUTPUT_FILE}"

# Create build directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Compile directly with g++
echo "Compiling with g++..."
g++ -std=c++17 -O2 -Wall -Wextra "${SOURCE_FILE}" -o "${OUTPUT_FILE}"

# Check compilation status
if [ $? -eq 0 ]; then
    echo -e "\nBuild successful! Run the example with:"
    echo -e "${OUTPUT_FILE}\n"
else
    echo -e "\nBuild failed. Please check the error messages above.\n"
fi

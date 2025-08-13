#!/bin/bash

echo "Starting basic test..."

# Test 1: Simple command
echo "Test 1: Checking cmake"
if command -v cmake &> /dev/null; then
    echo "✅ cmake found"
else
    echo "❌ cmake not found"
fi

# Test 2: Version check
echo "Test 2: Checking cmake version"
cmake_version=$(cmake --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1)
echo "cmake version: $cmake_version"

# Test 3: Math check
echo "Test 3: Simple math"
major_version=$(echo "$cmake_version" | cut -d. -f1)
minor_version=$(echo "$cmake_version" | cut -d. -f2)
echo "Major: $major_version, Minor: $minor_version"

# Test 4: Comparison
echo "Test 4: Version comparison"
if [ "$major_version" -gt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -ge 20 ]); then
    echo "✅ CMake version meets requirement (>= 3.20)"
else
    echo "❌ CMake version too old (requires >= 3.20)"
fi

echo "Basic test completed!"


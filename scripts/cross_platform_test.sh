#!/bin/bash

# TDA Platform Cross-Platform Compatibility Testing Script
# Simple version for basic compatibility validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Logging
log_section() { echo -e "\n🔍 $1"; }
log_success() { echo -e "✅ $1"; }
log_warning() { echo -e "⚠️  $1"; }
log_error() { echo -e "❌ $1"; }

# Main tests
main() {
    echo -e "\n🌐 TDA Platform Cross-Platform Compatibility Testing"
    echo -e "===================================================="
    
    # Test compilers
    log_section "Testing Compiler Compatibility"
    for compiler in "g++" "clang++"; do
        if command -v "$compiler" &> /dev/null; then
            local version=$("$compiler" --version | head -1)
            log_success "Found: $compiler - $version"
            
            if "$compiler" -std=c++23 -E -x c++ /dev/null >/dev/null 2>&1; then
                log_success "  C++23 support: ✅"
            else
                log_warning "  C++23 support: ❌"
            fi
        else
            log_warning "Not found: $compiler"
        fi
    done
    
    # Test libraries
    log_section "Testing Library Compatibility"
    local libraries=(
        "eigen3:pkg-config --exists eigen3"
        "tbb:pkg-config --exists tbb"
        "gudhi:test -f /usr/include/gudhi/Simplex_tree.h"
        "openmp:gcc -fopenmp -E -x c /dev/null"
        "pybind11:python3 -c 'import pybind11'"
    )
    
    for lib_config in "${libraries[@]}"; do
        IFS=':' read -r lib_name test_cmd <<< "$lib_config"
        if eval "$test_cmd" >/dev/null 2>&1; then
            log_success "$lib_name: ✅"
        else
            log_warning "$lib_name: ❌"
        fi
    done
    
    # Test build system
    log_section "Testing Build System Compatibility"
    for tool in "cmake" "make" "ninja"; do
        if command -v "$tool" &> /dev/null; then
            local version=$("$tool" --version | head -1)
            log_success "$tool: ✅ - $version"
        else
            log_warning "$tool: ❌"
        fi
    done
    
    # Test CMake version
    if command -v cmake &> /dev/null; then
        local cmake_version=$(cmake --version | head -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        if [[ "$cmake_version" =~ ^3\.[2-9][0-9]$ ]] || [[ "$cmake_version" =~ ^[4-9]\. ]]; then
            log_success "CMake version requirement met: $cmake_version >= 3.20"
        else
            log_warning "CMake version requirement not met: $cmake_version < 3.20"
        fi
    fi
    
    log_success "Compatibility testing completed"
}

main "$@"

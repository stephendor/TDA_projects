#!/bin/bash

# Simplified TDA Platform Dependency Validation Script
# Basic validation without complex logging

# set -e  # Temporarily disabled for debugging

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 TDA Platform Dependency Validation (Simple)${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Initialize counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Simple logging functions
log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
    ((TOTAL_CHECKS++))
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_section() {
    echo -e "${BLUE}📋 $1${NC}"
    echo "--------------------------------------------------"
}

# Check command availability
check_command() {
    local cmd="$1"
    local name="${2:-$1}"
    local required="${3:-true}"
    
    if command -v "$cmd" &> /dev/null; then
        local version
        version=$("$cmd" --version 2>/dev/null | head -n1 || echo "version unknown")
        log_success "$name found: $version"
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "$name not found - REQUIRED"
            return 1
        else
            log_warning "$name not found - OPTIONAL"
            return 0
        fi
    fi
}

# Check core dependencies
check_core_dependencies() {
    log_section "Core Dependencies"
    
    check_command "cmake" "CMake" true
    check_command "make" "Make" true
    check_command "g++" "GCC C++ Compiler" true
    check_command "python3" "Python 3" false
    
    # Check CMake version
    if command -v cmake &> /dev/null; then
        local cmake_version
        cmake_version=$(cmake --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1)
        local major_version
        major_version=$(echo "$cmake_version" | cut -d. -f1)
        local minor_version
        minor_version=$(echo "$cmake_version" | cut -d. -f2)
        
        if [ "$major_version" -gt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -ge 20 ]); then
            log_success "CMake version $cmake_version meets requirement (>= 3.20)"
        else
            log_error "CMake version $cmake_version too old (requires >= 3.20)"
            return 1
        fi
    fi
}

# Check compiler capabilities
check_compiler_capabilities() {
    log_section "Compiler Capabilities"
    
    # Check C++23 support
    if g++ -std=c++23 -E -x c++ /dev/null &> /dev/null; then
        log_success "C++23 standard supported"
    else
        log_error "C++23 standard not supported - REQUIRED"
        return 1
    fi
    
    # Check SIMD support
    local simd_supported=()
    
    for flag in "-mavx2" "-mfma" "-msse4.2"; do
        if g++ "$flag" -E -x c++ /dev/null &> /dev/null; then
            simd_supported+=("$flag")
        fi
    done
    
    if [ ${#simd_supported[@]} -gt 0 ]; then
        log_success "SIMD support: ${simd_supported[*]}"
    else
        log_warning "No SIMD support detected"
    fi
}

# Check mathematical libraries
check_math_libraries() {
    log_section "Mathematical Libraries"
    
    # Check Eigen3
    local eigen_found=false
    
    # Method 1: pkg-config
    if pkg-config --exists eigen3 2>/dev/null; then
        local eigen_version
        eigen_version=$(pkg-config --modversion eigen3)
        log_success "Eigen3 found via pkg-config: $eigen_version"
        eigen_found=true
    fi
    
    # Method 2: Manual search
    if [ "$eigen_found" = false ]; then
        local eigen_paths=("/usr/include/eigen3" "/usr/local/include/eigen3" "/opt/eigen3/include")
        
        for path in "${eigen_paths[@]}"; do
            if [ -d "$path" ] && [ -f "$path/Eigen/Core" ]; then
                log_success "Eigen3 found in: $path"
                eigen_found=true
                break
            fi
        done
    fi
    
    if [ "$eigen_found" = false ]; then
        log_error "Eigen3 not found - REQUIRED"
        return 1
    fi
    
    # Check GUDHI
    local gudhi_found=false
    local gudhi_paths=("/usr/include" "/usr/local/include" "/opt/gudhi/include")
    
    for path in "${gudhi_paths[@]}"; do
        if [ -f "$path/gudhi/Simplex_tree.h" ]; then
            log_success "GUDHI found in: $path"
            gudhi_found=true
            break
        fi
    done
    
    if [ "$gudhi_found" = false ]; then
        log_error "GUDHI not found - REQUIRED"
        return 1
    fi
    
    # Check TBB
    local tbb_found=false
    local tbb_paths=("/usr/include/tbb" "/usr/local/include/tbb" "/opt/intel/oneapi/tbb/latest/include/tbb")
    
    for path in "${tbb_paths[@]}"; do
        if [ -d "$path" ] && [ -f "$path/tbb.h" ]; then
            log_success "TBB found in: $path"
            tbb_found=true
            break
        fi
    done
    
    if [ "$tbb_found" = false ]; then
        log_error "TBB not found - REQUIRED"
        return 1
    fi
    
    # Check OpenMP
    if g++ -fopenmp -E -x c++ /dev/null &> /dev/null; then
        log_success "OpenMP support available"
    else
        log_warning "OpenMP support not available"
    fi
}

# Generate summary
generate_summary() {
    echo ""
    echo -e "${BLUE}📊 VALIDATION SUMMARY${NC}"
    echo "=================================================="
    echo -e "Total checks: ${TOTAL_CHECKS}"
    echo -e "Passed: ${GREEN}${PASSED_CHECKS}${NC}"
    echo -e "Failed: ${RED}${FAILED_CHECKS}${NC}"
    echo -e "Warnings: ${YELLOW}${WARNINGS}${NC}"
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        echo ""
        echo -e "${GREEN}🎉 All required dependencies are available!${NC}"
        echo -e "${GREEN}Your system is ready to build the TDA Platform.${NC}"
        echo -e "${BLUE}Run: ./build.sh release${NC}"
    else
        echo ""
        echo -e "${RED}❌ $FAILED_CHECKS required dependencies are missing!${NC}"
        echo -e "${YELLOW}Please install the missing dependencies before building.${NC}"
    fi
}

# Main execution
main() {
    # Run all checks
    check_core_dependencies
    check_compiler_capabilities
    check_math_libraries
    
    # Generate summary
    generate_summary
    
    # Exit with appropriate code
    if [ $FAILED_CHECKS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"

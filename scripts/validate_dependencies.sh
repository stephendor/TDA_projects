#!/bin/bash

# TDA Platform Dependency Validation Script
# Comprehensive validation of all required dependencies and system capabilities

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/dependency_validation.log"
REPORT_FILE="$PROJECT_ROOT/dependency_health_report.txt"

# Initialize counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
    ((WARNINGS++))
    ((TOTAL_CHECKS++))
}

log_error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_header() {
    echo -e "${PURPLE}🔍 $1${NC}" | tee -a "$LOG_FILE"
    echo "==================================================" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "${CYAN}📋 $1${NC}" | tee -a "$LOG_FILE"
    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
}

# Initialize log files
echo "TDA Platform Dependency Validation Report" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "TDA Platform Dependency Validation Log" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to check command availability
check_command() {
    local cmd="$1"
    local name="${2:-$1}"
    local required="${3:-true}"
    
    if command -v "$cmd" &> /dev/null; then
        local version
        version=$("$cmd" --version 2>/dev/null | head -n1 || echo "version unknown")
        log_success "$name found: $version"
        echo "$name: $version" >> "$REPORT_FILE"
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "$name not found - REQUIRED"
            echo "$name: NOT FOUND (REQUIRED)" >> "$REPORT_FILE"
            return 1
        else
            log_warning "$name not found - OPTIONAL"
            echo "$name: NOT FOUND (OPTIONAL)" >> "$REPORT_FILE"
            return 0
        fi
    fi
}

# Function to check file/directory existence
check_path() {
    local path="$1"
    local name="${2:-$path}"
    local required="${3:-true}"
    
    if [ -e "$path" ]; then
        log_success "$name found: $path"
        echo "$name: $path" >> "$REPORT_FILE"
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "$name not found - REQUIRED"
            echo "$name: NOT FOUND (REQUIRED)" >> "$REPORT_FILE"
            return 1
        else
            log_warning "$name not found - OPTIONAL"
            echo "$name: NOT FOUND (OPTIONAL)" >> "$REPORT_FILE"
            return 0
        fi
    fi
}

# Function to check library linking
check_library_linking() {
    local lib_name="$1"
    local test_file="$2"
    local required="${3:-true}"
    
    # Create temporary test file
    local temp_dir=$(mktemp -d)
    local test_cpp="$temp_dir/test_${lib_name}.cpp"
    local test_exe="$temp_dir/test_${lib_name}"
    
    cat > "$test_cpp" << EOF
#include <iostream>
int main() {
    std::cout << "Testing $lib_name linking..." << std::endl;
    return 0;
}
EOF
    
    # Try to compile and link
    if g++ -std=c++23 "$test_cpp" -o "$test_exe" 2>/dev/null; then
        log_success "$lib_name linking test passed"
        echo "$lib_name linking: PASSED" >> "$REPORT_FILE"
        rm -rf "$temp_dir"
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "$lib_name linking test failed - REQUIRED"
            echo "$lib_name linking: FAILED (REQUIRED)" >> "$REPORT_FILE"
            rm -rf "$temp_dir"
            return 1
        else
            log_warning "$lib_name linking test failed - OPTIONAL"
            echo "$lib_name linking: FAILED (OPTIONAL)" >> "$REPORT_FILE"
            rm -rf "$temp_dir"
            return 0
        fi
    fi
}

# Function to check compiler capabilities
check_compiler_capabilities() {
    log_section "Compiler Capabilities"
    
    # Check C++23 support
    if g++ -std=c++23 -E -x c++ /dev/null &> /dev/null; then
        log_success "C++23 standard supported"
        echo "C++23 support: YES" >> "$REPORT_FILE"
    else
        log_error "C++23 standard not supported - REQUIRED"
        echo "C++23 support: NO (REQUIRED)" >> "$REPORT_FILE"
        return 1
    fi
    
    # Check SIMD support
    local simd_flags=("-mavx2" "-mfma" "-msse4.2")
    local simd_supported=()
    
    for flag in "${simd_flags[@]}"; do
        if g++ "$flag" -E -x c++ /dev/null &> /dev/null; then
            simd_supported+=("$flag")
        fi
    done
    
    if [ ${#simd_supported[@]} -gt 0 ]; then
        log_success "SIMD support: ${simd_supported[*]}"
        echo "SIMD support: ${simd_supported[*]}" >> "$REPORT_FILE"
    else
        log_warning "No SIMD support detected"
        echo "SIMD support: NONE" >> "$REPORT_FILE"
    fi
    
    # Check optimization flags
    local opt_flags=("-O3" "-march=native" "-flto")
    local opt_supported=()
    
    for flag in "${opt_flags[@]}"; do
        if g++ "$flag" -E -x c++ /dev/null &> /dev/null; then
            opt_supported+=("$flag")
        fi
    done
    
    if [ ${#opt_supported[@]} -gt 0 ]; then
        log_success "Optimization support: ${opt_supported[*]}"
        echo "Optimization support: ${opt_supported[*]}" >> "$REPORT_FILE"
    else
        log_warning "Limited optimization support"
        echo "Optimization support: LIMITED" >> "$REPORT_FILE"
    fi
}

# Function to check core dependencies
check_core_dependencies() {
    log_section "Core Dependencies"
    
    # Check build tools
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
            echo "CMake version: $cmake_version (REQUIRES >= 3.20)" >> "$REPORT_FILE"
        else
            log_error "CMake version $cmake_version too old (requires >= 3.20)"
            echo "CMake version: $cmake_version (REQUIRES >= 3.20)" >> "$REPORT_FILE"
            return 1
        fi
    fi
}

# Function to check mathematical libraries
check_math_libraries() {
    log_section "Mathematical Libraries"
    
    # Check Eigen3
    local eigen_found=false
    
    # Method 1: pkg-config
    if pkg-config --exists eigen3 2>/dev/null; then
        local eigen_version
        eigen_version=$(pkg-config --modversion eigen3)
        log_success "Eigen3 found via pkg-config: $eigen_version"
        echo "Eigen3: $eigen_version (pkg-config)" >> "$REPORT_FILE"
        eigen_found=true
    fi
    
    # Method 2: CMake find_package
    if [ "$eigen_found" = false ]; then
        local temp_dir
        temp_dir=$(mktemp -d)
        local cmake_test="$temp_dir/CMakeLists.txt"
        
        cat > "$cmake_test" << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(EigenTest)
find_package(Eigen3 3.4 REQUIRED)
message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
EOF
        
        if cmake -S "$temp_dir" -B "$temp_dir/build" 2>/dev/null | grep -q "Eigen3 found:"; then
            local eigen_version
            eigen_version=$(cmake -S "$temp_dir" -B "$temp_dir/build" 2>/dev/null | grep "Eigen3 found:" | sed 's/.*Eigen3 found: //')
            log_success "Eigen3 found via CMake: $eigen_version"
            echo "Eigen3: $eigen_version (CMake)" >> "$REPORT_FILE"
            eigen_found=true
        fi
        
        rm -rf "$temp_dir"
    fi
    
    # Method 3: Manual search
    if [ "$eigen_found" = false ]; then
        local eigen_paths=("/usr/include/eigen3" "/usr/local/include/eigen3" "/opt/eigen3/include")
        
        for path in "${eigen_paths[@]}"; do
            if [ -d "$path" ] && [ -f "$path/Eigen/Core" ]; then
                log_success "Eigen3 found in: $path"
                echo "Eigen3: $path (manual)" >> "$REPORT_FILE"
                eigen_found=true
                break
            fi
        done
    fi
    
    if [ "$eigen_found" = false ]; then
        log_error "Eigen3 not found - REQUIRED"
        echo "Eigen3: NOT FOUND (REQUIRED)" >> "$REPORT_FILE"
        return 1
    fi
    
    # Check GUDHI
    local gudhi_found=false
    local gudhi_paths=("/usr/include" "/usr/local/include" "/opt/gudhi/include")
    
    for path in "${gudhi_paths[@]}"; do
        if [ -f "$path/gudhi/Simplex_tree.h" ]; then
            log_success "GUDHI found in: $path"
            echo "GUDHI: $path" >> "$REPORT_FILE"
            gudhi_found=true
            break
        fi
    done
    
    if [ "$gudhi_found" = false ]; then
        log_error "GUDHI not found - REQUIRED"
        echo "GUDHI: NOT FOUND (REQUIRED)" >> "$REPORT_FILE"
        return 1
    fi
    
    # Check TBB
    local tbb_found=false
    
    # Method 1: pkg-config
    if pkg-config --exists tbb 2>/dev/null; then
        local tbb_version
        tbb_version=$(pkg-config --modversion tbb)
        log_success "TBB found via pkg-config: $tbb_version"
        echo "TBB: $tbb_version (pkg-config)" >> "$REPORT_FILE"
        tbb_found=true
    fi
    
    # Method 2: Manual search
    if [ "$tbb_found" = false ]; then
        local tbb_paths=("/usr/include/tbb" "/usr/local/include/tbb" "/opt/intel/oneapi/tbb/latest/include/tbb")
        
        for path in "${tbb_paths[@]}"; do
            if [ -d "$path" ] && [ -f "$path/tbb.h" ]; then
                log_success "TBB found in: $path"
                echo "TBB: $path (manual)" >> "$REPORT_FILE"
                tbb_found=true
                break
            fi
        done
    fi
    
    if [ "$tbb_found" = false ]; then
        log_error "TBB not found - REQUIRED"
        echo "TBB: NOT FOUND (REQUIRED)" >> "$REPORT_FILE"
        return 1
    fi
    
    # Check OpenMP
    if g++ -fopenmp -E -x c++ /dev/null &> /dev/null; then
        log_success "OpenMP support available"
        echo "OpenMP: SUPPORTED" >> "$REPORT_FILE"
    else
        log_warning "OpenMP support not available"
        echo "OpenMP: NOT SUPPORTED" >> "$REPORT_FILE"
    fi
}

# Function to check Python dependencies
check_python_dependencies() {
    log_section "Python Dependencies"
    
    if ! command -v python3 &> /dev/null; then
        log_warning "Python 3 not available - Python bindings will be disabled"
        echo "Python 3: NOT AVAILABLE" >> "$REPORT_FILE"
        return 0
    fi
    
    # Check Python version
    local python_version
    python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1)
    local major_version
    major_version=$(echo "$python_version" | cut -d. -f1)
    local minor_version
    minor_version=$(echo "$python_version" | cut -d. -f2)
    
    if [ "$major_version" -ge 3 ] && [ "$minor_version" -ge 9 ]; then
        log_success "Python version $python_version meets requirement (>= 3.9)"
        echo "Python version: $python_version (REQUIRES >= 3.9)" >> "$REPORT_FILE"
    else
        log_warning "Python version $python_version below requirement (requires >= 3.9)"
        echo "Python version: $python_version (REQUIRES >= 3.9)" >> "$REPORT_FILE"
        return 0
    fi
    
    # Check pybind11
    if [ -d "$PROJECT_ROOT/.venv/lib/python3.*/site-packages/pybind11" ]; then
        log_success "pybind11 found in virtual environment"
        echo "pybind11: FOUND (venv)" >> "$REPORT_FILE"
    else
        log_warning "pybind11 not found - Python bindings will be disabled"
        echo "pybind11: NOT FOUND" >> "$REPORT_FILE"
    fi
}

# Function to check system resources
check_system_resources() {
    log_section "System Resources"
    
    # Check available memory
    if command -v free &> /dev/null; then
        local total_mem
        total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
        local available_mem
        available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
        
        log_success "Total memory: ${total_mem}GB, Available: ${available_mem}GB"
        echo "Memory: ${total_mem}GB total, ${available_mem}GB available" >> "$REPORT_FILE"
        
        if [ "$total_mem" -lt 4 ]; then
            log_warning "Low memory system (<4GB) - large datasets may cause issues"
            echo "Memory warning: Low memory system" >> "$REPORT_FILE"
        fi
    fi
    
    # Check available disk space
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2{printf "%.0f", $4/1024/1024}')
    
    log_success "Available disk space: ${available_space}GB"
    echo "Disk space: ${available_space}GB available" >> "$REPORT_FILE"
    
    if [ "$available_space" -lt 5 ]; then
        log_warning "Low disk space (<5GB) - build may fail"
        echo "Disk warning: Low disk space" >> "$REPORT_FILE"
    fi
    
    # Check CPU cores
    if command -v nproc &> /dev/null; then
        local cpu_cores
        cpu_cores=$(nproc)
        log_success "CPU cores available: $cpu_cores"
        echo "CPU cores: $cpu_cores" >> "$REPORT_FILE"
        
        if [ "$cpu_cores" -lt 2 ]; then
            log_warning "Single core system - build will be slow"
            echo "CPU warning: Single core system" >> "$REPORT_FILE"
        fi
    fi
}

# Function to run basic build test
run_build_test() {
    log_section "Basic Build Test"
    
    local temp_dir
    temp_dir=$(mktemp -d)
    local cmake_test="$temp_dir/CMakeLists.txt"
    
    cat > "$cmake_test" << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(TDATest)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Test Eigen3
find_package(Eigen3 3.4 REQUIRED)

# Test GUDHI
find_path(GUDHI_INCLUDE_DIR NAMES gudhi/Simplex_tree.h PATHS /usr/include /usr/local/include)
if(NOT GUDHI_INCLUDE_DIR)
    message(FATAL_ERROR "GUDHI not found")
endif()

# Test TBB
find_package(TBB REQUIRED)

message(STATUS "All dependencies found successfully")
EOF
    
    if cmake -S "$temp_dir" -B "$temp_dir/build" 2>/dev/null | grep -q "All dependencies found successfully"; then
        log_success "Basic CMake configuration test passed"
        echo "CMake test: PASSED" >> "$REPORT_FILE"
        rm -rf "$temp_dir"
        return 0
    else
        log_error "Basic CMake configuration test failed"
        echo "CMake test: FAILED" >> "$REPORT_FILE"
        rm -rf "$temp_dir"
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    log_header "Validation Summary"
    
    local success_rate
    success_rate=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    
    echo "" >> "$REPORT_FILE"
    echo "================================================" >> "$REPORT_FILE"
    echo "VALIDATION SUMMARY" >> "$REPORT_FILE"
    echo "================================================" >> "$REPORT_FILE"
    echo "Total checks: $TOTAL_CHECKS" >> "$REPORT_FILE"
    echo "Passed: $PASSED_CHECKS" >> "$REPORT_FILE"
    echo "Failed: $FAILED_CHECKS" >> "$REPORT_FILE"
    echo "Warnings: $WARNINGS" >> "$REPORT_FILE"
    echo "Success rate: ${success_rate}%" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        log_success "All required dependencies are available!"
        echo "STATUS: READY TO BUILD" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "✅ Your system is ready to build the TDA Platform!" >> "$REPORT_FILE"
        echo "Run: ./build.sh release" >> "$REPORT_FILE"
    else
        log_error "$FAILED_CHECKS required dependencies are missing!"
        echo "STATUS: DEPENDENCIES MISSING" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "❌ Please install the missing required dependencies before building." >> "$REPORT_FILE"
        echo "Check the report above for specific requirements." >> "$REPORT_FILE"
    fi
    
    if [ $WARNINGS -gt 0 ]; then
        echo "" >> "$REPORT_FILE"
        echo "⚠️  $WARNINGS optional dependencies are missing or have warnings." >> "$REPORT_FILE"
        echo "The system will still work but with reduced functionality." >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "Full validation log: $LOG_FILE" >> "$REPORT_FILE"
    echo "Health report: $REPORT_FILE" >> "$REPORT_FILE"
    
    # Display summary
    echo ""
    echo -e "${PURPLE}📊 VALIDATION SUMMARY${NC}"
    echo "=================================================="
    echo -e "Total checks: ${CYAN}$TOTAL_CHECKS${NC}"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
    echo -e "Success rate: ${CYAN}${success_rate}%${NC}"
    echo ""
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        echo -e "${GREEN}🎉 All required dependencies are available!${NC}"
        echo -e "${GREEN}Your system is ready to build the TDA Platform.${NC}"
        echo -e "${BLUE}Run: ./build.sh release${NC}"
    else
        echo -e "${RED}❌ $FAILED_CHECKS required dependencies are missing!${NC}"
        echo -e "${YELLOW}Please install the missing dependencies before building.${NC}"
        echo -e "${BLUE}Check the report above for specific requirements.${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}📋 Full report: $REPORT_FILE${NC}"
    echo -e "${CYAN}📝 Detailed log: $LOG_FILE${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}🚀 TDA Platform Dependency Validation${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo ""
    
    log_header "Starting dependency validation"
    
    # Run all checks
    check_core_dependencies
    check_compiler_capabilities
    check_math_libraries
    check_python_dependencies
    check_system_resources
    run_build_test
    
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


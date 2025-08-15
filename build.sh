#!/bin/bash

# Enhanced TDA Vector Stack Build Script with Comprehensive Error Handling
# This script provides robust error handling, recovery mechanisms, and graceful degradation

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global error tracking
BUILD_ERRORS=0
BUILD_WARNINGS=0
BUILD_START_TIME=$(date +%s)
BUILD_TIMEOUT=1800  # 30 minutes timeout
MEMORY_LIMIT_MB=8192  # 8GB memory limit

# Configuration
BUILD_TYPE=${1:-release}
ENABLE_CUDA=${2:-OFF}
CLEAN_BUILD=${3:-false}
RUN_TESTS=${4:-false}
RUN_BENCHMARKS=${5:-false}
SKIP_VALIDATION=${6:-false}

# Logging functions
log_header() {
    echo -e "\n${BLUE}🚀 TDA Vector Stack Build Script (Enhanced)${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "Build Type: ${BUILD_TYPE}"
    echo -e "CUDA Support: ${ENABLE_CUDA}"
    echo -e "Clean Build: ${CLEAN_BUILD}"
    echo -e "Run Tests: ${RUN_TESTS}"
    echo -e "Run Benchmarks: ${RUN_BENCHMARKS}"
    echo -e "Skip Validation: ${SKIP_VALIDATION}\n"
}

log_section() {
    echo -e "\n${CYAN}🔍 $1${NC}"
    echo -e "${CYAN}==================================================${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((BUILD_WARNINGS++))
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((BUILD_ERRORS++))
}

log_success_section() {
    echo -e "${GREEN}🎉 $1${NC}"
}

# Error handling functions
handle_build_error() {
    local error_type="$1"
    local exit_code="$2"
    local details="$3"
    
    log_error "Build failed: $error_type"
    echo -e "\n${RED}🔍 Build Error Report${NC}"
    echo -e "${RED}==================================================${NC}"
    echo -e "Error: $error_type"
    echo -e "Exit code: $exit_code"
    echo -e "Build type: $BUILD_TYPE"
    echo -e "Timestamp: $(date)"
    echo -e "Duration: $(( $(date +%s) - BUILD_START_TIME )) seconds"
    
    if [[ -n "$details" ]]; then
        echo -e "\nDetails: $details"
    fi
    
    # Provide recovery suggestions
    case "$error_type" in
        "Build compilation failed")
            echo -e "\n${YELLOW}💡 Recovery Suggestions:${NC}"
            echo -e "  • Check compiler errors above"
            echo -e "  • Verify all dependencies are installed"
            echo -e "  • Try cleaning build directory: ./build.sh $BUILD_TYPE $ENABLE_CUDA true false false false"
            echo -e "  • Check for memory issues: free -h"
            ;;
        "Test execution failed")
            echo -e "\n${YELLOW}💡 Recovery Suggestions:${NC}"
            echo -e "  • Tests may be hitting memory limits"
            echo -e "  • Try building without tests: ./build.sh $BUILD_TYPE $ENABLE_CUDA false false false false"
            echo -e "  • Check system memory: free -h"
            echo -e "  • Consider reducing test data size"
            ;;
        "Benchmark execution failed")
            echo -e "\n${YELLOW}💡 Recovery Suggestions:${NC}"
            echo -e "  • Benchmarks may be hitting memory limits"
            echo -e "  • Try building without benchmarks: ./build.sh $BUILD_TYPE $ENABLE_CUDA false false false false"
            echo -e "  • Check system resources: htop"
            ;;
        "Installation failed")
            echo -e "\n${YELLOW}💡 Recovery Suggestions:${NC}"
            echo -e "  • Permission denied - try: sudo ./build.sh $BUILD_TYPE $ENABLE_CUDA false false false false"
            echo -e "  • Or skip installation and use local build"
            ;;
        *)
            echo -e "\n${YELLOW}💡 General Recovery:${NC}"
            echo -e "  • Check the error details above"
            echo -e "  • Verify system resources"
            echo -e "  • Try a clean build"
            ;;
    esac
}

# Memory monitoring
check_memory() {
    local available_mb=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_mb -lt $MEMORY_LIMIT_MB ]]; then
        log_warning "Low memory available: ${available_mb}MB (recommended: ${MEMORY_LIMIT_MB}MB)"
        return 1
    fi
    return 0
}

# Timeout monitoring
check_timeout() {
    local elapsed=$(( $(date +%s) - BUILD_START_TIME ))
    if [[ $elapsed -gt $BUILD_TIMEOUT ]]; then
        log_error "Build timeout exceeded: ${elapsed}s > ${BUILD_TIMEOUT}s"
        return 1
    fi
    return 0
}

# Dependency validation
run_dependency_validation() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log_info "Skipping dependency validation as requested"
        return 0
    fi
    
    log_section "Dependency Validation"
    log_info "Running comprehensive dependency validation..."
    
    if [[ -f "scripts/validate_dependencies_simple.sh" ]]; then
        if ./scripts/validate_dependencies_simple.sh; then
            log_success "All dependencies validated successfully!"
            return 0
        else
            log_warning "Dependency validation had issues, but continuing..."
            return 0
        fi
    else
        log_warning "Dependency validation script not found, skipping..."
        return 0
    fi
}

# Build directory management
clean_build_directory() {
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        log_info "Cleaning build directory..."
        if [[ -d "build" ]]; then
            rm -rf build
            log_success "Build directory cleaned"
        else
            log_info "Build directory does not exist, nothing to clean"
        fi
    fi
}

# CMake configuration
configure_build() {
    log_info "Configuring with CMake..."
    
    # Create build directory
    mkdir -p "build/$BUILD_TYPE"
    cd "build/$BUILD_TYPE"
    
    # Configure CMake with error handling
    local cmake_config=(
        "../.."
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DENABLE_CUDA=$ENABLE_CUDA"
    )
    
    if ! cmake "${cmake_config[@]}" 2>&1; then
        cd ../..
        handle_build_error "CMake configuration failed" "$?" "CMake configuration step failed"
        return 1
    fi
    
    log_success "CMake configuration completed"
    cd ../..
    return 0
}

# Build compilation
build_project() {
    log_info "Building TDA Vector Stack..."
    
    cd "build/$BUILD_TYPE"
    
    # Get number of CPU cores for parallel build
    local num_cores=$(nproc)
    log_info "Using $num_cores parallel jobs"
    
    # Build with timeout and memory monitoring
    local build_start=$(date +%s)
    
    if ! timeout $((BUILD_TIMEOUT - (build_start - BUILD_START_TIME))) make -j$num_cores 2>&1; then
        local exit_code=$?
        cd ../..
        
        if [[ $exit_code -eq 124 ]]; then
            handle_build_error "Build timeout exceeded" "$exit_code" "Build took longer than ${BUILD_TIMEOUT} seconds"
        else
            handle_build_error "Build compilation failed" "$exit_code" "Make build failed with exit code $exit_code"
        fi
        return 1
    fi
    
    log_success "Build compilation completed"
    cd ../..
    return 0
}

# Test execution with safety measures
run_tests() {
    if [[ "$RUN_TESTS" != "true" ]]; then
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "build/$BUILD_TYPE"
    
    # Check if tests exist
    if ! make test 2>/dev/null; then
        log_warning "No tests found or test target failed"
        cd ../..
        return 0
    fi
    
    # Run tests with timeout and memory monitoring
    local test_timeout=300  # 5 minutes for tests
    
    if ! timeout $test_timeout make test 2>&1; then
        local exit_code=$?
        cd ../..
        
        if [[ $exit_code -eq 124 ]]; then
            log_warning "Tests timed out after ${test_timeout} seconds"
            handle_build_error "Test execution failed" "$exit_code" "Tests exceeded time limit"
        else
            log_warning "Some tests failed, but continuing..."
        fi
        return 0
    fi
    
    log_success "All tests passed"
    cd ../..
    return 0
}

# Benchmark execution with safety measures
run_benchmarks() {
    if [[ "$RUN_BENCHMARKS" != "true" ]]; then
        return 0
    fi
    
    log_info "Running benchmarks..."
    
    cd "build/$BUILD_TYPE"
    
    # Check for benchmark executables
    local benchmark_found=false
    for benchmark in tda_benchmarks test_performance_benchmarks controlled_performance_test; do
        if [[ -f "bin/$benchmark" ]]; then
            benchmark_found=true
            break
        fi
    done
    
    if [[ "$benchmark_found" == "false" ]]; then
        log_warning "Benchmark executable not found"
        cd ../..
        return 0
    fi
    
    # Run benchmarks with timeout
    local benchmark_timeout=600  # 10 minutes for benchmarks
    
    if ! timeout $benchmark_timeout ./bin/controlled_performance_test 2>&1; then
        local exit_code=$?
        cd ../..
        
        if [[ $exit_code -eq 124 ]]; then
            log_warning "Benchmarks timed out after ${benchmark_timeout} seconds"
        else
            log_warning "Benchmarks failed, but continuing..."
        fi
        return 0
    fi
    
    log_success "Benchmarks completed"
    cd ../..
    return 0
}

# Installation with graceful fallback
install_project() {
    log_info "Installing TDA Vector Stack..."
    
    cd "build/$BUILD_TYPE"
    
    # Try installation, but don't fail the build if it fails
    if make install 2>&1; then
        log_success "Installation completed successfully"
    else
        local exit_code=$?
        log_warning "Installation failed. You may need to run with sudo or check permissions."
        log_info "Build artifacts are available in build/$BUILD_TYPE/bin/ and build/$BUILD_TYPE/lib/"
    fi
    
    cd ../..
    return 0
}

# Build summary
generate_build_summary() {
    local build_end_time=$(date +%s)
    local duration=$((build_end_time - BUILD_START_TIME))
    
    echo -e "\n${BLUE}🔍 Build Summary${NC}"
    echo -e "${BLUE}==================================================${NC}"
    echo -e "Build type: $BUILD_TYPE"
    echo -e "Duration: $duration seconds"
    echo -e "Build directory: build/$BUILD_TYPE"
    
    if [[ $BUILD_ERRORS -eq 0 ]]; then
        echo -e "${GREEN}✅ Build completed successfully!${NC}"
        
        echo -e "\n${GREEN}📚 Next steps:${NC}"
        echo -e "  • Documentation: make docs"
        echo -e "  • Run tests: make test"
        echo -e "  • Run benchmarks: ./bin/tda_benchmarks"
        echo -e "  • Install: make install"
        
        if [[ $BUILD_WARNINGS -gt 0 ]]; then
            echo -e "\n${YELLOW}⚠️  Build completed with $BUILD_WARNINGS warning(s):${NC}"
            # List specific warnings here if needed
        fi
    else
        echo -e "${RED}❌ Build failed with $BUILD_ERRORS error(s)${NC}"
        echo -e "\n${YELLOW}💡 Check the error details above for recovery suggestions${NC}"
    fi
}

# Main build orchestration
main() {
    log_header
    
    # Check system resources
    if ! check_memory; then
        log_warning "Continuing with low memory, but may encounter issues"
    fi
    
    # Run dependency validation
    if ! run_dependency_validation; then
        handle_build_error "Dependency validation failed" "$?" "Critical dependencies missing"
        exit 1
    fi
    
    # Clean build directory if requested
    clean_build_directory
    
    # Configure build
    if ! configure_build; then
        exit 1
    fi
    
    # Build project
    if ! build_project; then
        exit 1
    fi
    
    # Run tests if requested
    if ! run_tests; then
        # Tests failed but don't stop the build
        log_warning "Test execution had issues, but build completed"
    fi
    
    # Run benchmarks if requested
    if ! run_benchmarks; then
        # Benchmarks failed but don't stop the build
        log_warning "Benchmark execution had issues, but build completed"
    fi
    
    # Install project
    install_project
    
    # Generate build summary
    generate_build_summary
    
    # Exit with appropriate code
    if [[ $BUILD_ERRORS -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Trap for cleanup and error handling
trap 'echo -e "\n${RED}Build interrupted by user${NC}"; exit 130' INT TERM

# Run main function
main "$@"


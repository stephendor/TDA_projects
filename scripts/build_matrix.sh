#!/bin/bash

# TDA Platform Build Matrix Testing Script
# Comprehensive testing across multiple configurations with performance regression detection
# and cross-platform compatibility validation

set -euo pipefail

# Color codes for output
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
BUILD_MATRIX_LOG="$PROJECT_ROOT/.taskmaster/reports/build-matrix-$(date +%Y%m%d-%H%M%S).log"
PERFORMANCE_BASELINE="$PROJECT_ROOT/.taskmaster/reports/performance-baseline.json"
PERFORMANCE_THRESHOLD=0.15  # 15% performance regression threshold

# Build configurations to test
BUILD_CONFIGS=(
    "release:OFF:false:false:false:false"
    "release:OFF:true:false:false:false"
    "release:OFF:false:true:false:false"
    "release:OFF:false:false:true:false"
    "debug:OFF:false:false:false:false"
    "debug:OFF:true:false:false:false"
    "debug:OFF:false:true:false:false"
    "debug:OFF:false:false:true:false"
)

# Performance test configurations
PERFORMANCE_CONFIGS=(
    "1000:0.1:1"      # 1K points, 10% sparsity, 1D
    "5000:0.05:2"     # 5K points, 5% sparsity, 2D
    "10000:0.02:2"    # 10K points, 2% sparsity, 2D
    "50000:0.01:1"    # 50K points, 1% sparsity, 1D
)

# Logging functions
log_header() {
    echo -e "\n${BLUE}🧪 TDA Platform Build Matrix Testing${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Project: $PROJECT_ROOT"
    echo -e "Timestamp: $(date)"
    echo -e "Log file: $BUILD_MATRIX_LOG"
    echo -e "Configurations: ${#BUILD_CONFIGS[@]}"
    echo -e "Performance tests: ${#PERFORMANCE_CONFIGS[@]}\n"
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
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_performance() {
    echo -e "${PURPLE}📊 $1${NC}"
}

# Initialize testing environment
init_testing_environment() {
    log_section "Initializing Testing Environment"
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/.taskmaster/reports"
    mkdir -p "$PROJECT_ROOT/.taskmaster/artifacts"
    
    # Initialize log file
    {
        echo "TDA Platform Build Matrix Test Report"
        echo "====================================="
        echo "Timestamp: $(date)"
        echo "Project: $PROJECT_ROOT"
        echo "System: $(uname -a)"
        echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
        echo ""
    } > "$BUILD_MATRIX_LOG"
    
    log_success "Testing environment initialized"
}

# Parse build configuration
parse_build_config() {
    local config="$1"
    IFS=':' read -r build_type cuda clean tests benchmarks validation <<< "$config"
    echo "Build Type: $build_type, CUDA: $cuda, Clean: $clean, Tests: $tests, Benchmarks: $benchmarks, Validation: $validation"
}

# Run single build configuration
run_build_config() {
    local config="$1"
    local config_id="$2"
    
    log_section "Testing Configuration $config_id: $config"
    
    # Parse configuration
    IFS=':' read -r build_type cuda clean tests benchmarks validation <<< "$config"
    
    # Create build directory name
    local build_dir="build_matrix_${build_type}_${config_id}"
    
    # Log configuration
    {
        echo "Configuration $config_id:"
        echo "  Build Type: $build_type"
        echo "  CUDA: $cuda"
        echo "  Clean: $clean"
        echo "  Tests: $tests"
        echo "  Benchmarks: $benchmarks"
        echo "  Validation: $validation"
        echo ""
    } >> "$BUILD_MATRIX_LOG"
    
    # Run build with timeout
    local build_start=$(date +%s)
    local build_success=false
    local build_output=""
    
    if timeout 1800 "$PROJECT_ROOT/build.sh" "$build_type" "$cuda" "$clean" "$tests" "$benchmarks" "$validation" 2>&1; then
        build_success=true
        log_success "Configuration $config_id built successfully"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_error "Configuration $config_id timed out after 30 minutes"
        else
            log_error "Configuration $config_id failed with exit code $exit_code"
        fi
    fi
    
    local build_duration=$(( $(date +%s) - build_start ))
    
    # Log results
    {
        echo "  Result: $([ "$build_success" = true ] && echo "SUCCESS" || echo "FAILED")"
        echo "  Duration: ${build_duration}s"
        echo "  Exit Code: $exit_code"
        echo ""
    } >> "$BUILD_MATRIINE_LOG"
    
    # Return success status
    [[ "$build_success" = true ]]
}

# Run performance tests
run_performance_tests() {
    log_section "Running Performance Tests"
    
    local performance_results=()
    
    for config in "${PERFORMANCE_CONFIGS[@]}"; do
        IFS=':' read -r num_points sparsity max_dim <<< "$config"
        
        log_info "Testing: ${num_points} points, ${sparsity}% sparsity, ${max_dim}D"
        
        # Run controlled performance test
        local test_start=$(date +%s)
        local test_output=""
        local test_success=false
        
        if cd "$PROJECT_ROOT/build/release" && timeout 300 "./bin/controlled_performance_test" "$num_points" "$sparsity" "$max_dim" 2>&1; then
            test_success=true
            test_output=$(cd "$PROJECT_ROOT/build/release" && "./bin/controlled_performance_test" "$num_points" "$sparsity" "$max_dim" 2>&1)
        fi
        
        local test_duration=$(( $(date +%s) - test_start ))
        
        if [[ "$test_success" = true ]]; then
            # Extract performance metrics
            local execution_time=$(echo "$test_output" | grep "Execution time" | awk '{print $3}')
            local memory_usage=$(echo "$test_output" | grep "Memory usage" | awk '{print $3}')
            
            performance_results+=("$num_points:$sparsity:$max_dim:$execution_time:$memory_usage:$test_duration")
            
            log_performance "Points: $num_points, Time: ${execution_time}ms, Memory: ${memory_usage}MB"
        else
            log_warning "Performance test failed for $num_points points"
        fi
    done
    
    # Save performance results
    save_performance_results "${performance_results[@]}"
}

# Save performance results
save_performance_results() {
    local results=("$@")
    
    log_info "Saving performance results..."
    
    # Create JSON performance report
    cat > "$PROJECT_ROOT/.taskmaster/reports/performance-$(date +%Y%m%d-%H%M%S).json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system_info": {
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "kernel": "$(uname -r)",
        "cpu": "$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)",
        "memory_gb": "$(free -g | grep Mem | awk '{print $2}')"
    },
    "performance_tests": [
EOF
    
    for result in "${results[@]}"; do
        IFS=':' read -r num_points sparsity max_dim exec_time memory duration <<< "$result"
        
        cat >> "$PROJECT_ROOT/.taskmaster/reports/performance-$(date +%Y%m%d-%H%M%S).json" << EOF
        {
            "num_points": $num_points,
            "sparsity": $sparsity,
            "max_dimension": $max_dim,
            "execution_time_ms": $exec_time,
            "memory_usage_mb": $memory,
            "test_duration_s": $duration
        }$( [[ "$result" != "${results[-1]}" ]] && echo "," || echo "" )
EOF
    done
    
    cat >> "$PROJECT_ROOT/.taskmaster/reports/performance-$(date +%Y%m%d-%H%M%S).json" << EOF
    ]
}
EOF
    
    log_success "Performance results saved"
}

# Detect performance regressions
detect_performance_regressions() {
    log_section "Performance Regression Detection"
    
    if [[ ! -f "$PERFORMANCE_BASELINE" ]]; then
        log_warning "No performance baseline found. Creating baseline from current results..."
        cp "$PROJECT_ROOT/.taskmaster/reports/performance-$(date +%Y%m%d-%H%M%S).json" "$PERFORMANCE_BASELINE"
        return 0
    fi
    
    # Compare current results with baseline
    local current_file="$PROJECT_ROOT/.taskmaster/reports/performance-$(date +%Y%m%d-%H%M%S).json"
    
    if [[ ! -f "$current_file" ]]; then
        log_warning "No current performance results found for comparison"
        return 0
    fi
    
    # Extract metrics for comparison (simplified comparison)
    local baseline_time=$(jq -r '.performance_tests[0].execution_time_ms' "$PERFORMANCE_BASELINE" 2>/dev/null || echo "0")
    local current_time=$(jq -r '.performance_tests[0].execution_time_ms' "$current_file" 2>/dev/null || echo "0")
    
    if [[ "$baseline_time" != "0" && "$current_time" != "0" ]]; then
        local time_diff=$((current_time - baseline_time))
        local time_ratio=$(echo "scale=2; $current_time / $baseline_time" | bc -l 2>/dev/null || echo "1.0")
        
        if (( $(echo "$time_ratio > $((1 + PERFORMANCE_THRESHOLD))" | bc -l) )); then
            log_error "Performance regression detected!"
            log_error "Baseline: ${baseline_time}ms, Current: ${current_time}ms"
            log_error "Regression: ${time_diff}ms (${time_ratio}x)"
            
            # Log regression
            {
                echo "PERFORMANCE REGRESSION DETECTED:"
                echo "  Baseline: ${baseline_time}ms"
                echo "  Current: ${current_time}ms"
                echo "  Regression: ${time_diff}ms (${time_ratio}x)"
                echo "  Threshold: ${PERFORMANCE_THRESHOLD}"
                echo ""
            } >> "$BUILD_MATRIX_LOG"
            
            return 1
        else
            log_success "Performance within acceptable range (${time_ratio}x)"
        fi
    fi
    
    return 0
}

# Cross-platform compatibility testing
run_cross_platform_tests() {
    log_section "Cross-Platform Compatibility Testing"
    
    # Test compiler compatibility
    log_info "Testing compiler compatibility..."
    
    local compiler_tests=(
        "g++ -std=c++23 -E -x c++ /dev/null"
        "g++ -mavx2 -mfma -msse4.2 -E -x c++ /dev/null"
        "g++ -fopenmp -E -x c++ /dev/null"
    )
    
    for test_cmd in "${compiler_tests[@]}"; do
        if eval "$test_cmd" >/dev/null 2>&1; then
            log_success "Compiler test passed: $test_cmd"
        else
            log_warning "Compiler test failed: $test_cmd"
        fi
    done
    
    # Test library compatibility
    log_info "Testing library compatibility..."
    
    local library_tests=(
        "pkg-config --exists eigen3"
        "pkg-config --exists tbb"
        "test -f /usr/include/gudhi/Simplex_tree.h"
    )
    
    for test_cmd in "${library_tests[@]}"; do
        if eval "$test_cmd" >/dev/null 2>&1; then
            log_success "Library test passed: $test_cmd"
        else
            log_warning "Library test failed: $test_cmd"
        fi
    done
    
    # Test Python compatibility
    log_info "Testing Python compatibility..."
    
    if cd "$PROJECT_ROOT" && python3 -c "import pybind11; print('pybind11 version:', pybind11.__version__)" 2>/dev/null; then
        log_success "Python pybind11 compatibility test passed"
    else
        log_warning "Python pybind11 compatibility test failed"
    fi
}

# Generate comprehensive test report
generate_test_report() {
    log_section "Generating Test Report"
    
    local report_file="$PROJECT_ROOT/.taskmaster/reports/build-matrix-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# TDA Platform Build Matrix Test Report

**Generated:** $(date -Iseconds)  
**Project:** $PROJECT_ROOT  
**System:** $(uname -a)

## Test Summary

- **Total Configurations:** ${#BUILD_CONFIGS[@]}
- **Performance Tests:** ${#PERFORMANCE_CONFIGS[@]}
- **Log File:** $BUILD_MATRIX_LOG

## Build Configurations Tested

EOF
    
    # Add configuration details
    for i in "${!BUILD_CONFIGS[@]}"; do
        local config="${BUILD_CONFIGS[$i]}"
        IFS=':' read -r build_type cuda clean tests benchmarks validation <<< "$config"
        
        cat >> "$report_file" << EOF
### Configuration $((i+1))
- **Build Type:** $build_type
- **CUDA Support:** $cuda
- **Clean Build:** $clean
- **Run Tests:** $tests
- **Run Benchmarks:** $benchmarks
- **Skip Validation:** $validation

EOF
    done
    
    cat >> "$report_file" << EOF
## Performance Test Configurations

EOF
    
    # Add performance test details
    for i in "${!PERFORMANCE_CONFIGS[@]}"; do
        local config="${PERFORMANCE_CONFIGS[$i]}"
        IFS=':' read -r num_points sparsity max_dim <<< "$config"
        
        cat >> "$report_file" << EOF
### Performance Test $((i+1))
- **Points:** $num_points
- **Sparsity:** ${sparsity}%
- **Max Dimension:** $max_dim

EOF
    done
    
    cat >> "$report_file" << EOF
## Detailed Results

See the log file for detailed results: \`$BUILD_MATRIX_LOG\`

## Performance Baseline

Performance baseline file: \`$PERFORMANCE_BASELINE\`

## Next Steps

1. Review any failed configurations
2. Investigate performance regressions
3. Address compatibility issues
4. Update baseline if performance improvements are confirmed

---

*Report generated by TDA Platform Build Matrix Testing Script*
EOF
    
    log_success "Test report generated: $report_file"
}

# Main testing orchestration
main() {
    log_header
    
    # Initialize environment
    init_testing_environment
    
    # Track overall results
    local total_configs=${#BUILD_CONFIGS[@]}
    local successful_configs=0
    local failed_configs=0
    
    # Test all build configurations
    log_section "Testing Build Configurations"
    
    for i in "${!BUILD_CONFIGS[@]}"; do
        local config="${BUILD_CONFIGS[$i]}"
        local config_id=$((i+1))
        
        if run_build_config "$config" "$config_id"; then
            ((successful_configs++))
        else
            ((failed_configs++))
        fi
        
        # Add separator between configurations
        echo "" >> "$BUILD_MATRIX_LOG"
    done
    
    # Run performance tests
    run_performance_tests
    
    # Detect performance regressions
    if ! detect_performance_regressions; then
        log_warning "Performance regression detected - review required"
    fi
    
    # Run cross-platform compatibility tests
    run_cross_platform_tests
    
    # Generate final report
    generate_test_report
    
    # Final summary
    log_section "Test Matrix Summary"
    echo -e "${GREEN}✅ Successful Configurations: $successful_configs/$total_configs${NC}"
    
    if [[ $failed_configs -gt 0 ]]; then
        echo -e "${RED}❌ Failed Configurations: $failed_configs/$total_configs${NC}"
    fi
    
    echo -e "${BLUE}📊 Performance Tests: ${#PERFORMANCE_CONFIGS[@]}${NC}"
    echo -e "${BLUE}📋 Report: $BUILD_MATRIX_LOG${NC}"
    
    # Exit with appropriate code
    if [[ $failed_configs -eq 0 ]]; then
        log_success "All build configurations passed successfully!"
        exit 0
    else
        log_error "Some build configurations failed. Check the report for details."
        exit 1
    fi
}

# Trap for cleanup
trap 'echo -e "\n${RED}Build matrix testing interrupted by user${NC}"; exit 130' INT TERM

# Run main function
main "$@"

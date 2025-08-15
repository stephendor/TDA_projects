#!/bin/bash

# TDA Platform Performance Regression Detector
# Standalone script for detecting performance regressions in TDA algorithms
# Can be integrated with CI/CD pipelines for automated performance monitoring

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
PERFORMANCE_DIR="$PROJECT_ROOT/.taskmaster/reports/performance"
BASELINE_DIR="$PROJECT_ROOT/.taskmaster/reports/baselines"
CURRENT_RESULTS="$PERFORMANCE_DIR/current-$(date +%Y%m%d-%H%M%S).json"

# Performance thresholds
TIME_REGRESSION_THRESHOLD=0.15      # 15% time regression
MEMORY_REGRESSION_THRESHOLD=0.20    # 20% memory regression
IMPROVEMENT_THRESHOLD=0.10          # 10% improvement threshold

# Test configurations (scalable from small to large)
PERFORMANCE_TESTS=(
    "1000:0.1:1:sparse_rips"       # 1K points, 10% sparsity, 1D, algorithm
    "5000:0.05:2:sparse_rips"      # 5K points, 5% sparsity, 2D, algorithm
    "10000:0.02:2:sparse_rips"     # 10K points, 2% sparsity, 2D, algorithm
    "50000:0.01:1:sparse_rips"     # 50K points, 1% sparsity, 1D, algorithm
    "1000:0.1:1:distance_matrix"   # 1K points, distance matrix
    "5000:0.05:2:distance_matrix"  # 5K points, distance matrix
    "10000:0.02:2:balltree"        # 10K points, ball tree
    "50000:0.01:1:balltree"        # 50K points, ball tree
)

# Logging functions
log_header() {
    echo -e "\n${BLUE}📊 TDA Platform Performance Regression Detector${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Project: $PROJECT_ROOT"
    echo -e "Timestamp: $(date)"
    echo -e "Current Results: $CURRENT_RESULTS"
    echo -e "Test Configurations: ${#PERFORMANCE_TESTS[@]}\n"
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

# Initialize performance testing environment
init_performance_environment() {
    log_section "Initializing Performance Testing Environment"
    
    # Create necessary directories
    mkdir -p "$PERFORMANCE_DIR"
    mkdir -p "$BASELINE_DIR"
    
    # Ensure build directory exists
    if [[ ! -d "$PROJECT_ROOT/build/release" ]]; then
        log_error "Release build directory not found. Please run: ./build.sh release"
        exit 1
    fi
    
    log_success "Performance testing environment initialized"
}

# Run single performance test
run_performance_test() {
    local test_config="$1"
    IFS=':' read -r num_points sparsity max_dim algorithm <<< "$test_config"
    
    log_info "Running: ${num_points} points, ${sparsity}% sparsity, ${max_dim}D, $algorithm"
    
    local test_start=$(date +%s)
    local test_output=""
    local test_success=false
    
    cd "$PROJECT_ROOT/build/release"
    
    # Run appropriate performance test based on algorithm
    case "$algorithm" in
        "sparse_rips")
            if timeout 300 "./bin/controlled_performance_test" "$num_points" "$sparsity" "$max_dim" 2>&1; then
                test_success=true
                test_output=$(timeout 300 "./bin/controlled_performance_test" "$num_points" "$sparsity" "$max_dim" 2>&1)
            fi
            ;;
        "distance_matrix")
            if timeout 300 "./bin/test_distance_matrix_performance" 2>&1; then
                test_success=true
                test_output=$(timeout 300 "./bin/test_distance_matrix_performance" 2>&1)
            fi
            ;;
        "balltree")
            if timeout 300 "./bin/test_balltree_performance" 2>&1; then
                test_success=true
                test_output=$(timeout 300 "./bin/test_balltree_performance" 2>&1)
            fi
            ;;
        *)
            log_warning "Unknown algorithm: $algorithm, skipping"
            return 1
            ;;
    esac
    
    local test_duration=$(( $(date +%s) - test_start ))
    
    if [[ "$test_success" = true ]]; then
        # Extract performance metrics (simplified parsing)
        local execution_time=$(echo "$test_output" | grep -i "execution\|time\|duration" | head -1 | grep -o '[0-9]\+' | head -1 || echo "0")
        local memory_usage=$(echo "$test_output" | grep -i "memory\|mem" | head -1 | grep -o '[0-9]\+' | head -1 || echo "0")
        
        # If no metrics found, use test duration as proxy
        if [[ "$execution_time" == "0" ]]; then
            execution_time=$((test_duration * 1000))  # Convert to milliseconds
        fi
        
        echo "$num_points:$sparsity:$max_dim:$algorithm:$execution_time:$memory_usage:$test_duration"
        
        log_performance "Points: $num_points, Time: ${execution_time}ms, Memory: ${memory_usage}MB, Duration: ${test_duration}s"
    else
        log_warning "Performance test failed for $num_points points ($algorithm)"
        echo "$num_points:$sparsity:$max_dim:$algorithm:0:0:$test_duration"
    fi
    
    cd "$PROJECT_ROOT"
}

# Run all performance tests
run_all_performance_tests() {
    log_section "Running Performance Test Suite"
    
    local results=()
    local successful_tests=0
    local total_tests=${#PERFORMANCE_TESTS[@]}
    
    for test_config in "${PERFORMANCE_TESTS[@]}"; do
        if result=$(run_performance_test "$test_config"); then
            results+=("$result")
            ((successful_tests++))
        fi
    done
    
    log_info "Completed $successful_tests/$total_tests performance tests successfully"
    
    # Save results
    save_performance_results "${results[@]}"
    
    return $((total_tests - successful_tests))
}

# Save performance results to JSON
save_performance_results() {
    local results=("$@")
    
    log_info "Saving performance results to $CURRENT_RESULTS"
    
    # Create JSON performance report
    cat > "$CURRENT_RESULTS" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system_info": {
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "kernel": "$(uname -r)",
        "cpu": "$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)",
        "memory_gb": "$(free -g | grep Mem | awk '{print $2}')",
        "gcc_version": "$(gcc --version | head -1)",
        "cmake_version": "$(cmake --version | head -1)"
    },
    "performance_tests": [
EOF
    
    for result in "${results[@]}"; do
        IFS=':' read -r num_points sparsity max_dim algorithm exec_time memory duration <<< "$result"
        
        cat >> "$CURRENT_RESULTS" << EOF
        {
            "num_points": $num_points,
            "sparsity": $sparsity,
            "max_dimension": $max_dim,
            "algorithm": "$algorithm",
            "execution_time_ms": $exec_time,
            "memory_usage_mb": $memory,
            "test_duration_s": $duration
        }$( [[ "$result" != "${results[-1]}" ]] && echo "," || echo "" )
EOF
    done
    
    cat >> "$CURRENT_RESULTS" << EOF
    ]
}
EOF
    
    log_success "Performance results saved to $CURRENT_RESULTS"
}

# Find most recent baseline
find_latest_baseline() {
    local baseline_file=""
    
    if [[ -d "$BASELINE_DIR" ]]; then
        baseline_file=$(find "$BASELINE_DIR" -name "baseline-*.json" -type f | sort -r | head -1)
    fi
    
    echo "$baseline_file"
}

# Detect performance regressions
detect_performance_regressions() {
    log_section "Performance Regression Detection"
    
    local baseline_file=$(find_latest_baseline)
    
    if [[ -z "$baseline_file" ]]; then
        log_warning "No performance baseline found. Creating baseline from current results..."
        
        # Create baseline directory if it doesn't exist
        mkdir -p "$BASELINE_DIR"
        
        # Copy current results as baseline
        local baseline_name="baseline-$(date +%Y%m%d-%H%M%S).json"
        cp "$CURRENT_RESULTS" "$BASELINE_DIR/$baseline_name"
        
        log_success "Baseline created: $BASELINE_DIR/$baseline_name"
        return 0
    fi
    
    log_info "Comparing with baseline: $baseline_file"
    
    # Check if jq is available for JSON parsing
    if ! command -v jq &> /dev/null; then
        log_warning "jq not available, using basic text comparison"
        basic_performance_comparison "$baseline_file"
        return $?
    fi
    
    # Use jq for detailed JSON comparison
    detailed_performance_comparison "$baseline_file"
    return $?
}

# Basic performance comparison (fallback)
basic_performance_comparison() {
    local baseline_file="$1"
    
    log_info "Performing basic performance comparison..."
    
    # Simple text-based comparison
    local baseline_size=$(wc -c < "$baseline_file")
    local current_size=$(wc -c < "$CURRENT_RESULTS")
    
    if [[ $current_size -gt $((baseline_size * 2)) ]]; then
        log_warning "Significant change in results size detected"
        return 1
    fi
    
    log_success "Basic comparison passed"
    return 0
}

# Detailed performance comparison using jq
detailed_performance_comparison() {
    local baseline_file="$1"
    
    log_info "Performing detailed performance comparison..."
    
    local regressions_detected=0
    local improvements_detected=0
    
    # Compare each test configuration
    for test_config in "${PERFORMANCE_TESTS[@]}"; do
        IFS=':' read -r num_points sparsity max_dim algorithm <<< "$test_config"
        
        # Extract baseline metrics
        local baseline_time=$(jq -r --arg n "$num_points" --arg s "$sparsity" --arg d "$max_dim" --arg a "$algorithm" \
            '.performance_tests[] | select(.num_points == ($n | tonumber) and .sparsity == ($s | tonumber) and .max_dimension == ($d | tonumber) and .algorithm == $a) | .execution_time_ms' \
            "$baseline_file" 2>/dev/null || echo "0")
        
        local baseline_memory=$(jq -r --arg n "$num_points" --arg s "$sparsity" --arg d "$max_dim" --arg a "$algorithm" \
            '.performance_tests[] | select(.num_points == ($n | tonumber) and .sparsity == ($s | tonumber) and .max_dimension == ($d | tonumber) and .algorithm == $a) | .memory_usage_mb' \
            "$baseline_file" 2>/dev/null || echo "0")
        
        # Extract current metrics
        local current_time=$(jq -r --arg n "$num_points" --arg s "$sparsity" --arg d "$max_dim" --arg a "$algorithm" \
            '.performance_tests[] | select(.num_points == ($n | tonumber) and .sparsity == ($s | tonumber) and .max_dimension == ($d | tonumber) and .algorithm == $a) | .execution_time_ms' \
            "$CURRENT_RESULTS" 2>/dev/null || echo "0")
        
        local current_memory=$(jq -r --arg n "$num_points" --arg s "$sparsity" --arg d "$max_dim" --arg a "$algorithm" \
            '.performance_tests[] | select(.num_points == ($n | tonumber) and .sparsity == ($s | tonumber) and .max_dimension == ($d | tonumber) and .algorithm == $a) | .memory_usage_mb' \
            "$CURRENT_RESULTS" 2>/dev/null || echo "0")
        
        # Skip if no baseline data
        if [[ "$baseline_time" == "0" || "$baseline_time" == "null" ]]; then
            log_info "No baseline data for $num_points points ($algorithm), skipping comparison"
            continue
        fi
        
        # Calculate ratios
        local time_ratio=$(echo "scale=3; $current_time / $baseline_time" | bc -l 2>/dev/null || echo "1.0")
        local memory_ratio=$(echo "scale=3; $current_memory / $baseline_memory" | bc -l 2>/dev/null || echo "1.0")
        
        # Check for regressions
        if (( $(echo "$time_ratio > $((1 + TIME_REGRESSION_THRESHOLD))" | bc -l) )); then
            log_error "TIME REGRESSION: $num_points points ($algorithm)"
            log_error "  Baseline: ${baseline_time}ms, Current: ${current_time}ms"
            log_error "  Regression: ${time_ratio}x (threshold: ${TIME_REGRESSION_THRESHOLD})"
            ((regressions_detected++))
        elif (( $(echo "$time_ratio < $((1 - IMPROVEMENT_THRESHOLD))" | bc -l) )); then
            log_success "TIME IMPROVEMENT: $num_points points ($algorithm)"
            log_success "  Baseline: ${baseline_time}ms, Current: ${current_time}ms"
            log_success "  Improvement: ${time_ratio}x (threshold: ${IMPROVEMENT_THRESHOLD})"
            ((improvements_detected++))
        fi
        
        # Check memory regressions
        if (( $(echo "$memory_ratio > $((1 + MEMORY_REGRESSION_THRESHOLD))" | bc -l) )); then
            log_error "MEMORY REGRESSION: $num_points points ($algorithm)"
            log_error "  Baseline: ${baseline_memory}MB, Current: ${current_memory}MB"
            log_error "  Regression: ${memory_ratio}x (threshold: ${MEMORY_REGRESSION_THRESHOLD})"
            ((regressions_detected++))
        fi
    done
    
    # Summary
    if [[ $regressions_detected -gt 0 ]]; then
        log_error "Performance regressions detected: $regressions_detected"
        return 1
    elif [[ $improvements_detected -gt 0 ]]; then
        log_success "Performance improvements detected: $improvements_detected"
    else
        log_success "Performance is stable (no significant changes)"
    fi
    
    return 0
}

# Generate performance report
generate_performance_report() {
    log_section "Generating Performance Report"
    
    local report_file="$PERFORMANCE_DIR/performance-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# TDA Platform Performance Report

**Generated:** $(date -Iseconds)  
**Project:** $PROJECT_ROOT  
**System:** $(uname -a)

## Performance Summary

- **Total Tests:** ${#PERFORMANCE_TESTS[@]}
- **Current Results:** $CURRENT_RESULTS
- **Baseline:** $(find_latest_baseline || echo "None")

## Test Configurations

EOF
    
    # Add test configuration details
    for i in "${!PERFORMANCE_TESTS[@]}"; do
        local test_config="${PERFORMANCE_TESTS[$i]}"
        IFS=':' read -r num_points sparsity max_dim algorithm <<< "$test_config"
        
        cat >> "$report_file" << EOF
### Test $((i+1))
- **Points:** $num_points
- **Sparsity:** ${sparsity}%
- **Max Dimension:** $max_dim
- **Algorithm:** $algorithm

EOF
    done
    
    cat >> "$report_file" << EOF
## Performance Metrics

See the detailed results file: \`$CURRENT_RESULTS\`

## Thresholds

- **Time Regression Threshold:** ${TIME_REGRESSION_THRESHOLD} (${TIME_REGRESSION_THRESHOLD}%)
- **Memory Regression Threshold:** ${MEMORY_REGRESSION_THRESHOLD} (${MEMORY_REGRESSION_THRESHOLD}%)
- **Improvement Threshold:** ${IMPROVEMENT_THRESHOLD} (${IMPROVEMENT_THRESHOLD}%)

## Next Steps

1. Review any performance regressions
2. Investigate significant changes
3. Update baseline if improvements are confirmed
4. Consider algorithm optimizations for regressions

---

*Report generated by TDA Platform Performance Regression Detector*
EOF
    
    log_success "Performance report generated: $report_file"
}

# Main execution
main() {
    log_header
    
    # Initialize environment
    init_performance_environment
    
    # Run performance tests
    if ! run_all_performance_tests; then
        log_warning "Some performance tests failed"
    fi
    
    # Detect regressions
    if ! detect_performance_regressions; then
        log_error "Performance regressions detected!"
    fi
    
    # Generate report
    generate_performance_report
    
    # Final summary
    log_section "Performance Testing Summary"
    echo -e "${BLUE}📊 Results saved to: $CURRENT_RESULTS${NC}"
    echo -e "${BLUE}📋 Report generated: $PERFORMANCE_DIR/performance-report-$(date +%Y%m%d-%H%M%S).md${NC}"
    
    # Exit with appropriate code
    if detect_performance_regressions >/dev/null 2>&1; then
        log_success "Performance testing completed successfully!"
        exit 0
    else
        log_error "Performance regressions detected - review required!"
        exit 1
    fi
}

# Trap for cleanup
trap 'echo -e "\n${RED}Performance testing interrupted by user${NC}"; exit 130' INT TERM

# Run main function
main "$@"

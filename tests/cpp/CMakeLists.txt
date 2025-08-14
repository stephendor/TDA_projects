# =============================================================================
# TDA C++ Test Suite
# =============================================================================

# Enable testing
enable_testing()

# Find required testing dependencies
find_package(Threads REQUIRED)

# Set up include directories for tests
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# =============================================================================
# ACTUAL TESTS THAT EXIST
# =============================================================================

# Core functionality tests
add_executable(test_result
    test_result.cpp
)
target_link_libraries(test_result 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_result OpenMP::OpenMP_CXX)
endif()
add_test(NAME ResultType COMMAND test_result)

# Algorithm tests  
add_executable(test_sparse_rips
    test_sparse_rips.cpp
)
target_link_libraries(test_sparse_rips 
    tda_algorithms 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_sparse_rips OpenMP::OpenMP_CXX)
endif()
if(TBB_FOUND)
    target_link_libraries(test_sparse_rips TBB::tbb)
endif()
add_test(NAME SparseRips COMMAND test_sparse_rips)

# Distance matrix tests
add_executable(test_distance_matrix
    test_distance_matrix.cpp
)
target_link_libraries(test_distance_matrix 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_distance_matrix OpenMP::OpenMP_CXX)
endif()
add_test(NAME DistanceMatrix COMMAND test_distance_matrix)

# Streaming distance matrix tests
add_executable(test_streaming_distance_matrix
    test_streaming_distance_matrix.cpp
)
target_link_libraries(test_streaming_distance_matrix 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_streaming_distance_matrix OpenMP::OpenMP_CXX)
endif()
add_test(NAME StreamingDistanceMatrix COMMAND test_streaming_distance_matrix)

# Streaming distance matrix telemetry tests
add_executable(test_streaming_distance_telemetry
    test_streaming_distance_telemetry.cpp
)
target_link_libraries(test_streaming_distance_telemetry 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_streaming_distance_telemetry OpenMP::OpenMP_CXX)
endif()
add_test(NAME StreamingDistanceTelemetry COMMAND test_streaming_distance_telemetry)

# Streaming distance matrix early-stop tests
add_executable(test_streaming_distance_earlystop
    test_streaming_distance_earlystop.cpp
)
target_link_libraries(test_streaming_distance_earlystop 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_streaming_distance_earlystop OpenMP::OpenMP_CXX)
endif()
add_test(NAME StreamingDistanceEarlyStop COMMAND test_streaming_distance_earlystop)

# Simplex pool tests
add_executable(test_simplex_pool
    test_simplex_pool.cpp
)
target_link_libraries(test_simplex_pool 
    tda_core 
    tda_utils 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_simplex_pool OpenMP::OpenMP_CXX)
endif()
add_test(NAME SimplexPool COMMAND test_simplex_pool)

# Integration tests
add_executable(test_full_pipeline
    test_full_pipeline.cpp
)
target_link_libraries(test_full_pipeline 
    tda_algorithms 
    tda_vector_stack 
    tda_spatial 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_full_pipeline OpenMP::OpenMP_CXX)
endif()
if(TBB_FOUND)
    target_link_libraries(test_full_pipeline TBB::tbb)
endif()
add_test(NAME FullPipeline COMMAND test_full_pipeline)

# Mathematical validation tests
add_executable(test_mathematical_validation
    test_mathematical_validation.cpp
)
target_link_libraries(test_mathematical_validation 
    tda_algorithms 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_mathematical_validation OpenMP::OpenMP_CXX)
endif()
if(TBB_FOUND)
    target_link_libraries(test_mathematical_validation TBB::tbb)
endif()
add_test(NAME MathematicalValidation COMMAND test_mathematical_validation)

# ST-101 Memory-Optimized Performance Test
add_executable(test_st101_memory_optimized
    test_st101_memory_optimized.cpp
)
target_link_libraries(test_st101_memory_optimized 
    tda_algorithms 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_st101_memory_optimized OpenMP::OpenMP_CXX)
endif()
if(TBB_FOUND)
    target_link_libraries(test_st101_memory_optimized TBB::tbb)
endif()
add_test(NAME ST101MemoryOptimized COMMAND test_st101_memory_optimized)

# Temporarily disabled - requires GoogleTest conversion
# add_executable(test_vietoris_rips
#     test_vietoris_rips.cpp
# )
# target_link_libraries(test_vietoris_rips 
#     tda_algorithms 
#     tda_core 
#     Eigen3::Eigen 
#     Threads::Threads
# )
# if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
#     target_link_libraries(test_vietoris_rips OpenMP::OpenMP_CXX)
# endif()
# if(TBB_FOUND)
#     target_link_libraries(test_vietoris_rips TBB::tbb)
# endif()
# add_test(NAME VietorisRips COMMAND test_vietoris_rips)

# Čech pooling integration test
add_executable(test_cech_pool_integration
    test_cech_pool_integration.cpp
)
target_link_libraries(test_cech_pool_integration 
    tda_algorithms 
    tda_spatial 
    tda_core 
    tda_utils 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_cech_pool_integration OpenMP::OpenMP_CXX)
endif()
add_test(NAME CechPoolIntegration COMMAND test_cech_pool_integration)

# DTM pooling integration test
add_executable(test_dtm_pool_integration
    test_dtm_pool_integration.cpp
)
target_link_libraries(test_dtm_pool_integration 
    tda_algorithms 
    tda_spatial 
    tda_core 
    tda_utils 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_dtm_pool_integration OpenMP::OpenMP_CXX)
endif()
add_test(NAME DTMPoolIntegration COMMAND test_dtm_pool_integration)

# Streaming Čech test
add_executable(test_streaming_cech
    test_streaming_cech.cpp
)
target_link_libraries(test_streaming_cech 
    tda_algorithms 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_streaming_cech OpenMP::OpenMP_CXX)
endif()
add_test(NAME StreamingCech COMMAND test_streaming_cech)

# Streaming Čech performance harness (not a pass/fail test)
add_executable(test_streaming_cech_perf
    test_streaming_cech_perf.cpp
)
target_link_libraries(test_streaming_cech_perf 
    tda_algorithms 
    tda_utils 
    tda_core 
    Eigen3::Eigen 
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_streaming_cech_perf OpenMP::OpenMP_CXX)
endif()
add_custom_target(run_streaming_cech_perf
        COMMAND $<TARGET_FILE:test_streaming_cech_perf> --n 2000 --d 3 --radius 0.9 --maxDim 2 --maxNeighbors 64 --block 64
    DEPENDS test_streaming_cech_perf
    COMMENT "Running Streaming Čech performance harness (n=2000,d=3)"
)

# Larger-n target for manual/nightly runs (with CSV/JSON logging). Adjust sizes as needed.
add_custom_target(run_streaming_cech_perf_large
    COMMAND $<TARGET_FILE:test_streaming_cech_perf> --n 200000 --d 3 --radius 0.9 --maxDim 1 --maxNeighbors 32 --block 128 --max-pairs 250000000 --time-limit 300 --csv streaming_cech_perf.csv --json streaming_cech_perf.jsonl --dm-only
        DEPENDS test_streaming_cech_perf
        COMMENT "Running large Streaming DM-only harness (n=200k,d=3)"
)

# Medium probe (100k) DM-only with tighter limits
add_custom_target(run_streaming_dm_probe
    COMMAND $<TARGET_FILE:test_streaming_cech_perf> --n 100000 --d 3 --radius 0.9 --block 128 --dm-only --max-pairs 100000000 --time-limit 120 --csv streaming_cech_perf.csv --json streaming_cech_perf.jsonl
    DEPENDS test_streaming_cech_perf
    COMMENT "Running medium Streaming DM-only probe (n=100k,d=3)"
)

# Optional: register small perf probes with CTest for CI
option(ENABLE_PERF_CTESTS "Enable performance harness probes in CTest" OFF)
if(ENABLE_PERF_CTESTS)
    add_test(NAME PerfDMOnlySmall COMMAND $<TARGET_FILE:test_streaming_cech_perf> --n 5000 --d 3 --radius 0.9 --dm-only)
    set_tests_properties(PerfDMOnlySmall PROPERTIES TIMEOUT 120)
    add_test(NAME PerfCechSmall COMMAND $<TARGET_FILE:test_streaming_cech_perf> --n 3000 --d 3 --radius 0.9 --maxDim 2 --maxNeighbors 64 --block 64)
    set_tests_properties(PerfCechSmall PROPERTIES TIMEOUT 180)
endif()

# Baseline JSONL parse test (validates separate-process baseline mode wiring)
add_executable(test_perf_baseline_jsonl
    test_perf_baseline_jsonl.cpp
)
target_link_libraries(test_perf_baseline_jsonl
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_perf_baseline_jsonl OpenMP::OpenMP_CXX)
endif()
# Provide path to the perf harness binary at compile time
target_compile_definitions(test_perf_baseline_jsonl PRIVATE STREAMING_CECH_PERF_PATH="$<TARGET_FILE:test_streaming_cech_perf>")
add_test(NAME PerfBaselineJSONL COMMAND test_perf_baseline_jsonl)
set_tests_properties(PerfBaselineJSONL PROPERTIES TIMEOUT 120)

# Monotonicity/accuracy assertion test (CI-safe sizes)
add_executable(test_perf_monotonicity
    test_perf_monotonicity.cpp
)
target_link_libraries(test_perf_monotonicity
    Threads::Threads
)
if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_perf_monotonicity OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(test_perf_monotonicity PRIVATE STREAMING_CECH_PERF_PATH="$<TARGET_FILE:test_streaming_cech_perf>")
add_test(NAME PerfMonotonicity COMMAND test_perf_monotonicity)
set_tests_properties(PerfMonotonicity PROPERTIES TIMEOUT 180)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Set test properties
set_tests_properties(ResultType PROPERTIES TIMEOUT 60)
set_tests_properties(SparseRips PROPERTIES TIMEOUT 300)
set_tests_properties(DistanceMatrix PROPERTIES TIMEOUT 180)
set_tests_properties(StreamingDistanceMatrix PROPERTIES TIMEOUT 120)
set_tests_properties(StreamingDistanceTelemetry PROPERTIES TIMEOUT 120)
set_tests_properties(SimplexPool PROPERTIES TIMEOUT 60)
set_tests_properties(FullPipeline PROPERTIES TIMEOUT 600)
set_tests_properties(MathematicalValidation PROPERTIES TIMEOUT 300)
set_tests_properties(ST101MemoryOptimized PROPERTIES TIMEOUT 1200)  # 20 minutes for large datasets
set_tests_properties(CechPoolIntegration PROPERTIES TIMEOUT 120)
set_tests_properties(DTMPoolIntegration PROPERTIES TIMEOUT 120)
set_tests_properties(StreamingCech PROPERTIES TIMEOUT 120)

# Add custom test target for running all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4
    DEPENDS test_result test_sparse_rips test_distance_matrix test_streaming_distance_matrix test_full_pipeline test_mathematical_validation test_st101_memory_optimized
    COMMENT "Running all TDA tests"
)

message(STATUS "C++ tests directory configured - tests will be added later")
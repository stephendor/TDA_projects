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

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Set test properties
set_tests_properties(ResultType PROPERTIES TIMEOUT 60)
set_tests_properties(SparseRips PROPERTIES TIMEOUT 300)
set_tests_properties(DistanceMatrix PROPERTIES TIMEOUT 180)
set_tests_properties(FullPipeline PROPERTIES TIMEOUT 600)
set_tests_properties(MathematicalValidation PROPERTIES TIMEOUT 300)
# set_tests_properties(VietorisRips PROPERTIES TIMEOUT 300)

# Add custom test target for running all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4
    DEPENDS test_result test_sparse_rips test_distance_matrix test_full_pipeline test_mathematical_validation
    COMMENT "Running all TDA tests"
)

message(STATUS "C++ tests directory configured - tests will be added later")
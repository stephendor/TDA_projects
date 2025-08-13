cmake_minimum_required(VERSION 3.20)

project(TDA_VectorStack VERSION 1.0.0 LANGUAGES CXX)

# Set C++23 standard with strict requirements
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -ffast-math -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=address,undefined")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -ffast-math -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=address,undefined")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
endif()

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(PkgConfig REQUIRED)

# Set pybind11 path for virtual environment
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.13/site-packages/pybind11/share/cmake/pybind11")

# Core dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# TDA Libraries - GUDHI is header-only
if(EXISTS "/usr/include/gudhi")
    set(GUDHI_FOUND TRUE)
    set(GUDHI_INCLUDE_DIRS "/usr/include")
    message(STATUS "GUDHI found: /usr/include/gudhi")
else()
    message(FATAL_ERROR "GUDHI headers not found. Please install libgudhi-dev")
endif()

# Add Eigen3 include path for GUDHI compatibility
include_directories(${EIGEN3_INCLUDE_DIRS})

# Optional CUDA support for GPU acceleration
option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Find optional mathematical libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GMP QUIET gmp)
    pkg_check_modules(MPFR QUIET mpfr)
endif()

# Try to find GMP and MPFR using find_library as fallback
if(NOT GMP_FOUND)
    find_library(GMP_LIBRARIES NAMES gmp)
    if(GMP_LIBRARIES)
        set(GMP_FOUND TRUE)
        set(GMP_LIBRARIES ${GMP_LIBRARIES})
    endif()
endif()

if(NOT MPFR_FOUND)
    find_library(MPFR_LIBRARIES NAMES mpfr)
    if(MPFR_LIBRARIES)
        set(MPFR_FOUND TRUE)
        set(MPFR_LIBRARIES ${MPFR_LIBRARIES})
    endif()
endif()

# Set compile definitions based on library availability
if(GMP_FOUND AND MPFR_FOUND)
    add_definitions(-DHAVE_GMP_MPFR)
    message(STATUS "Found GMP and MPFR - enabling high-precision arithmetic")
else()
    message(STATUS "GMP and MPFR not found - using standard precision arithmetic")
    add_definitions(-DUSE_STANDARD_PRECISION)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${EIGEN3_INCLUDE_DIRS}
)

# C++ Core Library - Foundation for TDA operations
add_library(tda_core SHARED
    src/cpp/core/types.cpp
    src/cpp/core/point_cloud.cpp
    src/cpp/core/simplex.cpp
    src/cpp/core/filtration.cpp
    src/cpp/core/persistent_homology.cpp
    src/cpp/core/performance_profiler.cpp
)

target_compile_features(tda_core PRIVATE cxx_std_23)
target_link_libraries(tda_core 
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

target_include_directories(tda_core PRIVATE ${GUDHI_INCLUDE_DIRS})

# Vector Stack Library - Main focus of the project
add_library(tda_vector_stack SHARED
    src/cpp/vector_stack/vector_stack.cpp
    src/cpp/vector_stack/persistence_diagram.cpp
    src/cpp/vector_stack/barcode.cpp
    src/cpp/vector_stack/persistence_utils.cpp
    src/cpp/vector_stack/betti_numbers.cpp
    src/cpp/vector_stack/vector_operations.cpp
)

target_compile_features(tda_vector_stack PRIVATE cxx_std_23)
target_link_libraries(tda_vector_stack 
    tda_core
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# TDA Algorithms Library
add_library(tda_algorithms
    src/cpp/algorithms/cech_complex.cpp
    src/cpp/algorithms/dtm_filtration.cpp
    src/cpp/algorithms/vietoris_rips.cpp
    src/cpp/algorithms/alpha_complex.cpp
    src/cpp/algorithms/vietoris_rips_main.cpp
)

target_link_libraries(tda_algorithms
    tda_core
    tda_spatial
    tda_utils
    Eigen3::Eigen
    ${GMP_LIBRARIES}
    ${MPFR_LIBRARIES}
)

target_include_directories(tda_algorithms PRIVATE ${GUDHI_INCLUDE_DIRS})

# TDA Spatial Library
add_library(tda_spatial
    src/cpp/spatial/spatial_factory.cpp
    src/cpp/spatial/kdtree.cpp
)

target_link_libraries(tda_spatial
    tda_core
    tda_utils
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# TDA Utils Library
add_library(tda_utils
    src/cpp/utils/simd_utils.cpp
)

target_link_libraries(tda_utils
    tda_core
    Eigen3::Eigen
)

# Main TDA Platform Library
add_library(tda_platform SHARED
    src/cpp/tda_platform.cpp
)

target_compile_features(tda_platform PRIVATE cxx_std_23)
target_link_libraries(tda_platform 
    tda_core
    tda_vector_stack
    tda_algorithms
    tda_utils
    tda_spatial
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Python bindings using pybind11
find_package(pybind11 REQUIRED)
pybind11_add_module(tda_python 
    src/python/tda_module.cpp
    src/python/core_bindings.cpp
    src/python/vector_stack_bindings.cpp
    src/python/algorithms_bindings.cpp
)

target_link_libraries(tda_python PRIVATE 
    tda_platform
    pybind11::module
)

# Test suite
enable_testing()
add_subdirectory(tests)

# Performance benchmarks
add_executable(tda_benchmarks
    src/cpp/benchmarks/vector_stack_benchmarks.cpp
    src/cpp/benchmarks/algorithm_benchmarks.cpp
)

target_link_libraries(tda_benchmarks 
    tda_platform
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Test executable for SIMD utilities
add_executable(test_simd
    test_simd.cpp
)

target_link_libraries(test_simd 
    tda_utils
    tda_core
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Test executable for Vietoris-Rips
add_executable(test_vietoris_rips
    test_vietoris_rips.cpp
)

target_link_libraries(test_vietoris_rips 
    tda_algorithms
    tda_core
    tda_utils
    tda_vector_stack
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Test executable for Alpha Complex
add_executable(test_alpha_complex
    test_alpha_complex.cpp
)

target_link_libraries(test_alpha_complex 
    tda_algorithms
    tda_core
    tda_utils
    tda_vector_stack
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
    gmp
    mpfr
)

# Test executable for Spatial Index
add_executable(test_spatial_index
    test_spatial_index.cpp
)

target_link_libraries(test_spatial_index 
    tda_spatial
    tda_core
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Test executable for DTM Filtration
add_executable(test_dtm_filtration
    test_dtm_filtration.cpp
)

target_link_libraries(test_dtm_filtration 
    tda_algorithms
    tda_spatial
    tda_core
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
    gmp
    mpfr
)

# Test executable for Čech Complex
add_executable(test_cech_complex
    test_cech_complex.cpp
)

target_link_libraries(test_cech_complex 
    tda_algorithms
    tda_spatial
    tda_core
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
    gmp
    mpfr
)

# Test executable for Persistence Structures
add_executable(test_persistence_structures
    test_persistence_structures.cpp
)

target_link_libraries(test_persistence_structures 
    tda_vector_stack
    tda_core
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Performance benchmark executable
add_executable(test_performance_benchmarks
    test_performance_benchmarks.cpp
)

target_link_libraries(test_performance_benchmarks 
    tda_algorithms
    tda_vector_stack
    tda_core
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Installation
install(TARGETS 
    tda_core tda_vector_stack tda_algorithms tda_utils tda_spatial tda_platform
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS tda_python
    LIBRARY DESTINATION lib/python3/site-packages
)

install(DIRECTORY include/ DESTINATION include)

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(docs ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Print configuration summary
message(STATUS "TDA Vector Stack Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${ENABLE_CUDA}")
message(STATUS "  OpenMP Support: ${OpenMP_CXX_FOUND}")
message(STATUS "  Eigen3: ${EIGEN3_VERSION}")
message(STATUS "  GUDHI: Header-only library found")
message(STATUS "  Python: ${Python3_VERSION}")


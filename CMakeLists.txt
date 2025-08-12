cmake_minimum_required(VERSION 3.20)
project(TDA_VectorStack VERSION 1.0.0 LANGUAGES CXX Python)

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

# Core dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# Optional CUDA support for GPU acceleration
option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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
)

target_compile_features(tda_core PRIVATE cxx_std_23)
target_link_libraries(tda_core 
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Vector Stack Library - Main focus of the project
add_library(tda_vector_stack SHARED
    src/cpp/vector_stack/vector_stack.cpp
    src/cpp/vector_stack/persistence_diagram.cpp
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
add_library(tda_algorithms SHARED
    src/cpp/algorithms/vietoris_rips.cpp
    src/cpp/algorithms/alpha_complex.cpp
    src/cpp/algorithms/cech_complex.cpp
    src/cpp/algorithms/dtm_filtration.cpp
)

target_compile_features(tda_algorithms PRIVATE cxx_std_23)
target_link_libraries(tda_algorithms 
    tda_core
    tda_vector_stack
    ${EIGEN3_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Utilities Library
add_library(tda_utils SHARED
    src/cpp/utils/memory_pool.cpp
    src/cpp/utils/thread_pool.cpp
    src/cpp/utils/performance_monitor.cpp
    src/cpp/utils/simd_utils.cpp
)

target_compile_features(tda_utils PRIVATE cxx_std_23)
target_link_libraries(tda_utils 
    OpenMP::OpenMP_CXX
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

# Installation
install(TARGETS 
    tda_core tda_vector_stack tda_algorithms tda_utils tda_platform
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
message(STATUS "  Python: ${Python3_VERSION}")


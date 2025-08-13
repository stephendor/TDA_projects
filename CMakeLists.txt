cmake_minimum_required(VERSION 3.20)

project(TDA_VectorStack 
    VERSION 1.0.0 
    LANGUAGES CXX
    DESCRIPTION "High-Performance Topological Data Analysis Platform"
    HOMEPAGE_URL "https://github.com/your-org/TDA_projects"
)

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Set C++23 standard with strict requirements
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Project options
option(TDA_BUILD_TESTS "Build test executables" ON)
option(TDA_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(TDA_BUILD_PYTHON "Build Python bindings" ON)
option(TDA_BUILD_EXAMPLES "Build example programs" ON)
option(TDA_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(TDA_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(TDA_ENABLE_CUDA "Enable CUDA support" OFF)

# =============================================================================
# COMPILER CONFIGURATION
# =============================================================================

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -flto -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fsanitize=address,undefined -fno-omit-frame-pointer -fPIC")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -flto -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fsanitize=address,undefined -fno-omit-frame-pointer -fPIC")
endif()

# SIMD support detection
if(TDA_ENABLE_SIMD)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    
    if(COMPILER_SUPPORTS_AVX2)
        add_compile_definitions(TDA_ENABLE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    endif()
    if(COMPILER_SUPPORTS_FMA)
        add_compile_definitions(TDA_ENABLE_FMA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    endif()
    if(COMPILER_SUPPORTS_SSE42)
        add_compile_definitions(TDA_ENABLE_SSE42)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    endif()
endif()

# =============================================================================
# DEPENDENCIES
# =============================================================================

# Python and pybind11 (if building Python bindings)
if(TDA_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.13/site-packages/pybind11/share/cmake/pybind11")
    find_package(pybind11 REQUIRED)
endif()

# Eigen3 - Robust multi-method detection for maximum compatibility
set(EIGEN3_FOUND FALSE)

# Method 1: Try modern CMake find_package
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
    set(EIGEN3_FOUND TRUE)
    message(STATUS "Found Eigen3 via CMake: ${Eigen3_VERSION}")
endif()

# Method 2: Try PkgConfig as fallback
if(NOT EIGEN3_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 QUIET eigen3>=3.4)
        if(EIGEN3_FOUND)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIRS})
            target_compile_options(Eigen3::Eigen INTERFACE ${EIGEN3_CFLAGS_OTHER})
            message(STATUS "Found Eigen3 via PkgConfig: ${EIGEN3_VERSION}")
        endif()
    endif()
endif()

# Method 3: Manual search in standard locations
if(NOT EIGEN3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        PATHS 
            /usr/include/eigen3
            /usr/local/include/eigen3
            /opt/homebrew/include/eigen3  # macOS Homebrew
            /opt/local/include/eigen3     # macOS MacPorts
            /usr/include
            /usr/local/include
        PATH_SUFFIXES eigen3
    )
    
    if(EIGEN3_INCLUDE_DIR)
        # Verify Eigen works by testing a simple compile
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES ${EIGEN3_INCLUDE_DIR})
        check_cxx_source_compiles(
            "#include <Eigen/Core>\nint main() { Eigen::Vector3d v; return 0; }"
            EIGEN3_COMPILES
        )
        
        if(EIGEN3_COMPILES)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
            set(EIGEN3_FOUND TRUE)
            message(STATUS "Found Eigen3 via manual search: ${EIGEN3_INCLUDE_DIR}")
        else()
            message(WARNING "Found Eigen3 headers at ${EIGEN3_INCLUDE_DIR} but they don't compile")
        endif()
    endif()
endif()

# Method 4: Try to install Eigen3 development headers if not found
if(NOT EIGEN3_FOUND)
    message(WARNING "Eigen3 not found. Please install Eigen3 development headers:")
    message(WARNING "  Ubuntu/Debian: sudo apt-get install libeigen3-dev")
    message(WARNING "  Red Hat/CentOS: sudo yum install eigen3-devel")
    message(WARNING "  macOS: brew install eigen")
    message(FATAL_ERROR "Eigen3 3.4+ is required but not found")
endif()

# OpenMP
if(TDA_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Threading Building Blocks (TBB) - Required by GUDHI
find_package(TBB QUIET)
if(NOT TBB_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TBB QUIET tbb)
    endif()
    
    if(NOT TBB_FOUND)
        find_path(TBB_INCLUDE_DIR tbb/tbb.h
            PATHS /usr/include /usr/local/include /opt/intel/oneapi/tbb/latest/include
        )
        find_library(TBB_LIBRARY tbb
            PATHS /usr/lib /usr/local/lib /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8
        )
        
        if(TBB_INCLUDE_DIR AND TBB_LIBRARY)
            set(TBB_FOUND TRUE)
            add_library(TBB::tbb UNKNOWN IMPORTED)
            set_target_properties(TBB::tbb PROPERTIES
                IMPORTED_LOCATION "${TBB_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}"
            )
        endif()
    endif()
endif()

# GUDHI - Comprehensive detection
find_path(GUDHI_INCLUDE_DIR 
    NAMES gudhi/Simplex_tree.h
    PATHS /usr/include /usr/local/include /opt/gudhi/include
    PATH_SUFFIXES gudhi
)

if(GUDHI_INCLUDE_DIR)
    set(GUDHI_FOUND TRUE)
    message(STATUS "GUDHI found: ${GUDHI_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "GUDHI not found. Please install GUDHI or specify GUDHI_INCLUDE_DIR")
endif()

# High-precision arithmetic libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GMP gmp)
    pkg_check_modules(MPFR mpfr)
endif()

if(GMP_FOUND AND MPFR_FOUND)
    message(STATUS "Found GMP and MPFR - enabling high-precision arithmetic")
    add_compile_definitions(TDA_USE_HIGH_PRECISION)
endif()

# =============================================================================
# TARGET DEFINITIONS
# =============================================================================

# Include directories
include_directories(include)
include_directories(${GUDHI_INCLUDE_DIR})

# Core libraries
add_library(tda_core STATIC
    src/cpp/core/performance_profiler.cpp
    src/cpp/core/filtration.cpp
    src/cpp/core/persistent_homology.cpp
    src/cpp/core/point_cloud.cpp
    src/cpp/core/simple_core.cpp
    src/cpp/core/simplex.cpp
    src/cpp/core/simplex_pool.cpp
    src/cpp/core/types.cpp
)

add_library(tda_vector_stack STATIC
    src/cpp/vector_stack/vector_stack.cpp
    src/cpp/vector_stack/persistence_diagram.cpp
    src/cpp/vector_stack/barcode.cpp
    src/cpp/vector_stack/betti_numbers.cpp
    src/cpp/vector_stack/persistence_utils.cpp
    src/cpp/vector_stack/vector_operations.cpp
)

add_library(tda_algorithms STATIC
    src/cpp/algorithms/vietoris_rips.cpp
    src/cpp/algorithms/vietoris_rips_main.cpp
    src/cpp/algorithms/alpha_complex.cpp
    src/cpp/algorithms/sparse_rips.cpp
    src/cpp/algorithms/witness_complex.cpp
    src/cpp/algorithms/adaptive_sampling.cpp
    src/cpp/algorithms/cech_complex.cpp
    src/cpp/algorithms/dtm_filtration.cpp
    src/cpp/algorithms/streaming_cech.cpp
)

add_library(tda_spatial STATIC
    src/cpp/spatial/kdtree.cpp
    src/cpp/spatial/balltree.cpp
    src/cpp/spatial/spatial_factory.cpp
)

add_library(tda_utils STATIC
    src/cpp/utils/simd_utils.cpp
    src/cpp/utils/memory_pool.cpp
    src/cpp/utils/performance_monitor.cpp
    src/cpp/utils/distance_matrix.cpp
    src/cpp/utils/streaming_distance_matrix.cpp
)

# Main platform executable
add_executable(tda_platform src/cpp/tda_platform.cpp)

# Python bindings (if enabled)
if(TDA_BUILD_PYTHON)
    pybind11_add_module(tda_python
        src/python/tda_module.cpp
        src/python/core_bindings.cpp
        src/python/algorithms_bindings.cpp
        src/python/vector_stack_bindings.cpp
    )
endif()

# =============================================================================
# TARGET PROPERTIES AND LINKING
# =============================================================================

# Set target properties for all libraries
set(TDA_LIBRARIES tda_core tda_vector_stack tda_algorithms tda_spatial tda_utils)

foreach(lib ${TDA_LIBRARIES})
    target_link_libraries(${lib} PUBLIC Eigen3::Eigen)
    if(TDA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${lib} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(TBB_FOUND)
        target_link_libraries(${lib} PUBLIC TBB::tbb)
    endif()
    if(GMP_FOUND)
        target_link_libraries(${lib} PUBLIC ${GMP_LIBRARIES})
        target_include_directories(${lib} PUBLIC ${GMP_INCLUDE_DIRS})
    endif()
    if(MPFR_FOUND)
        target_link_libraries(${lib} PUBLIC ${MPFR_LIBRARIES})
        target_include_directories(${lib} PUBLIC ${MPFR_INCLUDE_DIRS})
    endif()
endforeach()

# Platform executable linking
target_link_libraries(tda_platform ${TDA_LIBRARIES})

# Python module linking (if enabled)
if(TDA_BUILD_PYTHON)
    target_link_libraries(tda_python PRIVATE ${TDA_LIBRARIES})
endif()

# =============================================================================
# TESTS AND BENCHMARKS
# =============================================================================

if(TDA_BUILD_TESTS)
    enable_testing()
    
    # Core C++ tests
    add_subdirectory(tests/cpp)
    
    # Individual performance tests
    if(TDA_BUILD_BENCHMARKS)
        set(PERFORMANCE_TESTS
            controlled_performance_test
            test_balltree_performance
            test_distance_matrix_performance
            test_cpp23_working_features
        )
        
        foreach(test ${PERFORMANCE_TESTS})
            add_executable(${test} ${test}.cpp)
            target_link_libraries(${test} ${TDA_LIBRARIES})
        endforeach()
    endif()
endif()

# =============================================================================
# EXAMPLES
# =============================================================================

if(TDA_BUILD_EXAMPLES)
    add_executable(vietoris_rips_demo examples/vietoris_rips_demo.cpp)
    target_link_libraries(vietoris_rips_demo ${TDA_LIBRARIES})
endif()

# =============================================================================
# INSTALLATION
# =============================================================================

# Install libraries
install(TARGETS ${TDA_LIBRARIES} tda_platform
    EXPORT TDATargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install Python module (if built)
if(TDA_BUILD_PYTHON)
    install(TARGETS tda_python
        LIBRARY DESTINATION lib/python3/site-packages
    )
endif()

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Install CMake config files
install(EXPORT TDATargets
    FILE TDATargets.cmake
    NAMESPACE TDA::
    DESTINATION lib/cmake/TDA
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    TDAConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TDAConfigVersion.cmake"
    DESTINATION lib/cmake/TDA
)

# =============================================================================
# DOCUMENTATION
# =============================================================================

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    if(EXISTS ${DOXYGEN_IN})
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        add_custom_target(docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    else()
        message(STATUS "Doxygen configuration file not found at ${DOXYGEN_IN}")
    endif()
else()
    message(STATUS "Doxygen not found - documentation will not be generated")
endif()

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

message(STATUS "TDA Vector Stack Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${TDA_ENABLE_CUDA}")
message(STATUS "  OpenMP Support: ${TDA_ENABLE_OPENMP}")
message(STATUS "  SIMD Support: ${TDA_ENABLE_SIMD}")
message(STATUS "  Eigen3: ${Eigen3_VERSION}")
message(STATUS "  GUDHI: Header-only library found")
message(STATUS "  TBB: ${TBB_FOUND}")
message(STATUS "  Python: ${Python3_VERSION}")
message(STATUS "  Build Tests: ${TDA_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${TDA_BUILD_BENCHMARKS}")
message(STATUS "  Build Python: ${TDA_BUILD_PYTHON}")
message(STATUS "  Build Examples: ${TDA_BUILD_EXAMPLES}")
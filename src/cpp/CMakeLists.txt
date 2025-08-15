# C++ Source Directory CMakeLists.txt
# Builds the core TDA algorithms and data structures

cmake_minimum_required(VERSION 3.20)
project(TDA_Core VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found, building without parallelization")
    set(OpenMP_CXX_FLAGS "")
endif()

# Try to find Eigen3, but make it optional for now
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${Eigen3_INCLUDE_DIRS}")
    set(EIGEN3_AVAILABLE TRUE)
else()
    message(WARNING "Eigen3 not found, some features may be limited")
    set(EIGEN3_AVAILABLE FALSE)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra")

# Source files - using simplified versions for now
set(CORE_SOURCES
    core/simple_core.cpp
)

set(VECTOR_STACK_SOURCES
    vector_stack/vector_stack.cpp
)

set(ALGORITHM_SOURCES
    algorithms/simple_algorithms.cpp
)

set(UTILITY_SOURCES
    # utils/memory_pool.cpp
    # utils/thread_pool.cpp
    # utils/performance_monitor.cpp
    # utils/simd_utils.cpp
)

# Create core library
add_library(tda_core STATIC ${CORE_SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(tda_core OpenMP::OpenMP_CXX)
endif()
if(EIGEN3_AVAILABLE)
    target_link_libraries(tda_core Eigen3::Eigen)
endif()
target_compile_features(tda_core PRIVATE cxx_std_23)

# Create vector stack library
add_library(tda_vector_stack STATIC ${VECTOR_STACK_SOURCES})
target_link_libraries(tda_vector_stack tda_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tda_vector_stack OpenMP::OpenMP_CXX)
endif()
if(EIGEN3_AVAILABLE)
    target_link_libraries(tda_vector_stack Eigen3::Eigen)
endif()
target_compile_features(tda_vector_stack PRIVATE cxx_std_23)

# Create algorithms library
add_library(tda_algorithms STATIC ${ALGORITHM_SOURCES})
target_link_libraries(tda_algorithms tda_core tda_vector_stack)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tda_algorithms OpenMP::OpenMP_CXX)
endif()
if(EIGEN3_AVAILABLE)
    target_link_libraries(tda_algorithms Eigen3::Eigen)
endif()
target_compile_features(tda_algorithms PRIVATE cxx_std_23)

# Create utilities library (when implemented)
if(UTILITY_SOURCES)
    add_library(tda_utils STATIC ${UTILITY_SOURCES})
    target_link_libraries(tda_utils tda_core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tda_utils OpenMP::OpenMP_CXX)
    endif()
    if(EIGEN3_AVAILABLE)
        target_link_libraries(tda_utils Eigen3::Eigen)
    endif()
    target_compile_features(tda_utils PRIVATE cxx_std_23)
endif()

# Create main TDA library that combines everything
add_library(tda STATIC
    ${CORE_SOURCES}
    ${VECTOR_STACK_SOURCES}
    ${ALGORITHM_SOURCES}
    ${UTILITY_SOURCES}
)

target_link_libraries(tda 
    tda_core 
    tda_vector_stack 
    tda_algorithms
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tda OpenMP::OpenMP_CXX)
endif()

if(EIGEN3_AVAILABLE)
    target_link_libraries(tda Eigen3::Eigen)
endif()

target_compile_features(tda PRIVATE cxx_std_23)

# Create test executable
add_executable(tda_test test_main.cpp)
target_link_libraries(tda_test tda)
target_compile_features(tda_test PRIVATE cxx_std_23)

# Set output directories
set_target_properties(tda_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(tda_vector_stack PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(tda_algorithms PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(tda PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(tda_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install targets
install(TARGETS tda_core tda_vector_stack tda_algorithms tda
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS tda_test
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "TDA Core Library: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP: Not found")
endif()
if(EIGEN3_AVAILABLE)
    message(STATUS "Eigen3: ${Eigen3_INCLUDE_DIRS}")
else()
    message(STATUS "Eigen3: Not found")
endif()
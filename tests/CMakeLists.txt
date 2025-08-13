# Tests CMakeLists.txt
# Basic test configuration for TDA Vector Stack

# Enable testing
enable_testing()

# Add test subdirectories if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
    add_subdirectory(cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python")
    add_subdirectory(python)
endif()

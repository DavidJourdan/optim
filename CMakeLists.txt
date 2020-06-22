cmake_minimum_required(VERSION 3.9)
project(optim)

# Detects whether this is a top-level project
get_directory_property(PARENT_DIR PARENT_DIRECTORY)
if(NOT PARENT_DIR)
    set(TOPLEVEL_PROJECT ON)
else()
    set(TOPLEVEL_PROJECT OFF)
endif()

# Build tests
option(BUILD_TESTS    "Build optim unit test"  OFF)

# Build example project
if(TOPLEVEL_PROJECT AND NOT BUILD_TESTS)
    option(BUILD_EXAMPLE  "Build example project"  OFF)
endif()

option(OPTIM_USE_CHOLMOD  "Use Cholmod linear solver instead of Eigen's built-in one"  OFF)

### conditionally compile certain modules depending on libraries found on the system
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

### Adding optim: choose the path to your local copy optim
include(optim)

if(BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif()
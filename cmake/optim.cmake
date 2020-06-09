cmake_minimum_required(VERSION 3.9)

### Other options
option(OPTIM_USE_CHOLMOD  "Use Cholmod linear solver instead of Eigen's built-in one"  ON)

################################################################################

### Configuration
set(OPTIM_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(OPTIM_SOURCE_DIR "${OPTIM_ROOT}/include")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

################################################################################

### Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set (EXTRA_LIBS ${EXTRA_LIBS} Eigen3::Eigen)

### optim
file(GLOB SRC ${OPTIM_SOURCE_DIR}/optim/NewtonSolver.cpp ${OPTIM_SOURCE_DIR}/optim/filter_var.cpp)
add_library(optim ${SRC})

# c++ flags
set_target_properties(optim PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED OFF
        CXX_EXTENSIONS ON
        )

if (OPTIM_USE_CHOLMOD)
  find_package (Cholmod REQUIRED QUIET)
  include_directories (${CHOLMOD_INCLUDES})
  add_definitions (-DNEWTON_USE_CHOLMOD)
  set (EXTRA_LIBS ${EXTRA_LIBS} ${CHOLMOD_LIBRARIES})
endif (OPTIM_USE_CHOLMOD)

target_link_libraries(optim ${EXTRA_LIBS})
include_directories(${OPTIM_SOURCE_DIR})

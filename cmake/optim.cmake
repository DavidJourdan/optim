cmake_minimum_required(VERSION 3.9)

### Other options
option(OPTIM_USE_CHOLMOD  "Use Cholmod linear solver instead of Eigen's built-in one"  ON)

################################################################################

### Configuration
set(OPTIM_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(OPTIM_SOURCE_DIR "${OPTIM_ROOT}/include")
set(OPTIM_INCL_DIR "${OPTIM_ROOT}/include/optim")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

################################################################################

### Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(EXTRA_LIBS ${EXTRA_LIBS} Eigen3::Eigen)

find_package (Threads)
set(EXTRA_LIBS ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})

### optim
file(GLOB SRC ${OPTIM_INCL_DIR}/NewtonSolver.cpp ${OPTIM_INCL_DIR}/filter_var.cpp ${OPTIM_INCL_DIR}/SolverBase.cpp
  ${OPTIM_INCL_DIR}/LBFGS.cpp)
add_library(optim ${SRC})

# c++ flags
set_target_properties(optim PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED OFF
        CXX_EXTENSIONS ON
        )

option(WITH_CHOLMOD "Compile optim with CHOLMOD" ON)

if(WITH_CHOLMOD)
  find_package (CHOLMOD REQUIRED QUIET)
  if(CHOLMOD_FOUND)
    target_include_directories(optim PUBLIC ${CHOLMOD_INCLUDES})
    target_compile_definitions(optim PUBLIC OPTIM_USE_CHOLMOD)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CHOLMOD_LIBRARIES})
  endif(CHOLMOD_FOUND)
endif(WITH_CHOLMOD)

target_link_libraries(optim ${EXTRA_LIBS})
target_include_directories(optim PUBLIC ${OPTIM_SOURCE_DIR})

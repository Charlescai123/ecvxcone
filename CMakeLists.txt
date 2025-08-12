# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(ecvxcone C)

set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_BUILD_TYPE Release)

# Set the C standard to C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

# Option to enable or disable unit test building
option(BUILD_TESTS "Build unit tests" OFF)

# Try to find a BLAS library (OpenBLAS, ATLAS, MKL, etc.)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

# Add the ECVXCONE headers
set(ecvxcone_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/include/cvxopt.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lapack.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/misc.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/blas.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/solver.h"
)

# Add the ECVXCONE sources
set(ecvxcone_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/base.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/blas.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/debug.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dense.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/lapack.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/misc.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sparse.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver_utils.c"
)

# ECVXCONE library
add_library(${PROJECT_NAME} SHARED ${ecvxcone_headers} ${ecvxcone_sources})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# CVXPYGEN sources
file(GLOB CVXPYGEN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cvxpygen/ECVXCONE/c/src/cpg_solve.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cvxpygen/ECVXCONE/c/src/cpg_workspace.c
)

# Set the executable for example usage
add_executable(lmi 
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/lmi.c
    ${CVXPYGEN_SOURCES}
)

target_include_directories(lmi PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cvxpygen/ECVXCONE/c/include/"
)

# target_compile_definitions(${PROJECT_NAME} PRIVATE BLAS_NO_UNDERSCORE)

target_link_libraries(${PROJECT_NAME} m lapack blas)

target_link_libraries(lmi PRIVATE ${PROJECT_NAME})

message(STATUS "------ Debug Info ------")
message(STATUS "CMAKE_SOURCE_DIR        = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR        = ${CMAKE_BINARY_DIR}")
message(STATUS "------------------------")


# Conditional block to build tests if enabled
if(BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_FILES tests/c/*.c)
    
    add_executable(test_main ${TEST_FILES})

    target_include_directories(test_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include) 

    target_link_libraries(test_main ${PROJECT_NAME})
    
endif()


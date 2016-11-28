cmake_minimum_required (VERSION 2.6)
project (SpLDLT C CXX Fortran)

# Include our own cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

# Set C flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2" )
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2 -ffpe-trap=underflow,denormal" )

# Set CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2" )
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -ffpe-trap=underflow,denormal" )

# Set Fortran flags
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -g -O2" )
# set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -g -O2 -ffpe-trap=underflow,denormal")

## Set runtime system to be used for the execution
SET(RUNTIME "STF" CACHE STRING "Runtime system")
set_property(CACHE RUNTIME PROPERTY STRINGS STF StarPU OMP Parsec)
MESSAGE( STATUS "Runtime:         " ${RUNTIME} )

# Decide wheter we want to compile the kernel test driver or not
option(KERNEL_TEST "Kernel test" OFF)
# MESSAGE( STATUS "Kernel test:         " ${KERNEL_TEST})

## spral
set(SPRAL_DIR $ENV{SPRALDIR} CACHE PATH "Location of Spral Library")
include_directories(${SPRAL_DIR})

set(SPRAL_SRC_DIR $ENV{SPRALSRCDIR} CACHE PATH "Location of Spral sources")
include_directories(${SPRAL_SRC_DIR})
include_directories(${SPRAL_SRC_DIR}/src)

# Get object files 
file(
  GLOB_RECURSE
  obj
  ${SPRAL_DIR}/tests/ssids/kernels/*.o)

MESSAGE(STATUS "object files: " ${obj})
# add_library(spral_ssids_tests STATIC IMPORTED ${obj})

# Include source directory
include_directories(${SpLDLT_SOURCE_DIR}/src)

if (${RUNTIME} MATCHES "STF")

  # Use STF code
  add_definitions(-DSPLDLT_USE_STF)

elseif(${RUNTIME} MATCHES "StarPU")

  # For some reason StarPU need OpenMP on scarf
  # set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  # set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp")

  # Use StarPU code
  add_definitions(-DSPLDLT_USE_STARPU)

  # StarPU lib
  set(STARPU_DIR $ENV{STARPUDIR} CACHE PATH "Location of StarPU Library")
  include_directories(${STARPU_DIR}/include/starpu/1.3)
  find_library(STARPU_LIBRARY starpu-1.3 PATHS ${STARPU_DIR}/lib)
  
  # HWLOC
  set(HWLOC_DIR $ENV{HWLOCDIR} CACHE PATH "Location of HWLOC Library")
  include_directories(${HWLOC_DIR}/include)  
  find_library(HWLOC_LIBRARY hwloc PATHS ${HWLOC_DIR}/lib)

endif()

add_subdirectory(src)

add_executable(spldlt_test drivers/spldlt_test.F90)

## spldlt
target_include_directories(spldlt_test PUBLIC ${CMAKE_BINARY_DIR}/src)
target_link_libraries(spldlt_test spldlt)

## Kernel test driver
if (${KERNEL_TEST} MATCHES "ON")

  file(
    GLOB_RECURSE
    test_src
    tests/*.cxx
    tests/*.hxx)
  
  # add_executable(spldlt_kernel_test tests/kernel_test.cxx)  
  add_executable(spldlt_kernel_test ${test_src})
  target_link_libraries(spldlt_kernel_test ${obj})

endif()

## spral
target_include_directories(spldlt_test PUBLIC ${SPRAL_DIR})
find_library(SPRAL_LIBRARY spral PATHS ${SPRAL_DIR} ${SPRAL_DIR})
target_link_libraries(spldlt_test ${SPRAL_LIBRARY})

if (${KERNEL_TEST} MATCHES "ON")
  # MESSAGE( STATUS "spral directory:         " ${SPRAL_DIR})

  target_include_directories(spldlt_kernel_test PUBLIC ${SPRAL_DIR})
  target_link_libraries(spldlt_kernel_test ${SPRAL_LIBRARY})
endif()

## HWLOC
set(HWLOC_DIR $ENV{HWLOCDIR} CACHE PATH "Location of HWLOC Library")
include_directories(${HWLOC_DIR}/include)  
find_library(HWLOC_LIBRARY hwloc PATHS ${HWLOC_DIR}/lib)
target_link_libraries(spldlt_test ${HWLOC_LIBRARY})

## BLAS and LAPACK
### MKL
set(MKL_LIBS $ENV{MKL_LIBS})
set(MKL_ROOT "/opt/intel/mkl" CACHE PATH "Location of MKL Library")
set(MKL_DIR $ENV{MKLROOT})

find_library(MKL_IFACE_LIBRARY mkl_gf_lp64 PATHS ${MKL_LIBS} ${MKL_ROOT}/lib/intel64 ${MKL_DIR}/lib/intel64)
# find_library(MKL_THREAD_LIBRARY mkl_gnu_thread PATHS ${MKL_ROOT}/lib/intel64)
find_library(MKL_SEQ_LIBRARY mkl_sequential PATHS ${MKL_LIBS} ${MKL_DEFAULT_ROOT}/lib/intel64 ${MKL_DIR}/lib/intel64)
find_library(MKL_CORE_LIBRARY mkl_core PATHS ${MKL_LIBS} ${MKL_ROOT}/lib/intel64 ${MKL_DIR}/lib/intel64) 
# set(MKL_LIBRARIES ${MKL_IFACE_LIBRARY} ${MKL_THREAD_LIBRARY} ${MKL_CORE_LIBRARY})
set(MKL_LIBRARIES ${MKL_IFACE_LIBRARY} ${MKL_SEQ_LIBRARY} ${MKL_CORE_LIBRARY})

target_link_libraries(spldlt_test ${MKL_LIBRARIES})

if (${KERNEL_TEST} MATCHES "ON")
  target_link_libraries(spldlt_kernel_test ${MKL_LIBRARIES})
endif()

## Ordering
### Metis
set(METIS_DIR $ENV{METISDIR} CACHE PATH "Location of Metis Library")
find_library(METIS_LIBRARY metis PATHS ${METIS_DIR})
target_link_libraries(spldlt_test ${METIS_LIBRARY})

## pthread
target_link_libraries(spldlt_test pthread)

## std C++
target_link_libraries(spldlt_test stdc++)

## runtime system
### StarPU
if(${RUNTIME} MATCHES "StarPU")
  # StarPU
  target_link_libraries(spldlt_test ${STARPU_LIBRARY})
  # HWLOC
  target_link_libraries(spldlt_test ${HWLOC_LIBRARY})
endif()

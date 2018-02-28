cmake_minimum_required (VERSION 2.6)
project (SpLDLT)
enable_language(Fortran)
enable_language(C)
enable_language(CXX)

# Include our own cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

################################################################################
# Compiler options

# C flags
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2" )
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2 -ffpe-trap=underflow,denormal" )

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -std=c++11" )
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ffpe-trap=underflow,denormal" )

# Fortran flags
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mtune=native" )
# set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffpe-trap=underflow,denormal")

# Set default build type
set(default_build_type "RelWithDebInfo")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Using default build type '${default_build_type}' because none was specified")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build (Debug, Release, MinSizeRel, RelWithDebInfo)" FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_Fortran_FLAGS_ALL ${CMAKE_Fortran_FLAGS})
set(CMAKE_C_FLAGS_ALL ${CMAKE_C_FLAGS})
set(CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  list(APPEND CMAKE_Fortran_FLAGS_ALL ${CMAKE_Fortran_FLAGS_DEBUG})
  list(APPEND CMAKE_C_FLAGS_ALL ${CMAKE_C_FLAGS_DEBUG})
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
  list(APPEND CMAKE_Fortran_FLAGS_ALL ${CMAKE_Fortran_FLAGS_RELEASE})
  list(APPEND CMAKE_C_FLAGS_ALL ${CMAKE_C_FLAGS_RELEASE})
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_RELEASE})
elseif(CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
  list(APPEND CMAKE_Fortran_FLAGS_ALL ${CMAKE_Fortran_FLAGS_MINSIZEREL})
  list(APPEND CMAKE_C_FLAGS_ALL ${CMAKE_C_FLAGS_MINSIZEREL})
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_MINSIZEREL})
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")  
  list(APPEND CMAKE_Fortran_FLAGS_ALL ${CMAKE_Fortran_FLAGS_RELWITHDEBINFO})
  list(APPEND CMAKE_C_FLAGS_ALL ${CMAKE_C_FLAGS_RELWITHDEBINFO})
  list(APPEND CMAKE_CXX_FLAGS_ALL ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
endif()

################################################################################
# Options

## Set runtime system to be used for the execution
SET(RUNTIME "STF" CACHE STRING "Runtime system")
set_property(CACHE RUNTIME PROPERTY STRINGS STF StarPU)
MESSAGE( STATUS "Runtime:         " ${RUNTIME} )

# # Set BLAS and LAPACK libraries to use for linking
# SET(BLAS_LAPACK "MKL" CACHE STRING "BLAS and LAPACK libraries")
# set_property(CACHE BLAS_LAPACK PROPERTY STRINGS Ref MKL)
# MESSAGE( STATUS "BLAS/LAPACK:         " ${BLAS_LAPACK} )

# Decide wheter we want to compile the kernel test driver or not
option(KERNEL_TEST "Kernel test" OFF)
# MESSAGE( STATUS "Kernel test:         " ${KERNEL_TEST})

# spral
set(SPRAL_DIR $ENV{SPRALDIR} CACHE PATH "Location of Spral Library")
include_directories(${SPRAL_DIR})

set(SPRAL_SRC_DIR $ENV{SPRALSRCDIR} CACHE PATH "Location of Spral sources")
include_directories(${SPRAL_SRC_DIR})
include_directories(${SPRAL_SRC_DIR}/src)

MESSAGE(STATUS "spral source dir: " ${SPRAL_SRC_DIR})

# Get object files 
file(
  GLOB_RECURSE
  obj
  ${SPRAL_DIR}/tests/ssids/kernels/*.o)

MESSAGE(STATUS "object files: " ${obj})
# add_library(spral_ssids_tests STATIC IMPORTED ${obj})

# Use GPUs
option(USE_GPU "Use GPU" OFF)
MESSAGE( STATUS "Use GPU:         " ${USE_GPU} )

# Use MPI
option(USE_MPI "Use MPI" OFF)
MESSAGE( STATUS "Use MPI:         " ${USE_MPI} )   

################################################################################
# Third-party libraries

########################################
# HWLOC
set(HWLOC_DIR $ENV{HWLOCDIR} CACHE PATH "Location of HWLOC Library")
include_directories(${HWLOC_DIR}/include)  
find_library(HWLOC_LIBRARIES hwloc PATHS ${HWLOC_DIR}/lib)
set(LIBS ${LIBS} ${HWLOC_LIBRARIES})

########################################
# Metis
set(METIS_DIR $ENV{METISDIR} CACHE PATH "Location of Metis Library")
find_library(METIS_LIBRARIES metis PATHS ${METIS_DIR})
set(LIBS ${LIBS} ${METIS_LIBRARIES})

########################################
# std C++ (needed for linking with ssids)
set(LIBS ${LIBS} stdc++)

########################################
# BLAS
set(LBLAS "" CACHE STRING "BLAS library")

# If LBLAS not set, use environement variable
if(LBLAS)
  set(BLAS_LIBRARIES ${LBLAS})
elseif(DEFINED ENV{BLAS_LIB})
  set(BLAS_LIBRARIES $ENV{BLAS_LIB})
endif()

if(DEFINED BLAS_LIBRARIES)
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  CHECK_FORTRAN_FUNCTION_EXISTS("dgemm" BLAS_FOUND)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT BLAS_FOUND)
    message(ERROR "User supplied BLAS is NOT working")
  endif()
else()
  find_package(BLAS)
endif()

if(BLAS_FOUND)
  set(LIBS ${LIBS} ${BLAS_LIBRARIES})
else()
  message(FATAL_ERROR "BLAS NOT found")
endif (BLAS_FOUND)

########################################
# LAPACK
set(LLAPACK "" CACHE STRING "LAPACK library")
# If LBLAS not set, use environement variable 
if(LLAPACK)
  set(LAPACK_LIBRARIES ${LLAPACK})
elseif(DEFINED ENV{LAPACK_LIB})
  set(LAPACK_LIBRARIES $ENV{LAPACK_LIB})
endif()
if(DEFINED LAPACK_LIBRARIES)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  CHECK_FORTRAN_FUNCTION_EXISTS("dpotrf" LAPACK_FOUND)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT LAPACK_FOUND)
    message(ERROR "User supplied LAPACK is NOT working")
  endif()

else()
  find_package(LAPACK)
endif()

if (LAPACK_FOUND)
  set(LIBS ${LIBS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
else()
  message(FATAL_ERROR "LAPACK library NOT found")
endif (LAPACK_FOUND)

# GPU
if(${USE_GPU} MATCHES ON)

  # Use GPU
  add_definitions(-DSPLDLT_USE_GPU)

  # set(CUDA_NVCC_FLAGS "-g -arch=compute_20 -code=compute_20,sm_20,sm_35" CACHE
  #   STRING "NVCC flags")

  find_package(CUDA REQUIRED)

  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-arch=compute_20,code=compute_20,sm_20,sm_35")
  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-arch=compute_20;-code=compute_20,sm_20,sm_35" CACHE STRING "NVCC flags")

  # set(CUDA_NVCC_FLAGS "TET" CACHE STRING "NVCC flags")
  list(APPEND CUDA_NVCC_FLAGS "-g")
  
  MESSAGE( STATUS "CUDA found:         " ${CUDA_FOUND} )
  MESSAGE( STATUS "CUDA host compiler: " ${CUDA_HOST_COMPILER} )
  MESSAGE( STATUS "CUDA nvcc compiler: " ${CUDA_NVCC_EXECUTABLE})
  MESSAGE( STATUS "NVCC FLAGS:         " ${CUDA_NVCC_FLAGS} )

  # enable_language(CUDA)
  # include(FindCUDA)

  # Allow for dynamic parallelism with CUBLAS
  # set(CUDA_SEPARABLE_COMPILATION TRUE)

  # MAGMA library
  set(MAGMA_DIR $ENV{MAGMADIR} CACHE PATH "Location of MAGMA Library")
  set(MAGMA_SRCDIR $ENV{MAGMASRCDIR} CACHE PATH "Location of MAGMA Sources")
  include_directories(${MAGMA_DIR}/include)
  include_directories(${MAGMA_SRCDIR}/control)     
  find_library(MAGMA_LIBRARY magma PATHS ${MAGMA_DIR}/lib)

endif()

# Include source directory
include_directories(${SpLDLT_SOURCE_DIR}/src)

########################################
# Runtime system

if (${RUNTIME} MATCHES "STF")

  add_definitions(-DSPLDLT_USE_STF)

elseif(${RUNTIME} MATCHES "StarPU")

  ########################################
  # StarPU

  add_definitions(-DSPLDLT_USE_STARPU)

  find_package(STARPU)
  if (STARPU_FOUND)
    include_directories( ${STARPU_INCLUDE_DIRS} )
    link_directories( ${STARPU_LIBRARY_DIRS} )
    link_libraries( ${STARPU_LIBRARIES} )
    set(LIBS ${LIBS} ${STARPU_LIBRARIES})
    find_path(STARPU_F_MOD_FOUND fstarpu_mod.f90
      HINTS ${STARPU_INCLUDE_DIRS}
      )
    set(STARPU_F_MOD "${STARPU_F_MOD_FOUND}/fstarpu_mod.f90")
    include_directories(${STARPU_INCLUDE_DIRS}) 
  else()
    message(FATAL_ERROR "StarPU not found!")
  endif(STARPU_FOUND)

endif()

add_subdirectory(src)

add_executable(spldlt_test drivers/spldlt_test.F90)
# add_executable(spldlt_test_debug drivers/spldlt_test_debug.F90)

## spldlt_test
target_include_directories(spldlt_test PUBLIC ${CMAKE_BINARY_DIR}/src)
target_link_libraries(spldlt_test spldlt)

## spldlt_test_debug
# target_include_directories(spldlt_test_debug PUBLIC ${CMAKE_BINARY_DIR}/src)
# target_link_libraries(spldlt_test_debug spldlt)

# Kernel test driver
if (${KERNEL_TEST} MATCHES "ON")

  file(
    GLOB_RECURSE
    test_src
    tests/*.cxx
    tests/*.hxx)
  
  # add_executable(spldlt_kernel_test tests/kernel_test.cxx)  
  add_executable(spldlt_kernel_test ${test_src})
  target_link_libraries(spldlt_kernel_test ${obj})
  target_link_libraries(spldlt_kernel_test spldlt)

endif()

## spral
target_include_directories(spldlt_test PUBLIC ${SPRAL_DIR})
# target_include_directories(spldlt_test_debug PUBLIC ${SPRAL_DIR})
find_library(SPRAL_LIBRARY spral PATHS ${SPRAL_DIR} ${SPRAL_DIR})
target_link_libraries(spldlt_test ${SPRAL_LIBRARY})
# target_link_libraries(spldlt_test_debug ${SPRAL_LIBRARY})

if (${KERNEL_TEST} MATCHES "ON")
  # MESSAGE( STATUS "spral directory:         " ${SPRAL_DIR})

  target_include_directories(spldlt_kernel_test PUBLIC ${SPRAL_DIR})
  target_link_libraries(spldlt_kernel_test ${SPRAL_LIBRARY})
endif()

################################################################################
# Test drivers

########################################
# Setup list of test drivers
set(test_drivers)

list(APPEND test_drivers "drivers/spldlt_test.F90")
# list(APPEND test_drivers "drivers/spldlt_test_debug.F90")

if (${KERNEL_TEST} MATCHES "ON")

  list(APPEND test_drivers "drivers/spldlt_kernel_test.F90")

endif()

foreach(test_driver ${test_drivers})
  
  get_filename_component(test_driver_name ${test_driver} NAME_WE)

  ########################################
  # Link test drivers with external libraries
  target_link_libraries(${test_driver_name} ${LIBS})

endforeach()

################################################################################
# Print summary

message("-----------------------------------------")
message("Configuration of SpLDLT done.")
message(" ")
message(
  "Build type: ${CMAKE_BUILD_TYPE}\n"
  "Fortran compiler: ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID})\n"
  "Fortran compiler flags: ${CMAKE_Fortran_FLAGS_ALL}\n"
  "C compiler:  ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID})\n"
  "C compiler flags: ${CMAKE_Fortran_FLAGS_ALL}\n"
  "CXX compiler:  ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID})\n"
  "CXX compiler flags: ${CMAKE_CXX_FLAGS_ALL}")
message(" ")
message("BLAS libraries: ${BLAS_LIBRARIES}")
message("LAPACK libraries: ${LAPACK_LIBRARIES}")
message(" ")
if(${RUNTIME} MATCHES "StarPU")
  message("Runtime system: StarPU")
  message("StarPU libraries: ${STARPU_LIBRARIES}")
endif()
message("-----------------------------------------")

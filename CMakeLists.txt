cmake_minimum_required (VERSION 2.6)
include(CheckFunctionExists)
include(CheckFortranFunctionExists)
project (SyLVER)
enable_language(Fortran)
enable_language(C)
enable_language(CXX)

# CMake version
message(STATUS "CMake version: ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}")

# Include our own cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

################################################################################
# Linker options

set(LINKER_LANGUAGE "Fortran")
set(CMAKE_Fortran_LINKER_PREFERENCE 50)

################################################################################
# Compiler options

########################################
# C flags
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2" )
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2 -ffpe-trap=underflow,denormal" )
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")

########################################
# CXX flags
# STD C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Note: Not supported feature for Intel compilers in CMake version < 3.6
# set(CMAKE_CXX_STANDARD 11) # C++11...
# set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
# set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -march=native")
# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=leak")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=bounds -fbounds-check")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fbounds-check")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ffpe-trap=underflow,denormal" )

########################################
# Fortran flags

set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELWITHDEBINFO} -march=native")
# Debug flags
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -Wall -Wextra -Warray-temporaries -Wconversion -fbacktrace -ffree-line-length-0")
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -ffpe-trap=zero,overflow,underflow")
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -fcheck=bounds,do,mem,pointer")
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -finit-real=nan")

########################################
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
MESSAGE( STATUS "Runtime: " ${RUNTIME} )

# # Set BLAS and LAPACK libraries to use for linking
# SET(BLAS_LAPACK "MKL" CACHE STRING "BLAS and LAPACK libraries")
# set_property(CACHE BLAS_LAPACK PROPERTY STRINGS Ref MKL)
# MESSAGE( STATUS "BLAS/LAPACK:         " ${BLAS_LAPACK} )

# Decide wheter we want to compile the kernel test driver or not
option(KERNEL_TESTS "Kernel test" OFF)
MESSAGE( STATUS "Kernel tests: " ${KERNEL_TESTS})

# spral
set(SPRAL_DIR $ENV{SPRALDIR} CACHE PATH "Location of Spral Library")
include_directories(${SPRAL_DIR})

set(SPRAL_SRC_DIR $ENV{SPRALSRCDIR} CACHE PATH "Location of Spral sources")
include_directories(${SPRAL_SRC_DIR})
include_directories(${SPRAL_SRC_DIR}/src)

# MESSAGE(STATUS "spral source dir: " ${SPRAL_SRC_DIR})

# Get object files 
# file(
#   GLOB_RECURSE
#   obj
#   ${SPRAL_DIR}/tests/ssids/kernels/*.o)

# MESSAGE(STATUS "object files: " ${obj})
# add_library(spral_ssids_tests STATIC IMPORTED ${obj})

# Use GPUs
option(USE_GPU "Use GPU" OFF)
MESSAGE(STATUS "Use GPU: " ${USE_GPU} )

# Use MPI
option(USE_MPI "Use MPI" OFF)
MESSAGE(STATUS "Use MPI: " ${USE_MPI} )   

# Profiling
option(USE_PROFILING "Activate profiling info" OFF)
MESSAGE(STATUS "Profiling: " ${USE_PROFILING})
if (${USE_PROFILING} MATCHES "ON")
  add_definitions(-DSPLDLT_USE_PROFILING)
endif()

# Use OpenMP
option(USE_OMP "Use OpenMP" OFF)
MESSAGE(STATUS "Use OpenMP: " ${USE_OMP} )

# Have la-hp scheduler
option(HAVE_LAHP "LA-heteroprio scheduler available" OFF)
if (${HAVE_LAHP} MATCHES "ON")
  add_definitions(-DHAVE_LAHP)
endif()
MESSAGE(STATUS "LAHP: " ${HAVE_LAHP})

# Have CUTALSS library
option(HAVE_CUTLASS "CUTLASS library available" OFF)
if (${HAVE_CUTLASS} MATCHES "ON")

  set(CUTLASS_DIR "" CACHE STRING "CUTLASS library")

  if(CUTLASS_DIR)
    set(CUTLASS_HEADERS ${CUTLASS_DIR})
  elseif(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_HEADERS $ENV{CUTLASS_DIR})
  elseif(DEFINED ENV{CUTLASSDIR})
    set(CUTLASS_HEADERS $ENV{CUTLASSDIR})
  endif()

  if(EXISTS ${CUTLASS_HEADERS})
    MESSAGE(STATUS "CUTLASS: " ${HAVE_CUTLASS})
    MESSAGE(STATUS "CUTLASS Directory: " ${CUTLASS_HEADERS})
  else()
    message(FATAL_ERROR "CUTLASS NOT found")
  endif()
  
  add_definitions(-DHAVE_CUTLASS)
endif()


################################################################################
# Low-level options

SET(MEMLAYOUT "1D" CACHE STRING "Memory layout")
set_property(CACHE MEMLAYOUT PROPERTY STRINGS 2D 1D)
MESSAGE( STATUS "Memory layout: " ${MEMLAYOUT})
if (${MEMLAYOUT} MATCHES "1D")
  add_definitions(-DMEMLAYOUT_1D)
elseif (${MEMLAYOUT} MATCHES "2D")
  add_definitions(-DMEMLAYOUT_2D)
endif()

################################################################################
# Third-party libraries

########################################
# HWLOC
set(HWLOC_DIR $ENV{HWLOCDIR} CACHE PATH "Location of HWLOC Library")
MESSAGE( STATUS "hwloc dir: " ${HWLOC_DIR})
include_directories(${HWLOC_DIR}/include)
find_library(HWLOC_LIBRARIES hwloc PATHS ${HWLOC_DIR}/lib NO_DEFAULT_PATH)
set(LIBS ${LIBS} ${HWLOC_LIBRARIES})
MESSAGE( STATUS "hwloc libs: " ${HWLOC_LIBRARIES})

########################################
# Metis
set(METIS_DIR $ENV{METISDIR} CACHE PATH "Location of Metis Library")
find_library(METIS_LIBRARIES metis PATHS ${METIS_DIR} NO_DEFAULT_PATH)
set(LIBS ${LIBS} ${METIS_LIBRARIES})
MESSAGE( STATUS "metis libs: " ${METIS_LIBRARIES})

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

########################################
# OpenMP
if (${USE_OMP} MATCHES ON)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    message(STATUS "OpenMP CXX Flags: " ${OpenMP_Fortran_FLAGS})
    # Linker flags
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_Fortran_FLAGS}")
    # set(LIBS ${LIBS} ${OpenMP_CXX_LIBRARIES})

    # Compiler flags
    list(APPEND CMAKE_Fortran_FLAGS_ALL ${OpenMP_Fortran_FLAGS})
    list(APPEND CMAKE_CXX_FLAGS_ALL ${OpenMP_Fortran_FLAGS})
    
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_Fortran_FLAGS}")

    add_definitions(-DSPLDLT_USE_OMP)
  else()
    message(FATAL_ERROR "OpenMP library NOT found")
  endif()
endif()

# GPU
if (${USE_GPU} MATCHES ON)

  enable_language(CUDA)

  # Use GPU
  add_definitions(-DSPLDLT_USE_GPU)

  # set(CUDA_NVCC_FLAGS "-g -arch=compute_20 -code=compute_20,sm_20,sm_35" CACHE
  #   STRING "NVCC flags")

  find_package(CUDA REQUIRED)

  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-arch=compute_20,code=compute_20,sm_20,sm_35")
  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-arch=compute_20;-code=compute_20,sm_20,sm_35" CACHE STRING "NVCC flags")

  # list(APPEND CUDA_NVCC_FLAGS "-g")
  list(APPEND CUDA_NVCC_FLAGS "-g -O2")
  # list(APPEND CUDA_NVCC_FLAGS "-g -O2 -arch=compute_60")
  # list(APPEND CUDA_NVCC_FLAGS "-g -O2 -arch=compute_70")

  set(SyLVER_NVCC_ARCHS "60;61;70;75" CACHE STRING "The SM architectures to build code for.")
  # set(SyLVER_NVCC_ARCHS "50;60;61;70;75" CACHE STRING "The SM architectures to build code for.")

  # Set NVCC arguments
  foreach(ARCH ${SyLVER_NVCC_ARCHS})
    # string(APPEND CUDA_NVCC_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endforeach()
  
  MESSAGE( STATUS "CUDA found:         " ${CUDA_FOUND} )
  MESSAGE( STATUS "CUDA host compiler: " ${CUDA_HOST_COMPILER} )
  MESSAGE( STATUS "CUDA nvcc compiler: " ${CUDA_NVCC_EXECUTABLE})
  MESSAGE( STATUS "NVCC FLAGS:         " ${CUDA_NVCC_FLAGS} )

  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  set(LIBS ${LIBS} ${CUDA_CUBLAS_LIBRARIES})
  set(LIBS ${LIBS} ${CUDA_cusolver_LIBRARY})

  set(LIBS ${LIBS} spldlt_cuda)

  # find_library(NVIDIA_"/usr/lib64/nvidia")

  # enable_language(CUDA)
  # include(FindCUDA)

  # Allow for dynamic parallelism with CUBLAS
  # set(CUDA_SEPARABLE_COMPILATION TRUE)

  # MAGMA library
  # set(MAGMA_DIR $ENV{MAGMADIR} CACHE PATH "Location of MAGMA Library")
  # set(MAGMA_SRCDIR $ENV{MAGMASRCDIR} CACHE PATH "Location of MAGMA Sources")
  # include_directories(${MAGMA_DIR}/include)
  # include_directories(${MAGMA_SRCDIR}/control)     
  # find_library(MAGMA_LIBRARY magma PATHS ${MAGMA_DIR}/lib)

endif()

# Include source directory
include_directories(${SyLVER_SOURCE_DIR}/src)

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
add_executable(splu_test drivers/splu_test.F90)

## spldlt_test
target_include_directories(spldlt_test PUBLIC ${CMAKE_BINARY_DIR}/src)
target_link_libraries(spldlt_test sylver)
## splu_test
target_include_directories(splu_test PUBLIC ${CMAKE_BINARY_DIR}/src)
target_link_libraries(splu_test sylver)

########################################
## spral
target_include_directories(spldlt_test PUBLIC ${SPRAL_DIR})
target_include_directories(splu_test PUBLIC ${SPRAL_DIR})
# target_include_directories(spldlt_test_debug PUBLIC ${SPRAL_DIR})
find_library(SPRAL_LIBRARIES spral PATHS ${SPRAL_DIR} ${SPRAL_DIR})
# set(LIBS ${LIBS} ${SPRAL_LIBRARIES})
message(STATUS "SPRAL_LIBRARIES: ${SPRAL_LIBRARIES}")

target_link_libraries(spldlt_test ${SPRAL_LIBRARIES})
# target_link_libraries(spldlt_test_debug ${SPRAL_LIBRARIES})
target_link_libraries(splu_test ${SPRAL_LIBRARIES})

# set_property(TARGET spldlt_test PROPERTY LINKER_LANGUAGE Fortran)

########################################
# CUTLASS
if (${HAVE_CUTLASS} MATCHES "ON")
  include_directories( ${CUTLASS_HEADERS} )
endif()

################################################################################
# Tests

add_subdirectory(tests)

########################################
# Kernel test drivers
# if (${KERNEL_TESTS} MATCHES "ON")

#   file(
#     GLOB
#     test_src
#     tests/*.cxx
#     tests/*.hxx
#     tests/*.F90)
#   add_library(spldlt_tests ${test_src})
#   set(TESTS_LIBS ${TESTS_LIBS} spldlt_tests)
  
#   if(${USE_GPU} MATCHES ON)
#     file(
#       GLOB
#       cuda_tests_sources 
#       tests/*.cu)
#     cuda_add_library(sylver_cuda_tests ${cuda_tests_sources})
#       set(TESTS_LIBS ${TESTS_LIBS} sylver_cuda_tests)
#   endif()
  
#   if(${USE_GPU} MATCHES ON)

#     add_executable(spldlt_tests_update_block_gpu tests/drivers/update_block_gpu_test.cxx)

#     target_include_directories(spldlt_tests_update_block_gpu PUBLIC "${SyLVER_SOURCE_DIR}/tests")
#     target_include_directories(spldlt_tests_update_block_gpu PUBLIC ${SPRAL_DIR})
#     target_link_libraries(spldlt_tests_update_block_gpu spldlt)
#     target_link_libraries(spldlt_tests_update_block_gpu ${obj})
#     target_link_libraries(spldlt_tests_update_block_gpu ${SPRAL_LIBRARIES})
#     target_link_libraries(spldlt_tests_update_block_gpu ${LIBS})
#     target_link_libraries(spldlt_tests_update_block_gpu ${TESTS_LIBS})

#     add_executable(sylver_tests_factor_gpu tests/drivers/factor_gpu_test.cxx)

#     target_include_directories(sylver_tests_factor_gpu PUBLIC ${SPRAL_DIR})
#     target_include_directories(sylver_tests_factor_gpu PUBLIC "${SyLVER_SOURCE_DIR}/tests")
#     target_link_libraries(sylver_tests_factor_gpu spldlt)
#     target_link_libraries(sylver_tests_factor_gpu ${obj})
#     target_link_libraries(sylver_tests_factor_gpu ${SPRAL_LIBRARIES})
#     target_link_libraries(sylver_tests_factor_gpu ${LIBS})
#     target_link_libraries(sylver_tests_factor_gpu ${TESTS_LIBS})
#     # target_link_libraries(sylver_tests_factor_gpu cusolver) # CuSOLVER

#     add_executable(sylver_tests_gemm_gpu tests/drivers/gemm_gpu_test.cxx)

#     target_include_directories(sylver_tests_gemm_gpu PUBLIC ${SPRAL_DIR})
#     target_include_directories(sylver_tests_gemm_gpu PUBLIC "${SyLVER_SOURCE_DIR}/tests")
#     if (${HAVE_CUTLASS} MATCHES "ON")
#       target_include_directories(sylver_tests_gemm_gpu PUBLIC ${CUTLASS_HEADERS})
#     endif()    
#     target_link_libraries(sylver_tests_gemm_gpu spldlt)
#     target_link_libraries(sylver_tests_gemm_gpu ${obj})
#     target_link_libraries(sylver_tests_gemm_gpu ${SPRAL_LIBRARIES})
#     target_link_libraries(sylver_tests_gemm_gpu ${LIBS})
#     target_link_libraries(sylver_tests_gemm_gpu ${TESTS_LIBS})
#     # target_link_libraries(sylver_tests_gemm_gpu cusolver) # CuSOLVER

#   endif()

#   add_executable(spldlt_tests_factor_node tests/drivers/factor_node_test.cxx)

#   target_include_directories(spldlt_tests_factor_node PUBLIC "${SyLVER_SOURCE_DIR}/tests")
#   target_include_directories(spldlt_tests_factor_node PUBLIC ${SPRAL_DIR})
#   target_link_libraries(spldlt_tests_factor_node spldlt)
#   target_link_libraries(spldlt_tests_factor_node ${obj})
#   target_link_libraries(spldlt_tests_factor_node ${SPRAL_LIBRARIES})
#   target_link_libraries(spldlt_tests_factor_node ${TESTS_LIBS})
#   target_link_libraries(spldlt_tests_factor_node ${LIBS})

#   add_executable(spldlt_tests_kernels tests/drivers/kernel_test.cxx)
  
#   target_include_directories(spldlt_tests_kernels PUBLIC ${SyLVER_SOURCE_DIR}/tests)
#   target_include_directories(spldlt_tests_kernels PUBLIC ${SPRAL_DIR})
#   target_link_libraries(spldlt_tests_kernels spldlt_tests)
#   target_link_libraries(spldlt_tests_kernels spldlt)
#   target_link_libraries(spldlt_tests_kernels ${obj})
#   target_link_libraries(spldlt_tests_kernels ${SPRAL_LIBRARIES})
#   target_link_libraries(spldlt_tests_kernels ${LIBS})
#   target_link_libraries(spldlt_tests_kernels ${TESTS_LIBS})

# endif()

########################################
# Setup list of test drivers
MESSAGE( STATUS "Libs: " ${LIBS} )
set(test_drivers)

list(APPEND test_drivers "drivers/spldlt_test.F90")
list(APPEND test_drivers "drivers/splu_test.F90")

foreach(test_driver ${test_drivers})
    
  get_filename_component(test_driver_name ${test_driver} NAME_WE)

  set_property(TARGET ${test_driver_name} PROPERTY LINKER_LANGUAGE Fortran)

  ########################################
  # Link test drivers with external libraries
  target_link_libraries(${test_driver_name} ${LIBS})

endforeach()

################################################################################
# Print summary

message("--------------------------------------------------------------------------------")
message("Configuration of SyLVER done.")
message(" ")
message(
  "Build type: ${CMAKE_BUILD_TYPE}\n"
  "Fortran compiler: ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID})\n"
  "Fortran compiler flags: ${CMAKE_Fortran_FLAGS_ALL}\n"
  "C compiler:  ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID})\n"
  "C compiler flags: ${CMAKE_Fortran_FLAGS_ALL}\n"
  "CXX compiler:  ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID})\n"
  "CXX compiler flags: ${CMAKE_CXX_FLAGS_ALL}\n"
  "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}\n"
  # "Linker lang: ${LINKER_LANGUAGE}"
  )
message(" ")
message("BLAS libraries: ${BLAS_LIBRARIES}")
message("LAPACK libraries: ${LAPACK_LIBRARIES}")
message(" ")
if(${RUNTIME} MATCHES "StarPU")
  message("Runtime system: StarPU")
  message("StarPU libraries: ${STARPU_LIBRARIES}")
  message(" ")
endif()
if(${USE_GPU} MATCHES ON)
  message(
    "CUDA host compiler: ${CUDA_HOST_COMPILER}\n"
    "CUDA libraries: ${CUDA_LIBRARIES}\n"
    "CUBLAS libraries: ${CUDA_CUBLAS_LIBRARIES}\n"
    "NVVC comiler: ${CUDA_NVCC_EXECUTABLE}\n"
    "NVCC FLAGS: ${CUDA_NVCC_FLAGS}\n"
    )
  message(" ")
endif()
# if (${USE_OMP} MATCHES ON)
#   message(
#     "OpenMP CXX libraries: ${OpenMP_CXX_LIBRARIES}\n"
#     "OpenMP Fortran libraries: ${OpenMP_Fortran_LIBRARIES}\n"
#     )
# endif()
message("--------------------------------------------------------------------------------")

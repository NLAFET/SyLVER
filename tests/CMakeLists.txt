include(${CMAKE_SOURCE_DIR}/cmake/create_test.cmake)

# Include modules files from SyLVER
include_directories(${CMAKE_BINARY_DIR}/src)

# Get object files from SSIDS tests 
file(
  GLOB_RECURSE
  obj
  ${SPRAL_DIR}/tests/ssids/kernels/*.o)

# Sources
file(
  GLOB
  tests_src
  *.cxx
  *.hxx
  *.F90)

if(SYLVER_ENABLE_CUDA)

  # Sources
  file(
    GLOB
    tests_cuda_src
    *.cu
    *.cuh)
endif()

if(SYLVER_ENABLE_CUDA)
  if(SYLVER_NATIVE_CUDA)
    add_library(
      sylver_tests
      ${tests_src}
      ${tests_cuda_src})
  else()
    add_library(
      sylver_tests
      ${tests_src})
    cuda_add_library(sylver_tests_cuda ${tests_cuda_src})
  endif()
else()
  add_library(
    sylver_tests
    ${tests_src})
endif()

sylver_tests_add_driver("drivers/factor_node_test.cxx")
# GPU
if(SYLVER_ENABLE_CUDA)
  sylver_tests_add_driver("drivers/update_block_gpu_test.cxx")
  sylver_tests_add_driver("drivers/factor_gpu_test.cxx")
endif()

sylver_create_test("drivers/kernel_test.cxx")
sylver_create_test("drivers/sylver_test.F90")

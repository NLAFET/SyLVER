# List tests drivers
set(tests_drivers)

# Include modules files from SyLVER
include_directories(${CMAKE_BINARY_DIR}/src)

# Unit tests
if (${KERNEL_TESTS} MATCHES "ON")
  
  list(APPEND tests_drivers "drivers/factor_node_test.cxx")
  list(APPEND tests_drivers "drivers/kernel_test.cxx")
  list(APPEND tests_drivers "drivers/sylver_test.F90")

  # GPU
  if(${USE_GPU} MATCHES ON)
    list(APPEND tests_drivers "drivers/update_block_gpu_test.cxx")
    list(APPEND tests_drivers "drivers/factor_gpu_test.cxx")
  endif()

  # Get object files from SSIDS tests 
  file(
    GLOB_RECURSE
    obj
    ${SPRAL_DIR}/tests/ssids/kernels/*.o)

endif()

# Sources
file(
  GLOB
  tests_src
  *.cxx
  *.hxx
  *.F90)

if(${USE_GPU} MATCHES ON)

  # Sources
  file(
    GLOB
    tests_cuda_src
    *.cu
    *.cuh)

  cuda_add_library(sylver_tests_cuda ${tests_cuda_src})

endif()

add_library(sylver_tests ${tests_src})
# if (${KERNEL_TESTS} MATCHES "ON")
#   set_target_properties(sylver_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# endif()
# set_target_properties(sylver_tests PROPERTIES POSITION_INDEPENDENT_CODE ON)

foreach(tests_driver ${tests_drivers})

  MESSAGE( STATUS "Tests driver: " ${tests_driver} )

  get_filename_component(tests_driver_name ${tests_driver} NAME_WE)

  add_executable(${tests_driver_name} ${tests_driver})

  target_include_directories(${tests_driver_name} PUBLIC ${SyLVER_SOURCE_DIR}/tests)
  target_include_directories(${tests_driver_name} PUBLIC ${SyLVER_DIR})
  target_include_directories(${tests_driver_name} PUBLIC ${CMAKE_BINARY_DIR}/src)
  target_include_directories(${tests_driver_name} PUBLIC ${CMAKE_BINARY_DIR}/tests)
  target_include_directories(${tests_driver_name} PUBLIC ${SPRAL_DIR})
  target_link_libraries(${tests_driver_name} sylver_tests)
  if(${USE_GPU} MATCHES ON)
    target_link_libraries(${tests_driver_name} sylver_tests_cuda)
  endif()
  target_link_libraries(${tests_driver_name} sylver)
  target_link_libraries(${tests_driver_name} ${obj})
  target_link_libraries(${tests_driver_name} ${SPRAL_LIBRARIES})
  target_link_libraries(${tests_driver_name} ${LIBS})

endforeach()

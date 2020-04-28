function(sylver_create_test tests_driver)

  get_filename_component(tests_driver_name ${tests_driver} NAME_WE)

  add_executable(${tests_driver_name} ${tests_driver})

  target_include_directories(${tests_driver_name} PUBLIC ${SyLVER_SOURCE_DIR}/tests)
  target_include_directories(${tests_driver_name} PUBLIC ${SyLVER_DIR})
  target_include_directories(${tests_driver_name} PUBLIC ${CMAKE_BINARY_DIR}/src)
  target_include_directories(${tests_driver_name} PUBLIC ${CMAKE_BINARY_DIR}/tests)
  target_include_directories(${tests_driver_name} PUBLIC ${SPRAL_DIR})
  target_link_libraries(${tests_driver_name} sylver_tests)
  if((SYLVER_ENABLE_CUDA) AND (NOT SYLVER_NATIVE_CUDA))
    target_link_libraries(${tests_driver_name} sylver_tests_cuda)
  endif()
  target_link_libraries(${tests_driver_name} sylver)
  target_link_libraries(${tests_driver_name} ${obj})
  target_link_libraries(${tests_driver_name} ${SPRAL_LIBRARIES})
  target_link_libraries(${tests_driver_name} ${LIBS})

  add_test(NAME ${tests_driver_name} COMMAND ${tests_driver_name})
  
endfunction()

function(sylver_tests_add_driver tests_driver)

  get_filename_component(tests_driver_name ${tests_driver} NAME_WE)

  add_executable(${tests_driver_name} ${tests_driver})

  target_include_directories(${tests_driver_name} PUBLIC ${SyLVER_SOURCE_DIR}/tests)
  target_include_directories(${tests_driver_name} PUBLIC ${SyLVER_DIR})
  target_include_directories(${tests_driver_name} PUBLIC ${CMAKE_BINARY_DIR}/src)
  target_include_directories(${tests_driver_name} PUBLIC ${CMAKE_BINARY_DIR}/tests)
  target_include_directories(${tests_driver_name} PUBLIC ${SPRAL_DIR})
  target_link_libraries(${tests_driver_name} sylver_tests)
  if((SYLVER_ENABLE_CUDA) AND (NOT SYLVER_NATIVE_CUDA))
    target_link_libraries(${tests_driver_name} sylver_tests_cuda)
  endif()
  target_link_libraries(${tests_driver_name} sylver)
  target_link_libraries(${tests_driver_name} ${obj})
  target_link_libraries(${tests_driver_name} ${SPRAL_LIBRARIES})
  target_link_libraries(${tests_driver_name} ${LIBS})
  
endfunction()

function(sylver_create_unit_test testfile)

  get_filename_component(testname ${testfile} NAME_WE)

  add_executable(${testname} ${testfile})

  target_link_libraries(${testname} gtest gmock gtest_main)
  target_link_libraries(${testname} sylver)
  target_link_libraries(${testname} ${LIBS})

  add_test(NAME ${testname} COMMAND ${testname})

endfunction()

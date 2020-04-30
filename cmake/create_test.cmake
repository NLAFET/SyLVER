function(sylver_create_test testfile)

  get_filename_component(testname ${testfile} NAME_WE)

  add_executable(${testname} ${testfile})

  target_include_directories(${testname} PUBLIC ${SyLVER_SOURCE_DIR}/tests)
  target_include_directories(${testname} PUBLIC ${SyLVER_DIR})
  target_include_directories(${testname} PUBLIC ${CMAKE_BINARY_DIR}/src)
  target_include_directories(${testname} PUBLIC ${CMAKE_BINARY_DIR}/tests)
  target_include_directories(${testname} PUBLIC ${SPRAL_DIR})
  target_include_directories(${testname} PUBLIC ${SyLVER_SOURCE_DIR}/include)

  target_link_libraries(${testname} sylver_tests)
  if((SYLVER_ENABLE_CUDA) AND (NOT SYLVER_NATIVE_CUDA))
    target_link_libraries(${testname} sylver_tests_cuda)
  endif()
  target_link_libraries(${testname} sylver)
  target_link_libraries(${testname} ${obj})
  target_link_libraries(${testname} ${SPRAL_LIBRARIES})
  target_link_libraries(${testname} ${LIBS})

  add_test(NAME ${testname} COMMAND ${testname})
  
endfunction()

function(sylver_tests_add_driver testfile)

  get_filename_component(testname ${testfile} NAME_WE)

  add_executable(${testname} ${testfile})

  target_include_directories(${testname} PUBLIC ${SyLVER_SOURCE_DIR}/tests)
  target_include_directories(${testname} PUBLIC ${SyLVER_DIR})
  target_include_directories(${testname} PUBLIC ${CMAKE_BINARY_DIR}/src)
  target_include_directories(${testname} PUBLIC ${CMAKE_BINARY_DIR}/tests)
  target_include_directories(${testname} PUBLIC ${SPRAL_DIR})
  target_include_directories(${testname} PUBLIC ${SyLVER_SOURCE_DIR}/include)

  target_link_libraries(${testname} sylver_tests)
  if((SYLVER_ENABLE_CUDA) AND (NOT SYLVER_NATIVE_CUDA))
    target_link_libraries(${testname} sylver_tests_cuda)
  endif()
  target_link_libraries(${testname} sylver)
  target_link_libraries(${testname} ${obj})
  target_link_libraries(${testname} ${SPRAL_LIBRARIES})
  target_link_libraries(${testname} ${LIBS})
  
endfunction()

function(sylver_create_unit_test testfile)

  get_filename_component(testname ${testfile} NAME_WE)

  add_executable(${testname} ${testfile})

  target_link_libraries(${testname} gtest gmock gtest_main)
  target_link_libraries(${testname} sylver)
  target_link_libraries(${testname} ${LIBS})

  add_test(NAME ${testname} COMMAND ${testname})

endfunction()
